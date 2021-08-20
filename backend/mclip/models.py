from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from torchvision import transforms
from torch import nn
from clip.model import ModifiedResNet, VisualTransformer




class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

def freeze_transformer_layers(model, target_layer_num):
    '''
        Freeze transformers before [layer_num]-th layer
    '''
    for name, params in model.named_parameters():
        if 'embeddings' in name:
            params.requires_grad = False
        elif '.layer.' in name:
            layer_num = int(name.split('.layer.', 1)[1].split('.', 1)[0])
            if layer_num < target_layer_num:
                params.requires_grad = False
            #     print('freeze', name)
            # else:
            #     print('not freeze' ,name)
    return model



def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32    
    
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    return image_resolution, vision_layers, vision_width, vision_patch_size, \
        embed_dim, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers


class CLIPTransformer(nn.Module):

    def __init__(self, 
            embed_dim: int,
            context_length: int,
            vocab_size: int,
            transformer_width: int,
            transformer_heads: int,
            transformer_layers: int,
            down_embed_dim: int):
        super().__init__()

        from clip.model import Transformer
        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.text_projection_stage2 = nn.Linear(embed_dim, down_embed_dim, bias=False)

    def load_pretrain_state(self, pretrained_state_dict):
        state_dict = self.state_dict()
        for key, tensor in pretrained_state_dict.items():
            if key in state_dict:
                state_dict[key] = tensor
        self.load_state_dict(state_dict)

    @property
    def dtype(self):
        return self.text_projection_stage2.weight.dtype

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return self.text_projection_stage2(x)


class MCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 transformer_width: int,
                 text_pretrain: Union[str, nn.Module ],
                 text_embed_dim: int,
                 include_visual: bool = True
        ):
        super().__init__()

        self.context_length = context_length
        self.visual = None
        self.task_embed = nn.Embedding(10, text_embed_dim)
        if include_visual:
            if isinstance(vision_layers, (tuple, list)):
                vision_heads = vision_width * 32 // 64
                self.visual = ModifiedResNet(
                    layers=vision_layers,
                    output_dim=embed_dim,
                    heads=vision_heads,
                    input_resolution=image_resolution,
                    width=vision_width
                )
            else:
                vision_heads = vision_width // 64
                self.visual = VisualTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim
                )

            self.img_proj = False
            if text_embed_dim != embed_dim:
                self.img_proj = True
                self.img_projection = nn.Linear(embed_dim, text_embed_dim, bias=False)

        if isinstance(text_pretrain, str):
            self.transformer = AutoModel.from_pretrained(text_pretrain)
        else:
            self.transformer = text_pretrain
        self.transformer_width = transformer_width
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Linear(transformer_width, text_embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        if self.text_projection is not None:
            for p in self.text_projection.parameters():
                torch.nn.init.normal_(p, std=self.transformer_width ** -0.5)


    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        latent = self.visual(image.type(self.dtype))
        if self.img_proj:
            return self.img_projection(latent)
        return latent

    def encode_text(self, text):
        outputs = self.transformer(**text)
        pooled_out = self.ln_final( outputs[0][:, 0] )
        return self.text_projection(pooled_out)

    def forward(self, input1, input2, pair=('image', 'text'), task_id=0):

        if pair[0] == 'image':
            feature1 = self.encode_image(input1)
        else:
            feature1 = self.encode_text(input1)

        if pair[1] == 'image':
            feature2 = self.encode_image(input2)
        else:
            feature2 = self.encode_text(input2)

        task_id_idx = torch.ones( (feature1.shape[0],),dtype=torch.long, device=feature1.device ) * task_id
        text_latent_offset = self.task_embed( task_id_idx )
        if pair[0] == 'text':
            feature1 += text_latent_offset
        task_id_idx = torch.ones( (feature2.shape[0],),dtype=torch.long, device=feature2.device ) * task_id
        text_latent_offset = self.task_embed( task_id_idx )

        if pair[1] == 'text':
            feature2 += text_latent_offset

        # normalized features
        feature1 = feature1 / feature1.norm(dim=-1, keepdim=True)
        feature2 = feature2 / feature2.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * feature1 @ feature2.t()
        logits_per_text = logit_scale * feature2 @ feature1.t()

        return logits_per_image, logits_per_text



    def encode_text2text_text(self, text):

        encodings = self.encode_text(text)
        task_id_idx = torch.ones( (encodings.shape[0],),dtype=torch.long, device=encodings.device )
        text_latent_offset = self.task_embed( task_id_idx )
        encodings += text_latent_offset
        if encodings.is_cuda:
            encodings = encodings.cpu()
        return encodings 


    def encode_text2image_text(self, text):
        encodings = self.encode_text(text)
        task_id_idx = torch.zeros( (encodings.shape[0],),dtype=torch.long, device=encodings.device )
        text_latent_offset = self.task_embed( task_id_idx )
        encodings += text_latent_offset
        if encodings.is_cuda:
            encodings = encodings.cpu()
        return encodings 


def build_multilingual_model(clip_name='RN50', tokenizer='distilbert-base-multilingual-cased', checkpoint=None, include_visual=True):
    import clip
    from transformers import AutoModel, AutoTokenizer

    # load only vision encoder
    model, _ = clip.load(clip_name, device='cpu')
    ( image_resolution, vision_layers, vision_width, vision_patch_size,
            embed_dim, context_length, vocab_size, transformer_width, 
            transformer_heads, transformer_layers) = build_model(model.state_dict())
    # load tokenizer and language model from huggingface ( the base of mutilingual encoder )
    text_model = AutoModel.from_pretrained(tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # image-text encoder for multilingual
    new_clip = MCLIP(
        embed_dim=1024, # RN50: 1024, ViT: 512
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=512, # not very important
        transformer_width=768,
        text_pretrain=text_model,
        text_embed_dim=256, # downsample
        include_visual=include_visual
    )

    if checkpoint is not None:
        print('load checkpoint')
        state_dict = torch.load(checkpoint, map_location='cpu')
        new_clip.load_state_dict(state_dict, strict=False)

    img_transform = Compose([
        transforms.ToTensor(),
        transforms.Resize((image_resolution, image_resolution)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    new_clip.eval()
    return new_clip, (tokenizer, img_transform)