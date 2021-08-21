import os, sys
sys.path.append(os.path.join(os.getcwd(), 'backend'))

import unittest
from mclip.models import MCLIP
import torch

from torchvision.transforms import Compose
from torchvision import transforms

from transformers import AutoModel, AutoTokenizer

class SimilarityTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.text_model = AutoModel.from_pretrained('distilbert-base-multilingual-cased')
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')

        self.model = MCLIP(
            embed_dim=1024, # RN50: 1024, ViT: 512
            image_resolution=224,
            vision_layers=(3,4,6,3),
            vision_width=64,
            vision_patch_size=None,
            context_length=512, # not very important
            transformer_width=768,
            text_pretrain=self.text_model,
            text_embed_dim=256, # downsample
            include_visual=True
        )
        # self.img_transform = Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((224, 224)),
        #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ])

    # a simple quality test
    def test_forward(self):
        context = 'Hello world'
        text_tensor = self.tokenizer(context, truncation=True, return_tensors='pt', padding=True)
        # text_tensor = { k:tensor.cuda() for k, tensor in text_tensor.items() }
        text_encodings = self.model.encode_text2image_text(text_tensor)


        example_img = torch.randn((1,3,224, 224))
        img_encodings = self.model.encode_image(example_img)
        # assert both outputs is same
        self.assertTrue(img_encodings.shape[-1], text_encodings.shape[-1])


        self.model = MCLIP(
            embed_dim=768, # RN50: 1024, ViT: 512
            image_resolution=224,
            vision_layers=12,
            vision_width=64,
            vision_patch_size=32,
            context_length=512, # not very important
            transformer_width=768,
            text_pretrain='distilbert-base-multilingual-cased',
            text_embed_dim=256, # downsample
            include_visual=True
        )
        text_tensor = self.tokenizer(context, truncation=True, return_tensors='pt', padding=True)
        # text_tensor = { k:tensor.cuda() for k, tensor in text_tensor.items() }
        text_encodings = self.model.encode_text2image_text(text_tensor)


        example_img = torch.randn((1,3,224, 224))
        img_encodings = self.model.encode_image(example_img)
        # assert both outputs is same
        self.assertTrue(img_encodings.shape[-1], text_encodings.shape[-1])


    def test_train(self):
        context = 'Test Example'
        
        self.model = MCLIP(
            embed_dim=768, # RN50: 1024, ViT: 512
            image_resolution=224,
            vision_layers=12,
            vision_width=64,
            vision_patch_size=32,
            context_length=512, # not very important
            transformer_width=768,
            text_pretrain='distilbert-base-multilingual-cased',
            text_embed_dim=256, # downsample
            include_visual=True
        )
        text_tensor = self.tokenizer(context, truncation=True, return_tensors='pt', padding=True)
        # text_tensor = { k:tensor.cuda() for k, tensor in text_tensor.items() }
        example_img = torch.randn((1,3,224, 224))
        logits_per_image, logits_per_text = self.model(example_img, text_tensor, pair=('image', 'text'))
        self.assertTrue(logits_per_image.shape[-1], logits_per_text.shape[-1])
        logits_per_image, logits_per_text = self.model(example_img, text_tensor, pair=('image', 'text'))
        self.assertTrue(logits_per_image.shape[-1], logits_per_text.shape[-1])

