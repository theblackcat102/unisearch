import torch
from urllib.parse import urlparse
from mclip.models import build_multilingual_model
from universe.settings import MODEL_CHECKPOINT
from functools import lru_cache
from .utils import np_cache

if MODEL_CHECKPOINT is not None:
    model, (tokenizer, img_transforms) = build_multilingual_model('RN50', 
        checkpoint=MODEL_CHECKPOINT)
else:
    tokenizer, model, img_transforms = None, None, None

@lru_cache
def text2text_retrieve_encodings(context):
    with torch.no_grad():
        text_tensor = tokenizer(context, truncation=True, return_tensors='pt', padding=True)
        # text_tensor = { k:tensor.cuda() for k, tensor in text_tensor.items() }
        encodings = model.encode_text2text_text(text_tensor)

    text_encoding = encodings[0].numpy().tolist()
    return text_encoding



@np_cache(maxsize=256)
def img2text_retrieve_encoding_by_img(image):
    img_tensor = img_transforms(image).unsqueeze(0)
    with torch.no_grad():
        encodings = model.encode_image(img_tensor)
    img_encoding = encodings[0].numpy().tolist()
    return img_encoding




@lru_cache
def img2text_retrieve_encoding_by_text(context):
    with torch.no_grad():
        text_tensor = tokenizer(context, truncation=True, return_tensors='pt', padding=True)
        # text_tensor = { k:tensor.cuda() for k, tensor in text_tensor.items() }
        encodings = model.encode_text2image_text(text_tensor)

    text_encoding = encodings[0].numpy().tolist()
    return text_encoding

