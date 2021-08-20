from sanic import Sanic
from sanic.blueprints import Blueprint
from sanic.response import json
from models.entries import Link, Image, link_milvus
import logging
from PIL import Image
import numpy as np
import torch
from urllib.parse import urlparse
from universe.settings import SEARCH_TOP_K
from mclip.functs import (
    text2text_retrieve_encodings,
    img2text_retrieve_encoding_by_img,
    img2text_retrieve_encoding_by_text,
    model, tokenizer
)
from mclip.retrieval import text2text_retrieve, img2text_retrieve_by_img, img2text_retrieve_by_text
logging.basicConfig(format='%(asctime)s:%(message)s',filename='query.log',level=logging.INFO)



model_routes = Blueprint('inference', url_prefix='inference')

@model_routes.post('/text/add')
def add_link_entry(request):
    inputs = request.form
    url = inputs.get('url')
    language = inputs.get('language')
    title = inputs.get('title')
    description = inputs.get('description')
    domain = inputs.get('domain')
    
    exists = Link.select(Link.milvus_id).where(Link.url == url).exists()
    if exists:
        return json({ "received": True,  "added": False, "id": None})

    link = Link.create(url=url, 
        language=language, 
        description=description, 
        domain=domain, 
        title=title
    )
    context = title+' '+description
    with torch.no_grad():
        text_tensor = tokenizer(context, truncation=True, return_tensors='pt', padding=True)
        # text_tensor = { k:tensor.cuda() for k, tensor in text_tensor.items() }
        encodings = model.encode_text2text_text(text_tensor)

    text_encoding = encodings[0].numpy().tolist()
    link_milvus.insert(collection_name=Link.collection_name, records=[text_encoding], ids=[link.milvus_id])

    return json({ "received": True,  "added": True, "id": link.milvus_id})



@model_routes.post('/text/encode/text')
def encode_text2text(request):
    inputs = request.form
    context = inputs.get('context')
    text_encoding = text2text_retrieve_encodings(context)
    return json({ "context": context,  "encoding": text_encoding})


@model_routes.post('/text/end2end/text')
def end2end_text2text(request):
    inputs = request.form
    context = inputs.get('context')
    topk = SEARCH_TOP_K
    if 'topk' in context:
        topk = int(context['topk'])

    links_json = text2text_retrieve(context, topk)
    logging.info('=:={}'.format( context ))
    return json({"success": "ok", "results": links_json})



@model_routes.post('/img/encode/text')
def encode_image2text_text(request):
    inputs = request.form
    context = inputs.get('context')

    text_encoding = img2text_retrieve_encoding_by_text(context)
    return json({ "context": context,  "encoding": text_encoding})


@model_routes.post('/img/encode/img')
def encode_image2text_image(request):
    img_raw =  request.files["image"][0].body
    
    image = Image.open(io.BytesIO(img_raw))
    image = np.array(image)
    if len(image.shape) == 2:
        w, h  = image.shape
        image = image.reshape(w, h, 1)
    w, h, c = image.shape
    if c == 1:
        image = np.repeat(image, 3, axis=2)
        c = 3
    image = image[:, :, :3]

    img_encoding = img2text_retrieve_encoding_by_img(image)
    return json({  "encoding": img_encoding})


@model_routes.post('/img/end2end/text')
def end2end_img2text(request):
    inputs = request.form
    context = inputs.get('context')
    topk = SEARCH_TOP_K
    if 'topk' in context:
        topk = int(context['topk'])

    img_json = img2text_retrieve_by_text(context, topk)
    return json({"success": "ok", "results": img_json})


@model_routes.post('/img/end2end/img')
def end2end_img2text_img(request):
    img_raw =  request.files["image"][0].body
    
    image = Image.open(io.BytesIO(img_raw))
    image = np.array(image)
    if len(image.shape) == 2:
        w, h  = image.shape
        image = image.reshape(w, h, 1)
    w, h, c = image.shape
    if c == 1:
        image = np.repeat(image, 3, axis=2)
        c = 3
    image = image[:, :, :3]

    img_json = img2text_retrieve_by_img(image)
    return json({"success": "ok", "results": img_json})