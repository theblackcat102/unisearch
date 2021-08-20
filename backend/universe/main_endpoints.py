import sys, os
from sanic import Sanic
from sanic.blueprints import Blueprint
from sanic.response import json

from universe.settings import VECTOR_SIZE, SEARCH_TOP_K, SEARCH_PARAM, LOAD_MODEL_ENDPOINTS
from models.entries import Link, Image, link_milvus, img_milvus

search_route = Blueprint('search', url_prefix='search')


@search_route.post('/text/vector')
def search_by_text_vector(requests):
    context = requests.json

    if 'vector' in context:
        vector = context['vector']
        topk = SEARCH_TOP_K
        if 'topk' in context:
            topk = int(context['topk'])

        if len(vector) != VECTOR_SIZE:
            return json({"success": "fail", "msg": "vector size invalid"})

        status, results = link_milvus.search(
                        collection_name=Link.collection_name, 
                        query_records=[vector], top_k=topk, 
                        params=SEARCH_PARAM)
        milvus_ids = [ r.id for r in results[0]]
        milvus_id2distance = { r.id: r.distance   for r in results[0] }
        links = Link.select().where((Link.milvus_id.in_(milvus_ids)))
        links = [ (l, milvus_id2distance[l.milvus_id] ) for idx, l in enumerate(links) ]
        links = sorted(links, key=lambda x:x[1])

        links_json = [ {
            'title': l.title, 
            'description': l.description, 
            'url': l.url, 
            'lang': l.language  } for (l, dist) in links   ]
        return json({"success": "ok", "results": links_json})

    return json({"success": "fail", "msg": "vector not found"})


@search_route.post('/img/vector')
def search_by_img_vector(requests):
    context = requests.json

    if 'vector' in context:
        vector = context['vector']
        topk = SEARCH_TOP_K
        if 'topk' in context:
            topk = int(context['topk'])

        if len(vector) != VECTOR_SIZE:
            return json({"success": "fail", "msg": "vector size invalid"})

        status, results = img_milvus.search(
                        collection_name=Image.collection_name, 
                        query_records=[vector], top_k=topk, 
                        params=SEARCH_PARAM)
        milvus_ids = [ r.id for r in results[0]]
        milvus_id2distance = { r.id: r.distance   for r in results[0] }
        images = Image.select().where((Image.milvus_id.in_(milvus_ids)))
        images = [ (l, milvus_id2distance[l.milvus_id] ) for idx, l in enumerate(images) ]
        images = sorted(images, key=lambda x:x[1])
        images_json = [ {
            'image_hash': l.image_hash, 
            'img_url': l.img_url, 
            'url': l.url, 
            'dist': dist   } for (l, dist) in images   ]
        return json({"success": "ok", "results": images_json})

    return json({"success": "fail", "msg": "vector not found"})


