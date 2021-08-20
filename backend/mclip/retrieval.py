from models.entries import Link, Image, link_milvus, img_milvus
from universe.settings import SEARCH_TOP_K, SEARCH_PARAM
from .functs import text2text_retrieve_encodings, \
    img2text_retrieve_encoding_by_img, \
    img2text_retrieve_encoding_by_text

def text2text_retrieve(context, topk=SEARCH_TOP_K):

    text_encoding = text2text_retrieve_encodings(context)

    status, results = link_milvus.search(
                    collection_name=Link.collection_name, 
                    query_records=[text_encoding], top_k=topk, 
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
        'dist': dist,
        'lang': l.language  
    } for (l, dist) in links   ]

    return links_json

def img2text_retrieve_by_img(image, topk=SEARCH_TOP_K):

    img_encoding = img2text_retrieve_encoding_by_img(image)

    status, results = img_milvus.search(
                    collection_name=Image.collection_name, 
                    query_records=[img_encoding], top_k=topk, 
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

    return images_json

def img2text_retrieve_by_text(query, topk=SEARCH_TOP_K):
    img_encoding = img2text_retrieve_encoding_by_text(query)

    status, results = img_milvus.search(
                    collection_name=Image.collection_name, 
                    query_records=[img_encoding], top_k=topk, 
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

    return images_json
