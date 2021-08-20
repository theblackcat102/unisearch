import json
from models.entries import Link
from models.entries import postgres_database, link_milvus

from fastlangid.langid import LID
langid = LID()


if __name__ == '__main__':
    search_param = {'nprobe': 16}
    with open('example_text.json') as f:
        example_text = json.load(f)

    for query, latent_vector in example_text.items():

        language = langid.predict(query)
        # if 'zh' == language[:2]:
        #     language = 'zh'

        status, results = link_milvus.search(collection_name=Link.collection_name, 
                                query_records=[latent_vector], top_k=50, 
                                params=search_param)
        milvus_ids = [ r.id for r in results[0] ]
        distances = [r.distance for r in results[0]]
        milvus_id2distance = { r.id: r.distance   for r in results[0] }
        
        print(query, sum(distances)/len(distances))
        links = Link.select(Link.title, Link.milvus_id, Link.language).where((Link.milvus_id.in_(milvus_ids)))
        links = [ (l, milvus_id2distance[l.milvus_id] ) for idx, l in enumerate(links) ]
        links = sorted(links, key=lambda x:x[1])
        for idx, (link, dist) in enumerate(links[:10]):
            print(idx, link.title, dist)
