from models.entries import Link
from models.entries import postgres_database, link_milvus
import argparse
import requests
from urllib.parse import urlparse
from milvus import MetricType
from mclip.models import build_multilingual_model
from currentsapi import CurrentsAPI
import torch
from tqdm import tqdm
from universe.settings import CURRENTS_API_TOKEN


parser = argparse.ArgumentParser(description='Fetch and index data from news api')
parser.add_argument('--checkpoint', type=str, help='Pretrained multilingual clip')
parser.add_argument('--reset', default=False, action=argparse.BooleanOptionalAction, help="reset previous index results?")
args = parser.parse_args()


def reset_db():
    postgres_database.drop_tables([Link])
    link_milvus.drop_collection(collection_name=Link.collection_name)


def index_results(country, language, category, total_pages ):
    api = CurrentsAPI(api_key=CURRENTS_API_TOKEN)

    for page_num in tqdm(range(1, total_pages)):
        try:
            results = api.search(country=country, 
                language=language, 
                category=category, 
                page_number=page_num)

            batch_context = []
            batch_links = []
            if len(results['news']) > 0:
                for row in results['news']:
                    domain = urlparse(row['url']).netloc
                    title = row['title'][:500]
                    description = row['description'][:1900]
                    url = row['url']
                    if len(url) < 2048 and not Link.select(Link.milvus_id).where(Link.url == url).exists():
                        context = row['title'] +' '+row['description']
                        link = Link.create(
                            title=title,
                            description=description,
                            domain=domain,
                            language=language if len(row['language']) >= 5 else row['language'],
                            url=url
                        )
                        assert Link.select(Link.milvus_id).where(Link.url == url).exists()
                        batch_links.append(link)
                        batch_context.append(context)
                if len(batch_context) > 0:
                    with torch.no_grad():
                        text_tensor = tokenizer(batch_context, truncation=True, return_tensors='pt', padding=True)
                        text_tensor = { k:tensor.cuda() for k, tensor in text_tensor.items() }

                        encodings = model.encode_text2text_text(text_tensor)

                    text_encodings = encodings.numpy().tolist()
                    link_milvus.insert(collection_name=Link.collection_name, records=text_encodings, ids=[link.milvus_id for link in batch_links])
            else:
                break
        except requests.exceptions.ReadTimeout:
            continue

if __name__ == '__main__':
    # reset_db()


    CHECKPOINT = args.checkpoint
    model, (tokenizer, img_transforms) = build_multilingual_model('RN50', checkpoint=CHECKPOINT)
    model = model.cuda()

    available_tables = postgres_database.get_tables()
    if 'link' not in available_tables:
        postgres_database.create_tables([Link])
        param = {'collection_name':Link.collection_name, 
            'dimension': 256, 
            'index_file_size':256, 
            'metric_type': MetricType.L2
        }
        link_milvus.create_collection(param)


    REGIONS = [
        ("zh", "TW"),
        ("en", "IN"), ("en", "CN"),("en", "INT"),("en", "PH"), ("en","SA"), ("en","TH"),
        ("zh", "HK"), ("zh", "CN"),("es","ES"), ("de", "AT"),  ("fr", "BE"), ("en", "IR"),
        ('en', 'US'), ('en', 'SG'), ('en', 'GB'),  ("en", "EU"),
        ('ru', 'RU'), ('it', 'IT'), ('en', 'IT'), ('en', 'DE'),  ('ja', 'JP'), ('ko', "SK"),
        ('de', 'DE'), ('fr', 'FR'), ('en', 'DE'), ('vi', "VI"),  ('msa', 'MY'),
        ('en', 'CA'), ('en', 'AU'), ('en', 'NZ'), ('en', 'JP'), ('en', 'FR'), ('en', 'RU'),
    ]
    categories = [
            'national', 'all', 'technology', 'business',
            'auto', 'celebrity', 'estate', 'fashion','medical', 'design', 'music', 'energy',
            'entertainment', 'general', 'politics', 'finance', 'travel']
    total_pages = 2
    for (language, country) in REGIONS:
        print(language, country)
        for category in categories:
            print(category)
            index_results(country, language=language, category=category, total_pages=total_pages)
