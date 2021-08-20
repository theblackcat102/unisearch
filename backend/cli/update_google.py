import os, glob
from dotenv import load_dotenv
import json
from urllib.parse import urlparse
from fastlangid.langid import LID
import requests
from tqdm import tqdm
langid = LID()

load_dotenv(verbose=True)

from serpapi import GoogleSearch

serapi_key = os.getenv('SERPAPI_KEY')

INFERENCE_ENDPOINT = 'http://127.0.0.1:5001'

if __name__ == '__main__':
    if serapi_key is None:
        exit(0)

    os.makedirs('google_results', exist_ok=True) 

    parse_log_filename = 'parsed_google_keywords.txt'
    if not os.path.exists(parse_log_filename):
        with open(parse_log_filename, 'w') as f:
            f.write('')
    if not os.path.exists('serapi_parse.txt'):
        with open('serapi_parse.txt', 'w') as f:
            f.write('')

    parsed_keywords = []
    with open(parse_log_filename, 'r') as f:
        for line in f:
            parsed_keywords.append(line.strip())
    parsed_keywords = set(parsed_keywords)

    with open('universe.log', 'r') as f, open('serapi_parse.txt', 'a') as g:
        for line in f:
            if ':=:=' in line:
                query = line.strip().split(':=:=', 1)[-1]
                output_file = os.path.join('google_results', query.replace('\\','-' ).replace(' ', '_')+'.json')
                if os.path.exists(output_file):
                    continue

                g.write(query+'\n')
                search = GoogleSearch({
                    "q": query, 
                    "location": "California, United States",
                    "api_key": serapi_key
                })
                result = search.get_dict()
                with open(output_file, 'w') as f:
                    json.dump(result, f)
    
    found_results = [ filename for filename in glob.glob('google_results/*.json') if filename not in parsed_keywords]
    for filename in tqdm(found_results):
        with open(filename, 'r') as f:
            result = json.load(f)
        if 'organic_results' in result:
            for r in result['organic_results']:
                if 'snippet' not in r:
                    continue
                title = r['title']
                description = r['snippet']
                url = r['link']
                domain = domain = urlparse(url).netloc
                language = langid.predict(title+' '+description)
                if '-' in language:
                    language = language.split('-')[0]
                add_url = INFERENCE_ENDPOINT+'/inference/text/add'
                payload={
                    'url': url,
                    'title': title,
                    'description': description[:2000],
                    'domain': domain,
                    'language': language
                }
                response = requests.request("POST", add_url, data=payload)

            with open(parse_log_filename, 'a') as f:
                f.write(filename+'\n')