import os
from dotenv import load_dotenv

# currents/.env
load_dotenv(verbose=True)

# only True string can trigger debug
DEBUG = True if os.getenv('DEBUG') == 'True' else False

SEARCH_PARAM = {'nprobe': 16}

VECTOR_SIZE = 256

SEARCH_TOP_K = 50

LOAD_MODEL_ENDPOINTS = True

POSTGRESQL_SETTINGS = {
    'DATABASE': os.getenv('POSTGRES_DATABASE'),
    'USER': os.getenv('POSTGRES_USER'),
    'HOST': os.getenv('POSTGRES_HOST'),
    'PORT': os.getenv('POSTGRES_PORT'),
    'PASSWORD': os.getenv('POSTGRES_SECRE')
}


MILVUS_IMG_SETTINGS = {
    'HOST': os.getenv('MILVUS_IMG_HOST'),
    'PORT': os.getenv('MILVUS_IMG_PORT'),
    'DATABASE': os.getenv('MILVUS_IMG_DATABASE'),
}

MILVUS_TEXT_SETTINGS = {
    'HOST': os.getenv('MILVUS_TEXT_HOST'),
    'PORT': os.getenv('MILVUS_TEXT_PORT'),
    'DATABASE': os.getenv('MILVUS_TEXT_DATABASE'),
}


CURRENTS_API_TOKEN = os.getenv('CURRENTS_API_TOKEN')


MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT')