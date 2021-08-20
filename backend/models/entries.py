import sys
sys.path.append("..") # Adds higher directory to python modules path.
import asyncio
import logging
from peewee import *
import peewee_asyncext
from playhouse.postgres_ext import *
from universe.settings import POSTGRESQL_SETTINGS, MILVUS_IMG_SETTINGS, MILVUS_TEXT_SETTINGS
from datetime import datetime
from milvus import Milvus, IndexType, MetricType, Status


postgres_database = peewee_asyncext.PooledPostgresqlExtDatabase(
    POSTGRESQL_SETTINGS['DATABASE'],
    user=POSTGRESQL_SETTINGS['USER'],
    host=POSTGRESQL_SETTINGS['HOST'],
    port=POSTGRESQL_SETTINGS['PORT'],
    password=POSTGRESQL_SETTINGS['PASSWORD'],
    register_hstore=False,
    max_connections=10)

class BaseModel(Model):
    class Meta:
        database = postgres_database


img_milvus = Milvus(host=MILVUS_IMG_SETTINGS['HOST'], port=str(MILVUS_IMG_SETTINGS['PORT']))

class Link(BaseModel):

    collection_name = MILVUS_IMG_SETTINGS['DATABASE']
    language = CharField(max_length=5)
    title = CharField(max_length=500)
    description = CharField(max_length=2000)
    url = CharField(max_length=2048, unique=True)
    domain = CharField(max_length=80)
    milvus_id = BigAutoField(primary_key=True)

link_milvus = Milvus(host=MILVUS_TEXT_SETTINGS['HOST'], port=str(MILVUS_TEXT_SETTINGS['PORT']))

class Image(BaseModel):
    collection_name = MILVUS_TEXT_SETTINGS['DATABASE']

    caption = CharField(max_length=300, null=True, default = None)
    url = CharField(max_length=2048)
    img_url = CharField(max_length=2048, unique=True)
    domain = CharField(max_length=80)
    milvus_id = BigAutoField(primary_key=True)
    image_hash = CharField(max_length=300)

