from milvus import Milvus, IndexType, MetricType, Status
from universe.settings import POSTGRESQL_SETTINGS, MILVUS_IMG_SETTINGS, MILVUS_TEXT_SETTINGS


link_milvus = Milvus(host=MILVUS_TEXT_SETTINGS['HOST'], port=str(MILVUS_TEXT_SETTINGS['PORT']))

img_milvus = Milvus(host=MILVUS_IMG_SETTINGS['HOST'], port=str(MILVUS_IMG_SETTINGS['PORT']))


postgres_database = peewee_asyncext.PooledPostgresqlExtDatabase(
    POSTGRESQL_SETTINGS['DATABASE'],
    user=POSTGRESQL_SETTINGS['USER'],
    host=POSTGRESQL_SETTINGS['HOST'],
    port=POSTGRESQL_SETTINGS['PORT'],
    password=POSTGRESQL_SETTINGS['PASSWORD'],
    register_hstore=False,
    max_connections=10)
