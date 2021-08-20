# Univec Backend

Setup Milvux vector similarity search

MySQL database for Milvus 

```
docker pull mysql:5.7
docker run -p 3306:3306 -e  MYSQL_ROOT_PASSWORD=<Your Password> -d mysql:5.7
```

```
docker pull milvusdb/milvus:1.1.0-cpu-d050721-5e559c
docker run -d --name milvus_cpu_1.1.0 \
    -p 19530:19530 \
    -p 19121:19121 \
    -v /home/$USER/milvus/db:/var/lib/milvus/db \
    -v /home/$USER/milvus/conf:/var/lib/milvus/conf \
    -v /home/$USER/milvus/logs:/var/lib/milvus/logs \
    -v /home/$USER/milvus/wal:/var/lib/milvus/wal \
    milvusdb/milvus:1.1.0-cpu-d050721-5e559c
```


## Populate text search data


