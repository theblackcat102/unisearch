from models.entries import Image, Link
from models.entries import postgres_database, img_milvus
import sys, os
from milvus import Milvus, IndexType, MetricType
import pandas as pd
from PIL import Image as PImage
from urllib.parse import urlparse
from mclip.models import build_multilingual_model
from universe.settings import MODEL_CHECKPOINT, VECTOR_SIZE
import torch
from tqdm import tqdm

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def reset_db():
    postgres_database.drop_tables([Image])
    img_milvus.drop_collection(collection_name=Image.collection_name)


if __name__ == '__main__':
    # reset_db()

    available_tables = postgres_database.get_tables()
    if 'image' not in available_tables:
        postgres_database.create_tables([Image])
        param = {'collection_name':Image.collection_name, 'dimension': 256, 'index_file_size':256, 'metric_type': MetricType.L2}
        img_milvus.create_collection(param)


    model, (tokenizer, img_transforms) = build_multilingual_model('RN50', checkpoint=MODEL_CHECKPOINT)

    file_list, image_dir, csv_file = sys.argv[1:4]
    valid_img_file = []
    with open(file_list, 'r') as f:
        for line in f:
            basename = os.path.basename(line.strip())
            valid_img_file.append(basename.replace('.jpg',''))
    valid_img_file.reverse()
    df = pd.read_csv(csv_file)
    keys = {}
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        domain = urlparse(row['photo_url']).netloc
        keys[row['keys']] = {
            'url': row['photo_url'],
            'img_url': row['photo_image_url'],
            'domain': domain,
            'image_hash': row['keys']+'.jpg',
            'caption': str(row['caption'])
        }
    model = model.cuda()
    batch_size = 32
    count = 0
    for img_chunk in tqdm(chunks(valid_img_file, batch_size), total=len(valid_img_file)//batch_size):
        pillow_img = []
        for img_file in img_chunk:
            if not Image.select(Image.milvus_id).where(Image.image_hash == img_file).exists():
                img_tensor = img_transforms(PImage.open(os.path.join(image_dir, img_file+'.jpg')).convert("RGB"))
                pillow_img.append(img_tensor)

        if len(pillow_img) > 0:
            tensor_imgs = torch.stack(pillow_img, 0).cuda()
            with torch.no_grad():
                img_encodings = model.encode_image(tensor_imgs)
            img_encodings = img_encodings.cpu()
            for encoding in img_encodings:
                img_file = valid_img_file[count]
                data = keys[img_file]
                img_encoding = encoding.numpy().tolist()
                img = Image.create(
                    url=data['url'],
                    img_url=data['img_url'],
                    caption=data['caption'][:120],
                    image_hash=data['image_hash'],
                    domain=data['domain'],
                )
                assert len(img_encoding) == VECTOR_SIZE
                img_milvus.insert(collection_name=Image.collection_name, records=[img_encoding], ids=[img.milvus_id])
                count += 1

