import os
import json
from PIL import Image
from sentence_transformers import SentenceTransformer
from pymilvus import DataType
from pymilvus import MilvusClient

import transformers
transformers.logging.set_verbosity_error()

###################

encoder = SentenceTransformer('clip-ViT-B-32')

collection_name = 'catalog'

dimension = 512

####################

def generate_embedding(image_path):
    image = Image.open(image_path)
    embedding = encoder.encode(image, normalize_embeddings=True).tolist()
    return embedding


def get_client():

    client = MilvusClient(uri= "http://localhost:19530")
    return client


def create_collection(client, collection_name, dimension):

    schema = client.create_schema(auto_id=True)

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="name", datatype=DataType.VARCHAR, max_length=200)
    schema.add_field(field_name="price", datatype=DataType.FLOAT)
    schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="vector",
        index_name="vector_index",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        nlist=128
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )


def insert_data(client, collection_name, encoder):
    
    with open('products/products.json', 'r') as json_file:
        products = json.load(json_file)

    valid_products = [
        p for p in products
        if os.path.exists(p['image_path'])
    ]

    images = [
        Image.open(p['image_path'])
        for p in valid_products
    ]

    embeddings = encoder.encode(images, normalize_embeddings=True)

    for p, emb in zip(valid_products, embeddings):
        p['vector'] = emb.tolist()
      
    client.insert(
        collection_name,
        data = valid_products
    )
    
    client.flush(collection_name=collection_name)


if __name__ == "__main__":
    client = get_client()

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        create_collection(client, collection_name, dimension)

    insert_data(client, collection_name, encoder)