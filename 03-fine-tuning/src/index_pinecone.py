# --- Imports --- #

import os
import sys
import time

# Add root dir to the first index in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import logging

import numpy as np

from tqdm import tqdm

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL_PATH,
    KNOWLEDGE_BASE_PATH
)

from sentence_transformers import SentenceTransformer

from pinecone import Pinecone, ServerlessSpec, IndexModel

# --- Logger --- #
logging.basicConfig(level = logging.INFO, format="%(message)s")

logger = logging.getLogger(__name__)

def main():
    logger.info(f"Loading Fine-Tuned Embedding Model...")
    print("-" * 50)

    embedding_model = SentenceTransformer(model_name_or_path=os.path.join(EMBEDDING_MODEL_PATH, "best-model"))

    EMBED_DIM = embedding_model.get_sentence_embedding_dimension()

    print("-" * 50)
    logger.info(f"Connecting to Pinecone vector-database...")
    print("-" * 50)

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if not pc.has_index(PINECONE_INDEX_NAME):

        logger.info(f"Creating Pinecone Index: {PINECONE_INDEX_NAME}")
        print("-" * 50)

        INDEX = pc.indexes.create(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            vector_type="dense",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

        while True:
            if INDEX.status.ready:
                break
            time.sleep(2)

        logger.info(f"Pinecone Index: {PINECONE_INDEX_NAME} created.")
        print("-" * 50)
    
    else:
        logger.info(f"Pinecone Index: {PINECONE_INDEX_NAME} already exists.")
        print("-" * 50)

    # Initialize index client
    index_client = pc.index(name=PINECONE_INDEX_NAME)

    # Load Data
    logger.info(f"Loading Data...")
    print("-" * 50)

    with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = [d['content'] for d in data]
    
    logger.info(f"{len(texts)} documents loaded.")
    print("-" * 50)

    logger.info(f"Generating embeddings...")
    print("-" * 50)
    
    embeddings = embedding_model.encode(
        sentences=texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    records=[]

    for vector, metadata in zip(embeddings, data):
        records.append(
            {
                'id': metadata.get('id'),
                'values': vector.tolist(),
                'metadata': {
                    "title": metadata.get('title'),
                    "content": metadata.get('content'),
                    "category": metadata.get('category')
                }
            }
        )
    
    batch_size = int(len(texts) / 4)

    NAMESPACE = 'finance-sentences'

    for start in tqdm(range(0, len(data), batch_size), "Upserting records batch:"):
        index_client.upsert(
            vectors=records[start: start+batch_size],
            namespace=NAMESPACE
        )

    status = index_client.describe_index_stats()
    print("-" * 50)

    logger.info(f"Index Status: {status}")
    print("-" * 50)
    
    logger.info(f"{status.get('total_vector_count')} documents inserted.")
    print("-" * 50)

    logger.info('End of script')


if __name__ == "__main__":
    main()