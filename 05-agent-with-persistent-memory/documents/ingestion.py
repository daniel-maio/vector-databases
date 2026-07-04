# --- Imports --- #

from pinecone import ServerlessSpec, Vector

from documents.data_loader import load_data
from documents.format_doc import format_doc

from embedding.embeddings import generate_embeddings

from config import *

from client import pc_client

# --- Program --- #

def prepare_vectors(documents):

    text_for_emb = [format_doc(doc) for doc in documents]

    embeddings = generate_embeddings(text_for_emb)

    vectors = []

    for doc, embedding in zip(documents, embeddings):
        vectors.append(Vector(
            id= doc['id'],
            values=[embedding],
            metadata = {
                "title": doc['title'],
                "category": doc['category'],
                "content": doc["content"],
                "type": doc["type"]
            }
        ))

    return vectors

def create_index(INDEX_NAME):
    
    if not pc_client.indexes.exists(INDEX_NAME):
        pc_client.indexes.create(
            name=INDEX_NAME,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
            dimension=DIM,
            metric="cosine",
            vector_type="dense"
        )

    index_client = pc_client.index(name=INDEX_NAME)
       
    return

def upsert_vectors(vectors):

    index_client = pc_client.index(INDEX_NAME)

    batch_size = int(len(vectors) / 4)

    response = index_client.upsert(
        vectors=vectors,
        namespace=NAMESPACE_DOCUMENTS,
        batch_size=batch_size,
        show_progress=True
    )

    return


if __name__ == "__main__":
    documents = load_data()
    vectors = prepare_vectors(documents)
    create_index(INDEX_NAME)
    upsert_vectors(vectors)
    
