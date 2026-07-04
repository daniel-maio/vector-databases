# --- Imports --- #

import os
import sys

from groq import Groq

from pinecone import Pinecone

from sentence_transformers import SentenceTransformer

# Add root dir to the first index in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    EMBEDDING_MODEL_PATH,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    GROK_API_KEY,
    GROQ_MODEL
)

# --- Example Questions --- #
example_questions = [
    "What is EBITDA and how to calculate EBITDA margin?",
    "How does discounted cash flow valuation work?",
    "What are the main SaaS metrics?",
    "Explain working capital management.",
    "What are financial covenants?",
    "What are the pricing strategies for products?",
    "How does the M&A process work?"
]

## --- Flow --- #

# User ask a question: What is EBITDA?

# Call the embedding_model and transform it into an embedding.

# Perform a query_search in Pinecone with this embedding (top_1).

# Pinecone returns the content.

# I take the content and pass it to LLM as the context.

# LLM reads the context and generates the response.

# --- Variables --- #

model_path = os.path.join(EMBEDDING_MODEL_PATH, "best-model")

embedding_model = SentenceTransformer(model_name_or_path=model_path)
pc_client = Pinecone(api_key=PINECONE_API_KEY)
groq_client = Groq(api_key=GROK_API_KEY)

# --- Main Program --- #

def generate_response(user_query):

    query_embedding = embedding_model.encode(
        sentences = user_query,
        normalize_embeddings=True).tolist()

    index = pc_client.index(name=PINECONE_INDEX_NAME)

    results = index.query(
        top_k=3,
        vector=query_embedding,
        namespace='finance-sentences',
        include_metadata=True,
        include_values=False
    )

    matches = [
        { 
            "id": res.get("id"),
            "category": res.metadata.get("category"),
            "title": res.metadata.get("title"),
            "score": round(res.get("score"), 4)
        }
        for res in results.get("matches", [])
    ]

    print(f"Retrived documents:\n{matches}")
    
    context = results['matches'][0]['metadata']['content']

    system_prompt = f"""
    You are a specialist assistant in business language, corporate finance, and business management.

    Rules:
    1. Answer only with the provided context.
    2. If context is insufficient, say so.
    3. Use technical terminology.
    4. Answer in English.

    Context:
    {context}
    """

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )

    response = response.choices[0].message.content
    print(response)
    
    return

# --- Ask a finance-related question --- #
user_query = "How does the M&A process work?"

if __name__ == "__main__":
    generate_response(user_query)


