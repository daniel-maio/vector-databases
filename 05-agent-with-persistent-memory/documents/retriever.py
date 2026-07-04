# --- Imports --- #

from config import *

from client import pc_client

# --- Program --- #

def retrieve_documents(
        query_embedding:list[float],
        filter_type:str,
        top_k:int,
        min_score = 0.5
    ) ->list[dict]:
    
    index = pc_client.index(INDEX_NAME)
    
    filter = {
        "type": {"$eq": filter_type}
    }

    results = index.query(
        vector=query_embedding,
        namespace=NAMESPACE_DOCUMENTS,
        top_k=top_k,
        filter=filter,
        include_values=False,
        include_metadata=True
    )

    matches = [{
        "id":res.get("id"),
        "score":round(res.get("score"),4),
        "title": res["metadata"]["title"],
        "category": res['metadata']['category'],
        "content": res['metadata']['content'],
        "type": res['metadata']['type']
        }
        for res in results['matches']
        if res.get("score") > min_score
    ]

    return matches