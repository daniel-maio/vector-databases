# --- Imports --- #

from client import pc_client

from config import *

# --- Program --- #


def retrieve_user_memories(
        user_id: str,
        query_embedding: list[float],
        top_k: int,
        min_score = 0.5
        ) -> list[dict]:

    index = pc_client.index(INDEX_NAME)

    filter = {
        "user_id": {"$eq": user_id}
    }

    results = index.query(
        vector=query_embedding,
        namespace=NAMESPACE_MEMORIES,
        top_k=top_k,
        filter=filter,
        include_values=False,
        include_metadata=True
    )

    matches = [{
        "id":res.get("id"),
        "score":round(res.get("score"),4),
        "user_id": res["metadata"]["user_id"],
        "timestamp": res["metadata"]["timestamp"],
        "agent": res['metadata']['agent'],
        "summary": res['metadata']['summary']
        }
        for res in results['matches']
        if res.get("score") > min_score
    ]

    return matches