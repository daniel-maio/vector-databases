# --- Imports --- #

from client import pc_client

from config import *


# --- Program --- #


def upsert_vector(vector):

    index_client = pc_client.index(INDEX_NAME)

    index_client.upsert(
        vectors=[vector],
        namespace=NAMESPACE_MEMORIES,
        show_progress=True
    )

    return