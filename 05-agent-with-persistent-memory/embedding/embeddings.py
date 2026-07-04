from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def generate_embeddings(texts: str | list[str]) -> list[float]:

    """
    Generates embeddings for a list of texts.

    Arg:

    texts: List of texts to be converted into embeddings.

    Returns:

    List of embedding vectors, in the same order as the input texts.
    """

    embeddings = embedding_model.encode(
        sentences=texts,
        normalize_embeddings=True)
    
    return embeddings.tolist()