from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def get_embedding(self, text):
        return self.model.encode(text).tolist()