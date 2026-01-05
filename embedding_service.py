# from sentence_transformers import SentenceTransformer
# from typing import List

# class EmbeddingService:
#     def __init__(self):
#         # Use this model instead (smaller, more reliable)
#         self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
#     def generate_embedding(self, text: str) -> List[float]:
#         embedding = self.model.encode(text)
#         return embedding.tolist()

# embedding_service = EmbeddingService()


from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    def __init__(self):
        # Arabic-optimized model
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')  # ← NEW MODEL
    
    def generate_embedding(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

embedding_service = EmbeddingService()