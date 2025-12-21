from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from uuid import uuid4
from typing import List, Dict

class VectorService:
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = "documents"
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        try:
            self.client.get_collection(collection_name=self.collection_name)
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
    
    def store_vector(self, vector: List[float], user_id: str, text: str, metadata: Dict = {}) -> str:
        vector_id = str(uuid4())
        payload = {"user_id": user_id, "text": text, **metadata}
        point = PointStruct(id=vector_id, vector=vector, payload=payload)
        self.client.upsert(collection_name=self.collection_name, points=[point])
        return vector_id
    
    def search_vectors(self, query_vector: List[float], user_id: str, limit: int = 10) -> List[Dict]:
        try:
            # Try newer API first
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                ),
                limit=limit
            )
        except AttributeError:
            # Fallback to query_points
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                ),
                limit=limit
            ).points
        
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]

vector_service = VectorService()