# schemas.py
from pydantic import BaseModel
from typing import Optional, List

class TextToStore(BaseModel):
    user_id: str
    text: str
    metadata: Optional[dict] = {}

class SearchQuery(BaseModel):
    user_id: str
    query: str
    limit: int = 10

class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict

class StoreResponse(BaseModel):
    success: bool
    message: str
    vector_id: Optional[str] = None