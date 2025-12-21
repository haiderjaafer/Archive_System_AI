# main.py
from fastapi import FastAPI, HTTPException
from schemas import TextToStore, SearchQuery, SearchResult, StoreResponse
from embedding_service import embedding_service
from vector_service import vector_service
from typing import List

app = FastAPI()

@app.post("/api/store-text", response_model=StoreResponse)
async def store_text(data: TextToStore):
    try:
        vector = embedding_service.generate_embedding(data.text)
        vector_id = vector_service.store_vector(vector, data.user_id, data.text, data.metadata)
        return StoreResponse(success=True, message="Stored successfully", vector_id=vector_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search", response_model=List[SearchResult])
async def search_text(query: SearchQuery):
    try:
        query_vector = embedding_service.generate_embedding(query.query)
        results = vector_service.search_vectors(query_vector, query.user_id, query.limit)
        return [SearchResult(id=r["id"], text=r["payload"]["text"], score=r["score"], 
                metadata={k: v for k, v in r["payload"].items() if k not in ["user_id", "text"]}) 
                for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))