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

# @app.post("/api/search", response_model=List[SearchResult])
# async def search_text(query: SearchQuery):
#     try:
#         query_vector = embedding_service.generate_embedding(query.query)
#         results = vector_service.search_vectors(query_vector, query.user_id, query.limit)
#         return [SearchResult(id=r["id"], text=r["payload"]["text"], score=r["score"], 
#                 metadata={k: v for k, v in r["payload"].items() if k not in ["user_id", "text"]}) 
#                 for r in results]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search", response_model=List[SearchResult])
async def search_text(query: SearchQuery):
    try:
        query_vector = embedding_service.generate_embedding(query.query)
        results = vector_service.search_vectors(query_vector, query.user_id, query.limit)
        
        # Filter by minimum score (0.7 = 70% similarity)
        filtered_results = [r for r in results if r["score"] >= 0.7]
        # filtered_results = [r for r in results if r["score"] >= 0.8]
        
        return [SearchResult(
            id=r["id"], 
            text=r["payload"]["text"], 
            score=r["score"], 
            metadata={k: v for k, v in r["payload"].items() if k not in ["user_id", "text"]}
        ) for r in filtered_results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/debug/test-vector")
async def test_vector(data: dict):
    """Test vector generation"""
    text = data.get("text", "")
    vector = embedding_service.generate_embedding(text)
    return {
        "text": text,
        "vector_length": len(vector),
        "first_10_values": vector[:10],
        "model": str(embedding_service.model)
    }





        
 # create .env file -> environment variable
   #python -m venv .venv

   # activate before running 
    #source .venv/Scripts/activate

    #py run.py

    # Freeze installed packages into requirements.txt and create file requirements.txt 
    #This command will overwrite your requirements.txt with all installed packages and their exact versions.
    
     #pip freeze > requirements.txt


    # install on another machine
    #   pip install -r requirements.txt


    #pip show passlib bcrypt
    # show all libraries installed
    # pip list 


    #uvicorn main:app --reload --port 8000
    #python -m uvicorn main:app --port 8000
      
