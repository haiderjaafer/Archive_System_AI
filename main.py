# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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








class HybridSearchQuery(BaseModel):
    user_id: str
    query: str
    limit: int = 10
    min_score: float = 0.7
    boost_exact_match: bool = True

@app.post("/api/hybrid-search", response_model=List[SearchResult])
async def hybrid_search(query: HybridSearchQuery):
    """Combines vector search with keyword boosting"""
    try:
        # Vector search
        query_vector = embedding_service.generate_embedding(query.query)
        # print(f"query_vector... ${query_vector}")
        results = vector_service.search_vectors(query_vector, query.user_id, query.limit * 2)
        # print(f"results... ${results}")
        # Boost exact keyword matches
        if query.boost_exact_match:
            keywords = query.query.split()
            print(f"keywords... ${keywords}")
            for r in results:
                text = r["payload"]["text"]
                print(f"texts... ${text}")
                # Boost score if any keyword appears in text
                
                exact_matches = sum(1 for kw in keywords if kw in text)
                print(f"exact_matches... ${exact_matches}")
                if exact_matches > 0:
                    r["score"] = r["score"] + (exact_matches * 0.1)  # Boost by 0.1 per match
        
        # Filter and sort
        filtered_results = [r for r in results if r["score"] >= query.min_score]
        print(f"filtered_results.1.. ${filtered_results}")

        filtered_results.sort(key=lambda x: x["score"], reverse=True)
        print(f"filtered_results.2.. ${filtered_results}")

        filtered_results = filtered_results[:query.limit]
        print(f"filtered_results.3.. ${filtered_results}")

        return [SearchResult(
            id=r["id"], 
            text=r["payload"]["text"], 
            score=r["score"], 
            metadata={k: v for k, v in r["payload"].items() if k not in ["user_id", "text"]}
        ) for r in filtered_results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.delete("/api/admin/reset-collection")
async def reset_collection():
    """Delete and recreate collection - USE CAREFULLY!"""
    try:
        vector_service.client.delete_collection(vector_service.collection_name)
        vector_service._ensure_collection_exists()
        return {"message": "Collection deleted and recreated with new dimensions"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


        
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
    
      
