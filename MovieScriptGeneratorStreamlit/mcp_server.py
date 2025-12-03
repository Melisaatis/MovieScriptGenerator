#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server for Movie Script Generator
FastAPI backend that integrates with the Streamlit frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add parent directory to import langchain+mistral
sys.path.append(str(Path(__file__).parent.parent))

import json
import chromadb
from chromadb.utils import embedding_functions
import requests


# FASTAPI APP

app = FastAPI(
    title="Movie Script Generator MCP Server",
    description="Backend API for script generation with RAG + Mistral",
    version="1.0.0"
)
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# CONFIGURATION

SCRIPTS_ROOT = Path("../GenAIMovie/scripts")
DB_PATH = Path("../chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
MISTRAL_MODEL = "mistral"
RETRIEVAL_K = 6

# Global state
actors_db = {}
collection = None


# PYDANTIC MODELS

class Actor(BaseModel):
    name: str
    age: int
    gender: str
    specialty: Optional[str] = ""

class ScriptRequest(BaseModel):
    title: str
    query: str
    genre: str
    actors: List[Dict]
    use_rag: bool = True

class ScriptResponse(BaseModel):
    screenplay: str
    sources: List[str]
    title: str
    genre: str


# HELPER FUNCTIONS

def initialize_rag():
    """Initialize ChromaDB collection"""
    global collection
    try:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        collection = client.get_or_create_collection(
            name="movie_scripts",
            embedding_function=embedding_fn
        )
        return True
    except Exception as e:
        print(f"RAG initialization error: {e}")
        return False

def call_mistral(prompt: str) -> str:
    """Call local Mistral via Ollama"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MISTRAL_MODEL, "prompt": prompt, "stream": False},
            timeout=600
        )
        data = response.json()
        return data.get("response", data.get("output", str(data)))
    except Exception as e:
        return f"[ERROR] {e}"

def retrieve_context(query: str, genre: str) -> tuple:
    """Retrieve relevant context from vector DB"""
    if not collection:
        return [], []
    
    where = {"genre": genre.lower().replace(" ", "")} if genre else None
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=RETRIEVAL_K,
            where=where
        )
        
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        
        context_parts = []
        sources = []
        
        for d, m in zip(docs, metas):
            src = f"{m['title']} ({m['year']})"
            context_parts.append(f"--- {src} ---\n{d}")
            sources.append(src)
        
        return context_parts, sources
    except Exception as e:
        print(f"Retrieval error: {e}")
        return [], []


# STARTUP

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    print("🚀 Starting MCP Server...")
    if initialize_rag():
        print(f"✅ RAG initialized with {collection.count()} chunks")
    else:
        print("⚠️ RAG initialization failed")


# ENDPOINTS

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Movie Script Generator MCP Server",
        "rag_status": "initialized" if collection else "not initialized",
        "actors_count": len(actors_db)
    }

@app.post("/actors")
async def add_actor(actor: Actor):
    """Add an actor to the database"""
    actors_db[actor.name] = {
        "age": actor.age,
        "gender": actor.gender,
        "specialty": actor.specialty
    }
    return {"message": f"Actor {actor.name} added successfully", "actor": actor}

@app.get("/actors")
async def get_actors():
    """Get all actors"""
    return actors_db

@app.delete("/actors/{actor_name}")
async def delete_actor(actor_name: str):
    """Delete an actor"""
    if actor_name in actors_db:
        del actors_db[actor_name]
        return {"message": f"Actor {actor_name} deleted"}
    raise HTTPException(status_code=404, detail="Actor not found")

@app.post("/generate-script", response_model=ScriptResponse)
async def generate_script(request: ScriptRequest):
    """Generate a movie script using RAG + Mistral"""
    
    if not collection:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    # Build cast list
    cast_list = ""
    if request.actors:
        cast_list = "\n\nCAST:\n" + "\n".join(
            [f"- {a['name']} (Age: {a['age']}, {a['gender']})" for a in request.actors]
        )
    
    # Retrieve context if RAG enabled
    sources = []
    if request.use_rag:
        context_parts, sources = retrieve_context(request.query, request.genre)
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""
You are a master screenwriter. Write an original movie script inspired by these reference scenes.

TITLE: {request.title}
USER REQUEST: {request.query}
GENRE: {request.genre}{cast_list}

--- REFERENCE SCENES FROM CLASSIC FILMS ---
{context_text}
--- END REFERENCES ---

Write a 3-5 page original screenplay in proper Hollywood script format.
Include:
- Scene headings (INT./EXT.)
- Character names in CAPS before dialogue
- Action descriptions
- Proper formatting

Begin with the title page and opening scenes:
""".strip()
    else:
        prompt = f"""
Write a {request.genre} movie script titled '{request.title}'.

Story: {request.query}{cast_list}

Write in proper Hollywood script format with scene headings, character names in CAPS, and action descriptions.
""".strip()
    
    # Generate screenplay
    screenplay = call_mistral(prompt)
    
    return ScriptResponse(
        screenplay=screenplay,
        sources=sources,
        title=request.title,
        genre=request.genre
    )

@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    return {
        "status": "healthy",
        "rag_initialized": collection is not None,
        "ollama_available": check_ollama()
    }

def check_ollama() -> bool:
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


# RUN SERVER
if __name__ == "__main__":
    import uvicorn
    print("Starting Movie Script Generator MCP Server...")
    print("Server will run at http://127.0.0.1:8000")
    print("API docs at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
