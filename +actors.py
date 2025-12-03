#!/usr/bin/env python3
"""
RAG Movie Script Generator – ULTRA-LIGHT CPU-ONLY + LOCAL LLAMA3 (OLLAMA)
Chunk → Embed → Retrieve → Build prompt → Generate screenplay with Llama 3
"""

import json
import re
import time
import uuid
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st

# ------------------------------------------------------------------- #
# TMDB CONFIG
# ------------------------------------------------------------------- #
TMDB_KEY = st.secrets.get("TMDB_KEY", "")

def fetch_actor_tmdb(actor_name: str) -> dict:
    """Fetch actor info from TMDB API and save JSON profile."""
    search_url = f"https://api.themoviedb.org/3/search/person?api_key={TMDB_KEY}&query={actor_name}"
    result = requests.get(search_url).json()

    if not result["results"]:
        raise Exception(f"Actor '{actor_name}' not found on TMDB")

    actor = result["results"][0]
    actor_id = actor["id"]

    details_url = f"https://api.themoviedb.org/3/person/{actor_id}?api_key={TMDB_KEY}&append_to_response=movie_credits"
    details = requests.get(details_url).json()

    known_roles = [m["title"] for m in details.get("movie_credits", {}).get("cast", [])[:10] if "title" in m]

    actor_data = {
        "name": details.get("name", actor_name),
        "biography": details.get("biography", ""),
        "known_for": known_roles,
        "traits": [],
        "physical_description": "",
        "typical_roles": ""
    }

    os.makedirs("actors", exist_ok=True)
    file_path = f"actors/{actor_name.lower().replace(' ', '_')}.json"
    with open(file_path, "w") as f:
        json.dump(actor_data, f, indent=4)

    print(f"Saved actor profile to: {file_path}")
    return actor_data

# ------------------------------------------------------------------- #
# ANSI COLORS
# ------------------------------------------------------------------- #
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def now(): return time.strftime("%H:%M:%S")

# ------------------------------------------------------------------- #
# LOCAL LLAMA 3 (OLLAMA) CALL
# ------------------------------------------------------------------- #
def call_llama_local(prompt: str, model: str = "llama3:8b") -> str:
    """Call local Llama3 via Ollama API."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=600  # 10 minutes
        )
        try:
            data = response.json()
        except Exception:
            return response.text

        if "response" in data:
            return data["response"]
        if "output" in data:
            return data["output"]
        return str(data)
    except Exception as e:
        return f"[ERROR calling local Llama3] {e}"

# ------------------------------------------------------------------- #
# CONFIG
# ------------------------------------------------------------------- #
SCRIPTS_ROOT = Path("GenAIMovie/scripts")
MANIFEST_PATH = SCRIPTS_ROOT / "dataset_manifest.json"
DB_PATH = Path("chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 6

# ------------------------------------------------------------------- #
# 1. LOAD MANIFEST
# ------------------------------------------------------------------- #
def load_manifest() -> List[Dict]:
    print(f"{bcolors.OKBLUE}[1/5] {now()} | Loading manifest...{bcolors.ENDC}")
    print(f"       Looking in: {MANIFEST_PATH}")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{bcolors.WARNING}Manifest not found at:\n{MANIFEST_PATH}{bcolors.ENDC}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    valid = [e for e in manifest if e.get("script_path") and e.get("source") != "NotFound"]
    print(f"{bcolors.OKGREEN}       → {len(valid)} valid script entries{bcolors.ENDC}")
    return valid

# ------------------------------------------------------------------- #
# 2. CHUNKING UTILITIES
# ------------------------------------------------------------------- #
def chunk_text(text: str) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

scene_regex = re.compile(r"(?im)^(INT\.|EXT\.|INT/EXT\.)[^\n]*$")

def chunk_script_with_scenes(text: str, title: str, year: str, genre: str) -> List[Dict]:
    text = text.replace('\r\n', '\n').replace('\r', '\n').strip()
    if not text:
        return []

    scenes_with_headers = []
    last_index = 0
    for match in scene_regex.finditer(text):
        start, end = match.span()
        if start > last_index:
            snippet = text[last_index:start].strip()
            if snippet:
                scenes_with_headers.append(f"SCENE {len(scenes_with_headers)+1}\n{snippet}")
        header = match.group().strip()
        scenes_with_headers.append(header)
        last_index = end
    remaining = text[last_index:].strip()
    if remaining:
        scenes_with_headers.append(f"SCENE {len(scenes_with_headers)+1}\n{remaining}")

    if not scenes_with_headers:
        for idx, snippet in enumerate(text.split('\n\n')):
            snippet = snippet.strip()
            if snippet:
                scenes_with_headers.append(f"SCENE {idx+1}\n{snippet}")

    chunks = []
    current_block = ""
    for scene in scenes_with_headers:
        if current_block:
            current_block += "\n\n" + scene
        else:
            current_block = scene

        if len(current_block.split()) >= CHUNK_SIZE:
            for sub in chunk_text(current_block):
                chunks.append({
                    "text": sub,
                    "metadata": {"title": title, "year": year, "genre": genre.lower().replace(" ", "")}
                })
            current_block = ""

    if current_block.strip():
        for sub in chunk_text(current_block):
            chunks.append({
                "text": sub,
                "metadata": {"title": title, "year": year, "genre": genre.lower().replace(" ", "")}
            })

    return chunks

# ------------------------------------------------------------------- #
# 3. BUILD DOCUMENTS
# ------------------------------------------------------------------- #
def build_documents(manifest: List[Dict]) -> List[Dict]:
    print(f"{bcolors.OKBLUE}[2/5] {now()} | Chunking scripts...{bcolors.ENDC}")
    docs = []
    for i, entry in enumerate(manifest, 1):
        script_file = SCRIPTS_ROOT / entry["script_path"]
        if not script_file.exists():
            print(f"{bcolors.WARNING}       → Missing: {script_file}{bcolors.ENDC}")
            continue

        raw = script_file.read_text(encoding="utf-8", errors="ignore")
        if len(raw) < 500:
            print(f"{bcolors.WARNING}       → Skip {entry['title']} (too short){bcolors.ENDC}")
            continue

        chunks = chunk_script_with_scenes(raw, entry["title"], entry["year"], entry["genre"])
        for chunk in chunks:
            docs.append({"id": str(uuid.uuid4()), "text": chunk["text"], "metadata": chunk["metadata"]})

        print(f"{bcolors.OKCYAN}       → [{i:3}/{len(manifest)}] {entry['title']} → {len(chunks)} chunks{bcolors.ENDC}")

    print(f"{bcolors.OKGREEN}       → TOTAL: {len(docs):,} chunks{bcolors.ENDC}")
    return docs

# ------------------------------------------------------------------- #
# 4. VECTOR DB
# ------------------------------------------------------------------- #
def build_or_load_db(docs: List[Dict]):
    print(f"{bcolors.OKBLUE}[3/5] {now()} | Initializing vector DB...{bcolors.ENDC}")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    col = client.get_or_create_collection(name="movie_scripts", embedding_function=embedding_fn)

    if col.count() > 0:
        print(f"{bcolors.OKGREEN}       → Loaded existing DB with {col.count()} items{bcolors.ENDC}")
        return col

    print(f"{bcolors.WARNING}       → Building DB... adding {len(docs)} chunks{bcolors.ENDC}")
    col.add(documents=[d["text"] for d in docs],
            metadatas=[d["metadata"] for d in docs],
            ids=[d["id"] for d in docs])
    print(f"{bcolors.OKGREEN}       → DB build complete{bcolors.ENDC}")
    return col

# ------------------------------------------------------------------- #
# 5. RETRIEVAL + PROMPT
# ------------------------------------------------------------------- #
def retrieve_and_build_prompt(col, query: str, genre: Optional[str] = None, actor_profile: Optional[dict] = None) -> Dict[str, Any]:
    print(f"{bcolors.OKBLUE}[4/5] {now()} | Retrieving...{bcolors.ENDC}")

    where = {"genre": genre.lower().replace(" ", "")} if genre else None

    results = col.query(
        query_texts=[query],
        n_results=RETRIEVAL_K,
        where=where
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    parts = []
    sources = []

    for d, m in zip(docs, metas):
        src = f"{m['title']} ({m['year']})"
        parts.append(f"--- {src} ---\n{d}")
        sources.append(src)

    # Deduplicate sources while keeping order
    seen = set()
    unique_sources = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            unique_sources.append(s)

    print(f"{bcolors.OKCYAN}       → Retrieved from: {', '.join(unique_sources)}{bcolors.ENDC}")

    parts_text = "\n\n".join(parts)

    actor_block = ""
    if actor_profile:
        actor_block = f"""
MAIN CHARACTER INSPIRATION (do NOT name the real actor in the script):
- In the style of: {actor_profile['name']}
- Traits: {', '.join(actor_profile['traits'])}
- Physical presence: {actor_profile['physical_description']}
- Typical roles: {actor_profile['typical_roles']}
"""

    prompt = f"""
You are a Hollywood screenwriter. Write a completely ORIGINAL screenplay.

===== YOUR EXACT ASSIGNMENT =====
Story: {query}
Genre: {genre or 'Any'}
{actor_block}
===================================

ABSOLUTELY CRITICAL RULES:
1. Write ONLY about: {query}
2. DO NOT copy plots from the reference scenes below
3. CREATE ENTIRELY NEW characters, plot, and dialogue
4. Reference scenes are BANNED from being copied - they show format only
5. The story MUST match: {query}

IGNORE THESE PLOTS (format reference only):
{parts_text[:500]}...
[Additional reference scenes omitted to prevent copying]

YOUR SCREENPLAY MUST INCLUDE:
- Title: Create an original title for the story: {query}
- Scene headings: INT./EXT. LOCATION - TIME
- Character names: Create NEW character names (do NOT use Alvy, Annie, Ray, Winston, etc.)
- Action lines: Describe what happens in: {query}
- Dialogue: Original dialogue for: {query}
- Length: 5-10 pages
- Plot: ONLY about {query}

START WRITING THE SCREENPLAY ABOUT "{query}" NOW:

FADE IN:
""".strip()

    print(f"{bcolors.OKBLUE}[5/5] {now()} | Prompt ready{bcolors.ENDC}")
    return {"prompt": prompt, "sources": unique_sources}

# ------------------------------------------------------------------- #
# MAIN
# ------------------------------------------------------------------- #
def main():
    print(f"{bcolors.HEADER}{'='*70}")
    print(" ULTRA-LIGHT RAG (LOCAL LLAMA3:8B via OLLAMA)")
    print(f" Time: {now()}")
    print(f"{'='*70}{bcolors.ENDC}\n")

    manifest = load_manifest()
    docs = build_documents(manifest)
    col = build_or_load_db(docs)

    try:
        query = input("Enter your story idea (query): ").strip()
        if not query:
            query = "A romance story set on an old steamship during a storm"
        genre = input("Enter genre (blank = Any): ").strip()
        if not genre:
            genre = "Any"
        actor_name = input("Enter actor name to base main character on (optional): ").strip()
    except (EOFError, KeyboardInterrupt):
        query = "A romance story set on an old steamship during a storm"
        genre = "Any"
        actor_name = ""
        print("\nUsing defaults.")

    actor_profile = None
    if actor_name:
        actor_file = Path(f"actors/{actor_name.lower().replace(' ','_')}.json")
        if actor_file.exists():
            print("Loading saved actor profile...")
            actor_profile = json.load(open(actor_file))
        else:
            print("Actor not found locally. Fetching from TMDB...")
            actor_profile = fetch_actor_tmdb(actor_name)

        print(f"Using actor profile for: {actor_profile['name']}")

    result = retrieve_and_build_prompt(col, query, genre, actor_profile)
    final_prompt = result["prompt"]
    sources = result["sources"]

    print(f"\n{bcolors.BOLD}Inspiration sources:{bcolors.ENDC} {', '.join(sources)}\n")

    print(f"{bcolors.OKBLUE}Calling local Llama 3 (Ollama)...{bcolors.ENDC}")
    screenplay = call_llama_local(final_prompt, model="llama3:8b")

    print(f"\n{bcolors.OKGREEN}{'='*70}")
    print(" GENERATED SCREENPLAY")
    print(f"{'='*70}{bcolors.ENDC}\n")
    print(screenplay)

if __name__ == "__main__":
    main()
