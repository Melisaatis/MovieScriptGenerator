#!/usr/bin/env python3
"""
RAG Movie Script Generator – ULTRA-LIGHT CPU-ONLY + LOCAL MISTRAL (OLLAMA)
Chunk → Embed → Retrieve → Build prompt → Generate screenplay with Mistral
"""

import json
import re
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import requests
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

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
# LOCAL MISTRAL (OLLAMA) CALL
# ------------------------------------------------------------------- #
def call_mistral_local(prompt: str, model: str = "mistral") -> str:
    """
    Works with ALL Ollama output formats:
    - response
    - output
    - streaming chunks
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=600
        )

        # Try JSON parse — if fail, return raw text
        try:
            data = response.json()
        except Exception:
            return response.text

        # Format A
        if "response" in data:
            return data["response"]

        # Format B
        if "output" in data:
            return data["output"]

        # Format C (fallback for unexpected responses)
        return str(data)

    except Exception as e:
        return f"[ERROR calling local Mistral] {e}"


# ------------------------------------------------------------------- #
# CONFIG - EDIT THESE PATHS
# ------------------------------------------------------------------- #

SCRIPTS_ROOT = Path("/Users/alessiacolumban/MovieScriptGenerator/GenAIMovie/scripts")
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
        raise FileNotFoundError(
            f"{bcolors.WARNING}Manifest not found at:\n{MANIFEST_PATH}{bcolors.ENDC}"
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    valid = []
    for e in manifest:
        if e.get("script_path") and e.get("source") != "NotFound":
            valid.append(e)

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

scene_regex = re.compile(
    r"(?im)^(INT\.|EXT\.|INT/EXT\.)[^\n]*$"
)

def chunk_script_with_scenes(text: str, title: str, year: str, genre: str) -> List[Dict]:
    scenes = scene_regex.split(text)
    chunks = []
    current_block = ""

    for i in range(1, len(scenes), 2):
        header = scenes[i].strip()
        content = scenes[i + 1].strip() if i + 1 < len(scenes) else ""
        scene_block = f"SCENE: {header}\n{content}"

        if not current_block:
            current_block = scene_block
        else:
            current_block += "\n\n" + scene_block

        if len(current_block.split()) >= CHUNK_SIZE:
            for sub in chunk_text(current_block):
                chunks.append({
                    "text": sub,
                    "metadata": {
                        "title": title,
                        "year": year,
                        "genre": genre.lower().replace(" ", "")
                    }
                })
            current_block = ""

    if current_block.strip():
        for sub in chunk_text(current_block):
            chunks.append({
                "text": sub,
                "metadata": {
                    "title": title,
                    "year": year,
                    "genre": genre.lower().replace(" ", "")
                }
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

        chunks = chunk_script_with_scenes(
            raw,
            entry["title"],
            entry["year"],
            entry["genre"]
        )

        for chunk in chunks:
            docs.append({
                "id": str(uuid.uuid4()),
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })

        print(f"{bcolors.OKCYAN}       → [{i:3}/{len(manifest)}] {entry['title']} → {len(chunks)} chunks{bcolors.ENDC}")

    print(f"{bcolors.OKGREEN}       → TOTAL: {len(docs):,} chunks{bcolors.ENDC}")
    return docs

# ------------------------------------------------------------------- #
# 4. VECTOR DB
# ------------------------------------------------------------------- #
def build_or_load_db(docs: List[Dict]):
    print(f"{bcolors.OKBLUE}[3/5] {now()} | Initializing vector DB...{bcolors.ENDC}")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    col = client.get_or_create_collection(
        name="movie_scripts",
        embedding_function=embedding_fn
    )

    if col.count() > 0:
        print(f"{bcolors.OKGREEN}       → Loaded existing DB with {col.count()} items{bcolors.ENDC}")
        return col

    print(f"{bcolors.WARNING}       → Building DB... adding {len(docs)} chunks{bcolors.ENDC}")

    col.add(
        documents=[d["text"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
        ids=[d["id"] for d in docs]
    )

    print(f"{bcolors.OKGREEN}       → DB build complete{bcolors.ENDC}")
    return col

# ------------------------------------------------------------------- #
# 5. RETRIEVAL + PROMPT
# ------------------------------------------------------------------- #
def retrieve_and_build_prompt(col, query: str, genre: Optional[str] = None) -> str:
    print(f"{bcolors.OKBLUE}[4/5] {now()} | Retrieving...{bcolors.ENDC}")

    where = None
    if genre:
        where = {"genre": genre.lower().replace(" ", "")}

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

    print(f"{bcolors.OKCYAN}       → Retrieved from: {', '.join(sources)}{bcolors.ENDC}")

    prompt = """
You are a master screenwriter. Write an original movie script inspired by these scenes.

USER REQUEST: {query}
GENRE: {genre}

--- REFERENCE SCENES ---
{parts}
--- END ---

Write a 3–5 page original screenplay in proper script format.
""".format(
    query=query,
    genre=genre or "Any",
    parts="\n\n".join(parts)
).strip()


    print(f"{bcolors.OKBLUE}[5/5] {now()} | Prompt ready{bcolors.ENDC}")
    return prompt

# ------------------------------------------------------------------- #
# MAIN
# ------------------------------------------------------------------- #
def main():
    print(f"{bcolors.HEADER}{'='*70}")
    print(" ULTRA-LIGHT RAG (NO CLOUD LLM — LOCAL MISTRAL ENABLED)")
    print(f" Time: {now()}")
    print(f"{'='*70}{bcolors.ENDC}\n")

    manifest = load_manifest()
    docs = build_documents(manifest)
    col = build_or_load_db(docs)

    query = "A rogue AI in a spaceship begins eliminating crew members, in the style of Alien and 2001."
    genre = "SciFi"

    final_prompt = retrieve_and_build_prompt(col, query, genre)

    # 🌟 NEW: CALL LOCAL MISTRAL HERE 🌟
    print(f"{bcolors.OKBLUE}Calling local Mistral (Ollama)...{bcolors.ENDC}")
    screenplay = call_mistral_local(final_prompt)

    print(f"\n{bcolors.OKGREEN}{'='*70}")
    print(" GENERATED SCREENPLAY")
    print(f"{'='*70}{bcolors.ENDC}\n")
    print(screenplay)

if __name__ == "__main__":
    main()
