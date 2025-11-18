#!/usr/bin/env python3
"""
RAG Movie Script Generator → Local Mistral (Ollama)
Fully offline, zero cost, perfect for Belgium 2025
"""

import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

# Retrieval part (CPU-only)
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# Local Mistral via Ollama
import ollama   # pip install ollama

# ------------------------------------------------------------------- #
# CONFIG — ONLY THING YOU MIGHT WANT TO CHANGE
# ------------------------------------------------------------------- #
SCRIPTS_ROOT = Path("scripts")
MANIFEST_PATH = SCRIPTS_ROOT / "dataset_manifest.json"
DB_PATH = Path("chroma_db")

# Pick your favorite Mistral variant (must be already pulled with `ollama pull`)
MISTRAL_MODEL = "mistral-small"          # ← best balance speed/quality for screenplays
# MISTRAL_MODEL = "openhermestral"       # very fast 12B
# MISTRAL_MODEL = "mistral-large"        # if you have 32GB+ RAM

RETRIEVAL_K = 6

# ------------------------------------------------------------------- #
# PRETTY COLORS
# ------------------------------------------------------------------- #
class bcolors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'; WARNING = '\033[93m'; FAIL = '\033[91m'
    ENDC = '\033[0m'; BOLD = '\033[1m'
def now(): return time.strftime("%H:%M:%S")

# ------------------------------------------------------------------- #
# 1–4: Load → Chunk → Embed → Retrieve (unchanged, just cleaned)
# ------------------------------------------------------------------- #
def load_manifest() -> List[Dict]:
    print(f"{bcolors.OKBLUE}[1/5] {now()} | Loading manifest...{bcolors.ENDC}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    print(f"{bcolors.OKGREEN}       → {len(manifest)} scripts loaded{bcolors.ENDC}")
    return manifest

def chunk_script(text: str, title: str, year: str, genre: str) -> List[Dict]:
    scene_pattern = r"(?i)\b(int\.|ext\.)\s*[^\n]*"
    parts = re.split(scene_pattern, text)
    chunks = []
    block = ""
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        block += f"\n\nSCENE: {header}\n{content}" if block else f"SCENE: {header}\n{content}"
        if len(block) > 1000:
            chunks.append({"text": block, "metadata": {"title": title, "year": year, "genre": genre}})
            block = ""
    if block.strip():
        chunks.append({"text": block, "metadata": {"title": title, "year": year, "genre": genre}})
    return chunks

def build_documents(manifest: List[Dict]) -> List[Dict]:
    print(f"{bcolors.OKBLUE}[2/5] {now()} | Chunking scripts...{bcolors.ENDC}")
    docs = []
    for i, entry in enumerate(manifest, 1):
        path = SCRIPTS_ROOT / entry["script_path"]
        if not path.exists(): continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_script(text, entry["title"], entry["year"], entry["genre"])
        for j, c in enumerate(chunks):
            docs.append({
                "id": f"id_{len(docs)}",
                "text": c["text"],
                "metadata": c["metadata"]
            })
        print(f"{bcolors.OKCYAN}       → [{i:3}] {entry['title']} ({entry['year']}) → {len(chunks)} chunks{bcolors.ENDC}")
    return docs

def get_collection(docs: List[Dict]):
    print(f"{bcolors.OKBLUE}[3/5] {now()} | Vector DB...{bcolors.ENDC}")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    coll = client.get_or_create_collection("movie_scripts", embedding_function=ef)
    if coll.count() == 0:
        print(f"{bcolors.WARNING}       → Building DB (2–3 min first time)...{bcolors.ENDC}")
        coll.add(
            documents=[d["text"] for d in docs],
            metadatas=[d["metadata"] for d in docs],
            ids=[d["id"] for d in docs]
        )
    else:
        print(f"{bcolors.OKGREEN}       → Loaded {coll.count()} chunks from DB{bcolors.ENDC}")
    return coll

def retrieve_and_prompt(collection, query: str, genre: str):
    print(f"{bcolors.OKBLUE}[4/5] {now()} | Retrieving scenes...{bcolors.ENDC}")
    results = collection.query(query_texts=[query], n_results=RETRIEVAL_K, where={"genre": genre} if genre else None)
    examples = []
    sources = set()
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        src = f"{meta['title']} ({meta['year']})"
        sources.add(src)
        examples.append(f"--- {src} ---\n{text}")

    print(f"{bcolors.OKCYAN}       → Sources: {', '.join(sorted(sources))}{bcolors.ENDC}")

    prompt = f"""
You are a master screenwriter. Write an original 3–5 page movie script.

USER REQUEST: {query}
GENRE: {genre}

--- RELEVANT SCENES FROM CLASSIC FILMS ---
{"\n\n".join(examples)}
--- END OF EXAMPLES ---

Write in proper screenplay format. Be original, cinematic, and terrifying.
""".strip()
    return prompt

# ------------------------------------------------------------------- #
# 5. Local Mistral generation
# ------------------------------------------------------------------- #
def generate_with_mistral(prompt: str) -> str:
    print(f"{bcolors.OKBLUE}[5/5] {now()} | Generating with {MISTRAL_MODEL} (local)...{bcolors.ENDC}")
    resp = ollama.generate(
        model=MISTRAL_MODEL,
        prompt=prompt,
        options={"temperature": 0.8, "num_predict": 6000}
    )
    return resp["response"]

# ------------------------------------------------------------------- #
# MAIN
# ------------------------------------------------------------------- #
def main():
    print(f"{bcolors.HEADER}{'='*80}")
    print(" RAG → LOCAL MISTRAL MOVIE SCRIPT GENERATOR (100% offline)")
    print(f" Time: {now()} | Belgium")
    print(f"{'='*80}{bcolors.ENDC}\n")

    manifest = load_manifest()
    docs = build_documents(manifest)
    collection = get_collection(docs)

    query = "A rogue AI in a spaceship starts locking doors and hunting the crew, in the style of Alien and 2001"
    genre = "SciFi"

    print(f"\n{bcolors.BOLD}QUERY: {query}{bcolors.ENDC}\n")

    prompt = retrieve_and_prompt(collection, query, genre)
    script = generate_with_mistral(prompt)

    print(f"\n{bcolors.OKGREEN}{'='*80}")
    print(" YOUR ORIGINAL SCRIPT (generated with local Mistral)")
    print(f"{'='*80}{bcolors.ENDC}\n")
    print(script)

    Path("mistral_generated_script.txt").write_text(script, encoding="utf-8")
    print(f"\n{bcolors.OKGREEN}→ Saved to mistral_generated_script.txt{bcolors.ENDC}")

if __name__ == "__main__":
    main()