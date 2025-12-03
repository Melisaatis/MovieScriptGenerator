#!/usr/bin/env python3
"""
🎬 MOVIE SCRIPT GENERATOR & ACTOR BOOKING SYSTEM (LLAMA3 VERSION)
Uses +actors.py RAG system with Llama3 instead of Mistral
"""

import streamlit as st
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import sys
import re
import uuid
import requests
from PIL import Image
import io
import base64

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import components
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.utils import embedding_functions
    from openai import OpenAI
except ImportError as e:
    st.error(f"Missing required packages. Please install: {e}")
    st.stop()


# PAGE CONFIG

st.set_page_config(
    page_title="🎬 Movie Script Generator (Llama3)",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CONFIGURATION

SCRIPTS_ROOT = Path("../GenAIMovie/scripts")
MANIFEST_PATH = SCRIPTS_ROOT / "dataset_manifest.json"
DB_PATH = Path("../chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RETRIEVAL_K = 6
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Llama3 Ollama endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL = "llama3:8b"

# OpenAI Configuration
OPENAI_MODEL = "gpt-4o-mini"

# TMDB Configuration
TMDB_KEY = st.secrets.get("TMDB_KEY", "")

# Image Generation Configuration
IMAGE_API_OPTIONS = {
    "Stable Diffusion (Local)": "stable-diffusion",
    "DALL-E via OpenAI": "dalle",
    "Stable Diffusion XL via Replicate": "sdxl-replicate",
    "Hugging Face Inference": "hf-inference"
}


# SESSION STATE INITIALIZATION

if 'actors' not in st.session_state:
    st.session_state.actors = {}
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'generated_scripts' not in st.session_state:
    st.session_state.generated_scripts = []
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None
if 'actor_enrichment_enabled' not in st.session_state:
    st.session_state.actor_enrichment_enabled = False


# TMDB ACTOR FUNCTIONS

def fetch_actor_tmdb(actor_name: str) -> dict:
    """Fetch actor info from TMDB API"""
    try:
        search_url = f"https://api.themoviedb.org/3/search/person?api_key={TMDB_KEY}&query={actor_name}"
        result = requests.get(search_url, timeout=10).json()

        if not result.get("results"):
            return None

        actor = result["results"][0]
        actor_id = actor["id"]

        details_url = f"https://api.themoviedb.org/3/person/{actor_id}?api_key={TMDB_KEY}&append_to_response=movie_credits"
        details = requests.get(details_url, timeout=10).json()

        known_roles = [m["title"] for m in details.get("movie_credits", {}).get("cast", [])[:10] if "title" in m]

        return {
            "name": details.get("name", actor_name),
            "age": None,  # TMDB doesn't provide age directly
            "biography": details.get("biography", ""),
            "known_for": known_roles,
            "popularity": details.get("popularity", 0)
        }
    except Exception as e:
        st.warning(f"Could not fetch actor from TMDB: {e}")
        return None

def enrich_actor_with_openai(actor_name: str, openai_client) -> Optional[Dict]:
    """Use OpenAI to fetch and enrich actor information"""
    try:
        prompt = f"""Provide detailed information about the actor {actor_name} in JSON format.
        
Include the following fields:
        - full_name: Their complete name
        - age: Current age (approximate)
        - gender: "Male" or "Female"
        - nationality: Country of origin
        - known_for: List of 3-5 most famous movies or roles
        - personality: Brief description of their on-screen personality and acting style
        - typical_roles: Types of characters they typically play
        - career_highlights: 2-3 major awards or achievements
        - current_status: Whether they are actively working
        
If the actor is not well-known or you don't have information, return a JSON with "error" field.
Return ONLY valid JSON, no additional text."""
        
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a film industry expert with extensive knowledge of actors and cinema. Provide accurate, factual information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        actor_info = json.loads(response.choices[0].message.content)
        
        if "error" not in actor_info:
            return actor_info
    except Exception as e:
        st.warning(f"Could not enrich {actor_name}: {e}")
    
    return None


# RAG FUNCTIONS (from +actors.py)

scene_regex = re.compile(r"(?im)^(INT\.|EXT\.|INT/EXT\.)[^\n]*$")

def chunk_text(text: str) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

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

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with ChromaDB and embeddings"""
    try:
        # Load manifest
        if not MANIFEST_PATH.exists():
            st.error(f"Manifest not found at: {MANIFEST_PATH}")
            return None
        
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        valid = [e for e in manifest if e.get("script_path") and e.get("source") != "NotFound"]
        
        # Build documents (check if we need to)
        client = chromadb.PersistentClient(path=str(DB_PATH))
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        col = client.get_or_create_collection(name="movie_scripts", embedding_function=embedding_fn)
        
        if col.count() > 0:
            return col
        
        # Need to build DB
        docs = []
        for entry in valid:
            script_file = SCRIPTS_ROOT / entry["script_path"]
            if not script_file.exists():
                continue
            
            raw = script_file.read_text(encoding="utf-8", errors="ignore")
            if len(raw) < 500:
                continue
            
            chunks = chunk_script_with_scenes(raw, entry["title"], entry["year"], entry["genre"])
            for chunk in chunks:
                docs.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk["text"],
                    "metadata": chunk["metadata"]
                })
        
        if docs:
            col.add(
                documents=[d["text"] for d in docs],
                metadatas=[d["metadata"] for d in docs],
                ids=[d["id"] for d in docs]
            )
        
        return col
        
    except Exception as e:
        st.error(f"RAG initialization failed: {e}")
        return None

def call_llama_local(prompt: str, model: str = LLAMA_MODEL) -> str:
    """Call local Llama3 via Ollama API"""
    import random
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.9,  # High randomness for unique scripts
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "num_predict": 4000,  # Allow longer outputs
                    "seed": random.randint(1, 1000000)  # Prevent caching
                }
            },
            timeout=600
        )
        data = response.json()
        
        if "response" in data:
            return data["response"]
        if "output" in data:
            return data["output"]
        return str(data)
    except Exception as e:
        return f"[ERROR calling Llama3] {e}"

def retrieve_and_build_prompt(collection, query: str, genre: Optional[str], actors: List[Dict]) -> tuple:
    """Retrieve relevant script chunks and build prompt"""
    where = None
    if genre:   # mapping for local testing
        genre_map = {           
            "scifi": "scifi",
            "science fiction": "scifi",
            "sci-fi": "scifi",
            "romance": "romance",
            "romantic": "romance",
            "romcom": "comedy",
            "rom-com": "comedy",
            "action": "action",
            "comedy": "comedy",
            "horror": "horror",
            "drama": "drama",
            "thriller": "thriller",
            "adventure": "adventure",
            "animation": "animation"
        }
        
        genre_key = genre.lower().replace(" ", "").replace("-", "")
        genre_normalized = genre_map.get(genre_key, genre_key)
        where = {"genre": genre_normalized}
        
        st.info(f"Searching for genre: '{genre_normalized}' (from '{genre}')")
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=RETRIEVAL_K,
            where=where
        )
        
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        
        parts = []
        sources = []
        
        for d, m in zip(docs, metas):
            doc_genre = m.get('genre', '').lower()
            if genre and genre_normalized not in doc_genre:
                continue
                
            src = f"{m['title']} ({m['year']})"
            parts.append(f"--- {src} ---\n{d}")
            sources.append(src)
        
        if not parts and genre:
            st.warning(f"No {genre} references found. Doing broader search...")
            results = collection.query(
                query_texts=[query],
                n_results=RETRIEVAL_K
            )
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            
            for d, m in zip(docs, metas):
                src = f"{m['title']} ({m['year']})"
                parts.append(f"--- {src} ---\n{d}")
                sources.append(src)
        
        # Build actor information
        actor_block = ""
        if actors:
            actor_lines = []
            for a in actors:
                line = f"- {a['name']} (Age: {a.get('age', 'unknown')}, {a.get('gender', 'unknown')})"
                if a.get('enriched_data'):
                    ed = a['enriched_data']
                    if ed.get('known_for'):
                        line += f"\n  Known for: {', '.join(ed['known_for'][:3])}"
                    if ed.get('personality'):
                        line += f"\n  Style: {ed['personality']}"
                    if ed.get('typical_roles'):
                        line += f"\n  Typical roles: {ed['typical_roles']}"
                actor_lines.append(line)
            
            actor_block = "\n\nCAST:\n" + "\n".join(actor_lines)
        
        parts_text = "\n\n".join(parts)
        
        # Deduplicate sources
        seen = set()
        unique_sources = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                unique_sources.append(s)
        
        prompt = f"""
You are a Hollywood screenwriter.

==========================================
YOUR ASSIGNMENT - READ THIS CAREFULLY:
==========================================

STORY YOU MUST WRITE ABOUT: {query}

GENRE: {genre or 'Any'}
{actor_block}

==========================================
CRITICAL: Your screenplay MUST be about:
"{query}"

IF YOUR SCREENPLAY IS NOT ABOUT "{query}", YOU HAVE FAILED.
==========================================

Write a {genre or ''} screenplay that tells THIS story: {query}

The title, characters, plot, and dialogue must ALL relate to: {query}

DO NOT write about:
- ANY story other than: {query}

YOUR SCREENPLAY FORMAT:
- Title: (create a title for a story about: {query})
- Scene headings: INT./EXT. LOCATION - TIME
- Character names: {('Use CAST names: ' + ', '.join([a['name'] for a in actors])) if actors else 'Create original names'}
- Action: Describe events in the story about: {query}
- Dialogue: Characters discussing events from: {query}
- Length: 5-10 pages

REMEMBER: Every single line must relate to the story: {query}

START YOUR {genre or ''} SCREENPLAY ABOUT "{query}" NOW:

FADE IN:
""".strip()
        
        return prompt, unique_sources
    except Exception as e:
        st.error(f"Retrieval error: {e}")
        return "", []


# IMAGE GENERATION FUNCTIONS

def extract_scenes_for_images(screenplay: str, max_scenes: int = 3) -> List[str]:
    """Extract scene headings from screenplay"""
    lines = screenplay.split('\n')
    scenes = []
    for line in lines:
        if line.strip().startswith(('INT.', 'EXT.', 'INT/EXT.')):
            scenes.append(line.strip())
            if len(scenes) >= max_scenes:
                break
    return scenes

def generate_image_prompt_from_scene(scene: str, genre: str, title: str) -> str:
    """Generate image prompt from scene heading"""
    return f"""Cinematic still from {genre} film '{title}'. Scene: {scene}.
Professional film photography, dramatic lighting, {genre.lower()} atmosphere, 
high quality, theatrical, cinematic composition."""

def generate_image_stable_diffusion_local(prompt: str) -> Optional[Image.Image]:
    """Generate image using local Stable Diffusion (Automatic1111)"""
    try:
        response = requests.post(
            "http://127.0.0.1:7860/sdapi/v1/txt2img",
            json={
                "prompt": prompt,
                "steps": 30,
                "width": 768,
                "height": 512,
                "cfg_scale": 7.5
            },
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            if "images" in data and len(data["images"]) > 0:
                img_data = base64.b64decode(data["images"][0])
                return Image.open(io.BytesIO(img_data))
    except Exception as e:
        st.error(f"SD Local error: {e}")
    
    return None

# def generate_image_huggingface(prompt: str, hf_token: Optional[str] = None) -> Optional[Image.Image]:
#     """Generate image using Hugging Face Inference API"""
#     if not hf_token:
#         st.error("HuggingFace token required! Add it in sidebar under 'Hugging Face API'")
#         st.info("Get free token: https://huggingface.co/settings/tokens")
#         return None
    
#     try:
#         API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
#         headers = {"Authorization": f"Bearer {hf_token}"}
        
#         st.info("Calling HuggingFace API... (may take 20-30 seconds)")
        
#         response = requests.post(
#             API_URL,
#             headers=headers,
#             json={"inputs": prompt},
#             timeout=90
#         )
        
#         if response.status_code == 200:
#             # Check if response is actually an image
#             content_type = response.headers.get('content-type', '')
#             if 'image' in content_type:
#                 return Image.open(io.BytesIO(response.content))
#             else:
#                 st.error("HuggingFace returned non-image response")
#                 st.code(response.text[:500])
#                 return None
#         elif response.status_code == 503:
#             st.warning("Model is loading on HuggingFace (takes ~20s). Try again in a moment.")
#             try:
#                 error_data = response.json()
#                 st.json(error_data)
#             except:
#                 st.code(response.text[:500])
#             return None
#         elif response.status_code == 401:
#             st.error("Invalid HuggingFace token! Check your token in the sidebar.")
#             st.info("Get a new token: https://huggingface.co/settings/tokens")
#             return None
#         elif response.status_code == 403:
#             st.error("Access forbidden. Your HuggingFace token might not have permission.")
#             st.info("Make sure your token has 'read' permission")
#             return None
#         else:
#             st.error(f"HuggingFace API error: Status {response.status_code}")
#             try:
#                 error_data = response.json()
#                 st.json(error_data)
#             except:
#                 st.code(response.text[:500])
#             return None
#     except requests.Timeout:
#         st.error("HuggingFace API timeout. Model might be loading - try again in 30 seconds.")
#         return None
#     except Exception as e:
#         st.error(f"HF Inference error: {type(e).__name__}: {e}")
#         import traceback
#         st.code(traceback.format_exc())
#         return None

# def generate_image_dalle(prompt: str) -> Optional[Image.Image]:
#     """Generate image using DALL-E via OpenAI API"""
#     if not st.session_state.get('openai_client'):
#         st.error("OpenAI API key required! Add it in sidebar under 'OpenAI Integration'")
#         return None
    
#     try:
#         st.info("Calling DALL-E 3... (may take 10-20 seconds)")
#         response = st.session_state.openai_client.images.generate(
#             model="dall-e-3",
#             prompt=prompt,
#             size="1024x1024",
#             quality="standard",
#             n=1,
#         )
#         # Check for errors in response
#         if hasattr(response, 'error') and response.error:
#             st.error(f"DALL-E API error: {response.error}")
#             st.code(str(response.error))
#             return None
#         if not hasattr(response, 'data') or not response.data or not hasattr(response.data[0], 'url'):
#             st.error("DALL-E did not return an image URL.")
#             st.code(str(response))
#             return None
#         image_url = response.data[0].url
#         # Download the image
#         img_response = requests.get(image_url, timeout=30)
#         if img_response.status_code == 200:
#             return Image.open(io.BytesIO(img_response.content))
#         else:
#             st.error(f"Failed to download DALL-E image: {img_response.status_code}")
#             st.code(img_response.text[:500])
#             return None
#     except Exception as e:
#         st.error(f"DALL-E error: {type(e).__name__}: {e}")
#         import traceback
#         st.code(traceback.format_exc())
#         return None

def generate_movie_poster(title: str, genre: str, screenplay: str, method: str = "stable-diffusion") -> Optional[Image.Image]:
    """Generate a movie poster"""
    poster_prompt = f"""Professional movie poster for '{title}', a {genre} film.
Style: Theatrical release poster, dramatic composition, title treatment, cinematic lighting.
Genre: {genre}
Atmosphere: {genre.lower()} mood, professional Hollywood poster design.
Quality: High resolution, theatrical poster, award-winning composition."""
    
    if method == "stable-diffusion":
        return generate_image_stable_diffusion_local(poster_prompt)
    # elif method == "hf-inference":
    #     hf_token = st.session_state.get('hf_token')
    #     return generate_image_huggingface(poster_prompt, hf_token)
    # elif method == "dalle":
    #     return generate_image_dalle(poster_prompt)
    # elif method == "sdxl-replicate":
    #     st.warning("Replicate not implemented yet. Use Stable Diffusion or HuggingFace.")
    #     return None
    else:
        st.error(f"Unknown method: {method}")
        return None

def generate_scene_images(screenplay: str, genre: str, title: str, method: str = "stable-diffusion") -> List[Dict]:
    """Generate images for key scenes"""
    scenes = extract_scenes_for_images(screenplay, max_scenes=3)
    scene_images = []
    
    for i, scene in enumerate(scenes):
        prompt = generate_image_prompt_from_scene(scene, genre, title)
        
        if method == "stable-diffusion":
            img = generate_image_stable_diffusion_local(prompt)
        # elif method == "hf-inference":
        #     hf_token = st.session_state.get('hf_token')
        #     img = generate_image_huggingface(prompt, hf_token)
        # elif method == "dalle":
        #     img = generate_image_dalle(prompt)
        # elif method == "sdxl-replicate":
        #     img = None
        else:
            img = None
        
        if img:
            scene_images.append({
                "scene": scene,
                "image": img,
                "prompt": prompt
            })
    
    return scene_images

# MAIN APP

st.title("🎬 Movie Script Generator (Llama3)")
st.caption("Powered by Llama3 + RAG + OpenAI Actor Enrichment")

# SIDEBAR

with st.sidebar:
    st.header("⚙️ System Status")
    
    # Check Ollama/Llama3
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            st.success("✅ Llama3 (Ollama) Connected")
        else:
            st.error("❌ Llama3 Not Available")
    except:
        st.error("❌ Llama3 Not Running")
        st.info("Start Ollama with: `ollama serve`")
    
    # Initialize RAG
    if not st.session_state.rag_initialized:
        with st.spinner("Initializing RAG system..."):
            st.session_state.collection = initialize_rag_system()
            if st.session_state.collection:
                st.session_state.rag_initialized = True
                st.success(f"✅ RAG Initialized ({st.session_state.collection.count()} chunks)")
            else:
                st.error("❌ RAG Failed to Initialize")
    else:
        st.success(f"✅ RAG Ready ({st.session_state.collection.count()} chunks)")
    
    st.divider()
    
    st.header("📊 Statistics")
    st.metric("Actors Booked", len(st.session_state.actors))
    st.metric("Scripts Generated", len(st.session_state.generated_scripts))
    
    st.divider()
    
    st.header("🤖 OpenAI Integration")
    
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="For actor enrichment using web knowledge",
        key="openai_key_input"
    )
    
    if openai_key:
        try:
            st.session_state.openai_client = OpenAI(api_key=openai_key)
            st.session_state.actor_enrichment_enabled = True
            st.success("✅ OpenAI connected - Actor enrichment enabled")
            st.caption(f"Using {OPENAI_MODEL} for actor information")
        except Exception as e:
            st.error(f"OpenAI connection failed: {e}")
            st.session_state.actor_enrichment_enabled = False
    else:
        st.info("ℹ️ Add OpenAI key to enable automatic actor enrichment")
        st.caption("When enabled, actor info is auto-fetched from web knowledge")
    
    st.divider()
    
    st.header("🎨 Image Generation")
    
    # Check for Stable Diffusion
    try:
        sd_check = requests.get("http://127.0.0.1:7860/", timeout=1)
        if sd_check.status_code == 200:
            st.success("✅ Stable Diffusion (A1111) Running")
        else:
            st.info("ℹ️ SD Local: Not detected")
    except:
        st.info("ℹ️ SD Local: Not running")
        st.caption("Start A1111: `./webui.sh --api`")
    
    # # HuggingFace token input
    # with st.expander("🤗 Hugging Face API (For Images)", expanded=False):
    #     st.caption("Get free token: https://huggingface.co/settings/tokens")
    #     hf_token = st.text_input(
    #         "HF Token",
    #         type="password",
    #         help="Required for HuggingFace image generation",
    #         key="hf_token_input"
    #     )
    #     if hf_token:
    #         st.session_state.hf_token = hf_token
    #         st.success("✅ HF Token saved - Ready for image generation!")
    #     else:
    #         st.warning("⚠️ No token - HuggingFace images will fail")

# ACTOR BOOKING SECTION

st.header("🎭 Actor Booking System")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Add New Actor")
    with st.form("actor_form", clear_on_submit=True):
        actor_name = st.text_input("Actor Name", placeholder="e.g., Leonardo DiCaprio")
        
        auto_enrich = st.checkbox(
            "🤖 Auto-fetch info from web",
            value=st.session_state.actor_enrichment_enabled,
            disabled=not st.session_state.actor_enrichment_enabled,
            help="Uses OpenAI to fetch actor details automatically"
        )
        
        use_tmdb = st.checkbox(
            "🎬 Fetch from TMDB",
            value=False,
            help="Use TMDB API for filmography"
        )
        
        actor_age = st.number_input("Age", min_value=1, max_value=100, value=35, 
                                    help="Will be auto-filled if enrichment enabled")
        actor_gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Other"])
        actor_specialty = st.text_input("Specialty/Type", placeholder="e.g., Action Hero, Comedian",
                                       help="Will be auto-filled if enrichment enabled")
        
        submitted = st.form_submit_button("➕ Add Actor", use_container_width=True)
        
        if submitted:
            if actor_name:
                enriched_data = None
                tmdb_data = None
                
                # Try TMDB first if selected
                if use_tmdb:
                    with st.spinner(f"🎬 Fetching from TMDB: {actor_name}..."):
                        tmdb_data = fetch_actor_tmdb(actor_name)
                        if tmdb_data:
                            st.success(f"Found on TMDB: {tmdb_data['name']}")
                
                # Try OpenAI enrichment
                if auto_enrich and st.session_state.openai_client:
                    with st.spinner(f"🔍 Fetching info about {actor_name}..."):
                        enriched_data = enrich_actor_with_openai(actor_name, st.session_state.openai_client)
                        
                        if enriched_data:
                            st.success(f"Found detailed info for {actor_name}!")
                            if 'age' in enriched_data:
                                actor_age = enriched_data['age']
                            if 'typical_roles' in enriched_data:
                                actor_specialty = enriched_data['typical_roles']
                            # Override gender from enriched data if available
                            if 'gender' in enriched_data:
                                actor_gender = enriched_data['gender']
                
                # Merge TMDB and OpenAI data
                if enriched_data and tmdb_data:
                    enriched_data['tmdb_filmography'] = tmdb_data['known_for']
                elif tmdb_data and not enriched_data:
                    enriched_data = tmdb_data
                
                st.session_state.actors[actor_name] = {
                    "age": actor_age,
                    "gender": actor_gender,
                    "specialty": actor_specialty,
                    "enriched_data": enriched_data
                }
                st.success(f"Added {actor_name} to roster!")
                st.rerun()
            else:
                st.warning("Please enter an actor name")

with col2:
    st.subheader("Current Actor Roster")
    if st.session_state.actors:
        for name, data in st.session_state.actors.items():
            with st.container():
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.write(f"**{name}**")
                    st.caption(f"Age: {data['age']} | {data['gender']} | {data.get('specialty', 'N/A')}")
                    
                    # Show enriched data if available
                    if data.get('enriched_data'):
                        ed = data['enriched_data']
                        with st.expander("ℹ️ Detailed Info", expanded=False):
                            if ed.get('known_for'):
                                st.write("**Known for:**", ", ".join(ed['known_for'][:5]))
                            if ed.get('personality'):
                                st.write("**Style:**", ed['personality'])
                            if ed.get('nationality'):
                                st.write("**Nationality:**", ed['nationality'])
                            if ed.get('career_highlights'):
                                st.write("**Highlights:**", ", ".join(ed['career_highlights']))
                            if ed.get('tmdb_filmography'):
                                st.write("**TMDB Films:**", ", ".join(ed['tmdb_filmography'][:5]))
                
                with col_b:
                    if st.button("🗑️", key=f"del_{name}"):
                        del st.session_state.actors[name]
                        st.rerun()
                st.divider()
    else:
        st.info("No actors booked yet. Add some actors to get started!")

st.divider()


# SCRIPT GENERATION SECTION

st.header("🎬 Generate Movie Script")

with st.form("script_form"):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        movie_title = st.text_input("🎥 Movie Title", placeholder="e.g., Time Traveler's Paradox")
        query = st.text_area(
            "📝 Story Idea / Prompt",
            placeholder="e.g., Time traveller gets lost in the timeline",
            height=150
        )
    
    with col2:
        genre = st.selectbox(
            "🎭 Genre",
            ["Action", "SciFi", "Comedy", "Horror", "Drama", "Romance", "Thriller", "Adventure", "Animation"]
        )
        
        actor_selection = st.multiselect(
            "👥 Select Actors",
            options=list(st.session_state.actors.keys()),
            help="Choose actors from your roster"
        )
        
        use_rag = st.checkbox("Use RAG (Reference classic films)", value=True)
        
        st.divider()
        
        generate_images = st.checkbox("🎨 Generate Images", value=False, help="Generate poster and scene images")
        
        if generate_images:
            image_method = st.selectbox(
                "Image Generation Method",
                options=list(IMAGE_API_OPTIONS.keys()),
                help="Choose your image generation backend"
            )
    
    generate = st.form_submit_button("🎬 Generate Script", use_container_width=True)

if generate:
    if not movie_title or not query:
        st.warning("⚠️ Please enter both a movie title and story idea.")
    elif not st.session_state.rag_initialized:
        st.error("❌ RAG system not initialized. Please check the sidebar.")
    else:
        # Build actor list with enriched data
        actor_list = [
            {
                "name": a,
                "age": st.session_state.actors[a]["age"],
                "gender": st.session_state.actors[a]["gender"],
                "specialty": st.session_state.actors[a].get("specialty", ""),
                "enriched_data": st.session_state.actors[a].get("enriched_data")
            }
            for a in actor_selection
        ]

        with st.spinner("🎬 Generating screenplay with Llama3 + RAG..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Retrieve
            status_text.text(f"📚 Retrieving {genre} reference scenes...")
            progress_bar.progress(25)

            if use_rag:
                prompt, sources = retrieve_and_build_prompt(
                    st.session_state.collection,
                    query,
                    genre,
                    actor_list
                )

                # Show what was retrieved
                if sources:
                    with st.expander(f"📚 Reference Films Used ({len(sources)} sources)", expanded=False):
                        for source in sources:
                            st.write(f"• {source}")
                else:
                    st.warning(f"⚠️ No {genre} reference scenes found!")
            else:
                sources = []
                actor_block = ""
                if actor_list:
                    actor_lines = []
                    for a in actor_list:
                        actor_lines.append(f"- {a['name']} (Age: {a['age']}, {a['gender']})")
                    actor_block = "\n\nCAST:\n" + "\n".join(actor_lines)

                prompt = f"""Write a {genre} movie script titled '{movie_title}'.

Story: {query}{actor_block}

Write in proper Hollywood script format with scene headings, character names in CAPS, and action descriptions."""

            # Step 2: Generate Script
            status_text.text(f"Generating {genre} script with Llama3...")
            progress_bar.progress(50)

            # DEBUG: Show prompt details
            with st.expander("🔍 DEBUG: Prompt Preview", expanded=False):
                st.write(f"**Query:** {query}")
                st.write(f"**Genre:** {genre}")
                st.write(f"**Title:** {movie_title}")
                st.code(prompt[:1000] + "...\n[truncated]")

            screenplay = call_llama_local(prompt)

            progress_bar.progress(70)
            status_text.text("Script generated!")

            # Step 3: Generate Images (if requested)
            poster_image = None
            scene_images = []

            if generate_images:
                status_text.text("🎨 Generating movie poster...")
                progress_bar.progress(75)

                method = IMAGE_API_OPTIONS[image_method]
                st.write(f"🔍 DEBUG: Using method '{method}' for image generation")

                try:
                    poster_image = generate_movie_poster(movie_title, genre, screenplay, method)
                    st.write(f"🔍 DEBUG: Poster result type = {type(poster_image)}, is None? {poster_image is None}")

                    if poster_image:
                        st.success("✅ Poster generated!")
                        # Show immediately
                        st.image(poster_image, caption="Generated Poster Preview", width=300)
                        progress_bar.progress(85)
                    else:
                        st.warning("⚠️ Poster generation returned None")

                    status_text.text("🎨 Generating scene images...")
                    progress_bar.progress(90)
                    scene_images = generate_scene_images(screenplay, genre, movie_title, method)

                    st.write(f"🔍 DEBUG: Scene images count = {len(scene_images)}")

                    if scene_images:
                        st.success(f"✅ Generated {len(scene_images)} scene images!")
                        # Show immediately
                        for i, scene_data in enumerate(scene_images):
                            st.image(scene_data['image'], caption=f"Scene {i+1} Preview", width=300)
                    else:
                        st.warning("⚠️ No scene images generated")

                except Exception as e:
                    st.error(f"⚠️ Image generation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

            progress_bar.progress(100)
            status_text.text("✅ Complete!")

            # Store in session
            script_data = {
                "title": movie_title,
                "genre": genre,
                "query": query,
                "actors": actor_selection,
                "screenplay": screenplay,
                "sources": sources,
                "poster": poster_image,
                "scene_images": scene_images,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "images_generated": generate_images,
                "model": "Llama3"
            }

            # Only append if screenplay is different from last
            if not st.session_state.generated_scripts or st.session_state.generated_scripts[-1]["screenplay"] != screenplay:
                st.session_state.generated_scripts.append(script_data)
            else:
                st.session_state.generated_scripts[-1] = script_data

            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            st.rerun()


# DISPLAY GENERATED SCRIPTS

if st.session_state.generated_scripts:
    st.divider()
    st.header("📜 Generated Scripts")
    
    for idx, script in enumerate(reversed(st.session_state.generated_scripts)):
        with st.expander(f"🎬 {script['title']} ({script['genre']}) - {script['timestamp']}", expanded=(idx==0)):
            
            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Genre:** {script['genre']}")
                st.write(f"**Model:** {script.get('model', 'Unknown')}")
            with col2:
                st.write(f"**Actors:** {', '.join(script['actors']) if script['actors'] else 'None'}")
            with col3:
                st.write(f"**Generated:** {script['timestamp']}")
            
            if script.get('sources'):
                st.caption(f"**References:** {', '.join(script['sources'])}")
            
            st.divider()
            
            # Display Images if available
            poster = script.get('poster')
            scene_images = script.get('scene_images', [])
            
            if poster or scene_images:
                st.subheader("🎨 Generated Images")
                
                if poster and scene_images:
                    img_cols = st.columns([1, 2])
                elif poster:
                    img_cols = [st.container()]
                else:
                    img_cols = [None, st.container()]
                
                # Movie Poster
                if poster:
                    with img_cols[0]:
                        st.image(poster, caption=f"🎬 {script['title']} - Poster", use_container_width=True)
                        
                        buf = io.BytesIO()
                        poster.save(buf, format='PNG')
                        buf.seek(0)
                        st.download_button(
                            label="💾 Download Poster",
                            data=buf.getvalue(),
                            file_name=f"{script['title'].replace(' ', '_')}_poster.png",
                            mime="image/png",
                            key=f"dl_poster_{idx}",
                            use_container_width=True
                        )
                
                # Scene Images
                if scene_images:
                    scene_col_idx = 1 if poster else 0
                    if img_cols[scene_col_idx]:
                        with img_cols[scene_col_idx]:
                            st.write("**🎞️ Key Scenes:**")
                            for i, scene_data in enumerate(scene_images):
                                st.image(
                                    scene_data['image'],
                                    caption=f"Scene {i+1}: {scene_data['scene']}",
                                    use_container_width=True
                                )
                                
                                scene_buf = io.BytesIO()
                                scene_data['image'].save(scene_buf, format='PNG')
                                scene_buf.seek(0)
                                st.download_button(
                                    label=f"💾 Download Scene {i+1}",
                                    data=scene_buf.getvalue(),
                                    file_name=f"{script['title'].replace(' ', '_')}_scene_{i+1}.png",
                                    mime="image/png",
                                    key=f"dl_scene_{idx}_{i}",
                                    use_container_width=True
                                )
                
                st.divider()
            
            # Screenplay
            st.subheader("📄 Screenplay")
            
            col_script, col_actions = st.columns([3, 1])
            
            with col_script:
                st.text_area(
                    "Script Content",
                    value=script['screenplay'],
                    height=400,
                    key=f"script_{idx}",
                    label_visibility="collapsed"
                )
            
            with col_actions:
                st.download_button(
                    label="💾 Download Script",
                    data=script['screenplay'],
                    file_name=f"{script['title'].replace(' ', '_')}.txt",
                    mime="text/plain",
                    key=f"download_{idx}",
                    use_container_width=True
                )
                
                if st.button("🗑️ Delete Script", key=f"delete_{idx}", use_container_width=True):
                    actual_idx = len(st.session_state.generated_scripts) - 1 - idx
                    del st.session_state.generated_scripts[actual_idx]
                    st.success("✅ Script deleted!")
                    st.rerun()

# Footer
st.divider()
st.caption("🎬 Movie Script Generator (Llama3) | Powered by Llama3 + RAG + OpenAI")
