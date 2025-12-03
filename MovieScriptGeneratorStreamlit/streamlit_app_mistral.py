#!/usr/bin/env python3
"""
🎬 MOVIE SCRIPT GENERATOR & ACTOR BOOKING SYSTEM
Integrates MCP + Langchain + Mistral RAG for intelligent script generation
"""

import streamlit as st
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add parent directory to path to import langchain+mistral module
sys.path.append(str(Path(__file__).parent.parent))

# Import RAG components
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.utils import embedding_functions
    import requests
    from PIL import Image
    import io
    import base64
    from openai import OpenAI
except ImportError as e:
    st.error(f"Missing required packages. Please install: {e}")
    st.stop()

# ===================================================================
# PAGE CONFIG
# ===================================================================
st.set_page_config(
    page_title="🎬 Movie Script Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================================
# CONFIGURATION
# ===================================================================
SCRIPTS_ROOT = Path("../GenAIMovie/scripts")
MANIFEST_PATH = SCRIPTS_ROOT / "dataset_manifest.json"
DB_PATH = Path("../chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RETRIEVAL_K = 6
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Mistral Ollama endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"
MISTRAL_MODEL = "mistral"

# OpenAI Configuration
OPENAI_MODEL = "gpt-4o-mini"  

# Image Generation Configuration
IMAGE_API_OPTIONS = {
    "Stable Diffusion (Local)": "stable-diffusion",
    "DALL-E via OpenAI": "dalle",
    "Stable Diffusion XL via Replicate": "sdxl-replicate",
    "Hugging Face Inference": "hf-inference"
}

# ===================================================================
# SESSION STATE INITIALIZATION
# ===================================================================
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

# ===================================================================
# RAG FUNCTIONS
# ===================================================================

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with ChromaDB and embeddings"""
    try:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        col = client.get_or_create_collection(
            name="movie_scripts",
            embedding_function=embedding_fn
        )
        return col
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

def call_mistral_local(prompt: str, model: str = MISTRAL_MODEL) -> str:
    """Call local Mistral via Ollama"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=600
        )
        data = response.json()
        if "response" in data:
            return data["response"]
        elif "output" in data:
            return data["output"]
        return str(data)
    except Exception as e:
        return f"[ERROR calling Mistral] {e}"

def extract_scenes_for_images(screenplay: str, max_scenes: int = 3) -> List[str]:
    """Extract key scenes from screenplay for image generation"""
    import re
    
    # Find scene headers
    scene_pattern = re.compile(r'(INT\.|EXT\.)[^\n]+', re.IGNORECASE)
    scenes = scene_pattern.findall(screenplay)
    
    # Get unique scenes, limit to max_scenes
    unique_scenes = list(dict.fromkeys(scenes))[:max_scenes]
    
    return unique_scenes

def generate_image_prompt_from_scene(scene: str, genre: str, title: str) -> str:
    """Create a detailed image prompt from a scene description"""
    prompt = f"""Create a cinematic movie poster style image for a {genre} film titled '{title}'.
Scene: {scene}
Style: Professional movie poster, dramatic lighting, high quality, cinematic composition, {genre.lower()} atmosphere.
Details: Photorealistic, film grain, movie poster aesthetic, theatrical release quality."""
    return prompt

def generate_image_stable_diffusion_local(prompt: str) -> Optional[Image.Image]:
    """Generate image using local Stable Diffusion via Ollama or Automatic1111"""
    try:
        # Try Automatic1111 API first
        response = requests.post(
            "http://127.0.0.1:7860/sdapi/v1/txt2img",
            json={
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, distorted, ugly, bad anatomy",
                "steps": 30,
                "width": 512,
                "height": 768,
                "cfg_scale": 7.5,
                "sampler_name": "DPM++ 2M Karras"
            },
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            if "images" in data and len(data["images"]) > 0:
                img_data = base64.b64decode(data["images"][0])
                return Image.open(io.BytesIO(img_data))
    except Exception as e:
        st.warning(f"Local SD not available: {e}")
    
    return None

def generate_image_huggingface(prompt: str, hf_token: Optional[str] = None) -> Optional[Image.Image]:
    """Generate image using Hugging Face Inference API"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt},
            timeout=60
        )
        
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"HF Inference error: {e}")
    
    return None

def generate_movie_poster(title: str, genre: str, screenplay: str, method: str = "stable-diffusion") -> Optional[Image.Image]:
    """Generate a movie poster based on the script"""
    
    # Create poster prompt
    poster_prompt = f"""Professional movie poster for '{title}', a {genre} film.
Style: Theatrical release poster, dramatic composition, title treatment, cinematic lighting.
Genre: {genre}
Atmosphere: {genre.lower()} mood, professional Hollywood poster design.
Quality: High resolution, theatrical poster, award-winning composition."""
    
    if method == "stable-diffusion":
        return generate_image_stable_diffusion_local(poster_prompt)
    elif method == "hf-inference":
        hf_token = st.session_state.get('hf_token')
        return generate_image_huggingface(poster_prompt, hf_token)
    
    return None

def generate_scene_images(screenplay: str, genre: str, title: str, method: str = "stable-diffusion") -> List[Dict]:
    """Generate images for key scenes in the screenplay"""
    scenes = extract_scenes_for_images(screenplay, max_scenes=3)
    scene_images = []
    
    for i, scene in enumerate(scenes):
        prompt = generate_image_prompt_from_scene(scene, genre, title)
        
        if method == "stable-diffusion":
            img = generate_image_stable_diffusion_local(prompt)
        elif method == "hf-inference":
            hf_token = st.session_state.get('hf_token')
            img = generate_image_huggingface(prompt, hf_token)
        else:
            img = None
        
        if img:
            scene_images.append({
                "scene": scene,
                "image": img,
                "prompt": prompt
            })
    
    return scene_images

def enrich_actor_with_openai(actor_name: str, openai_client) -> Optional[Dict]:
    """Use OpenAI to fetch and enrich actor information from web knowledge"""
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
Return ONLY valid JSON, no additional text."""        response = openai_client.chat.completions.create(
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

def retrieve_and_build_prompt(collection, query: str, genre: Optional[str], actors: List[Dict]) -> tuple:
    """Retrieve relevant script chunks and build prompt"""
    where = None
    if genre:
        # Normalize genre name to match database format (lowercase, no spaces)
        # Map common variations to database format
        genre_map = {
            "scifi": "scifi",
            "science fiction": "scifi",
            "sci-fi": "scifi",
            "romance": "romance",
            "romantic": "romance",
            "romcom": "comedy",  # romantic comedy uses comedy genre
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
        
        # Debug: Show what we're searching for
        st.info(f"🔍 Searching for genre: '{genre_normalized}' (from '{genre}')")
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=RETRIEVAL_K,
            where=where
        )
        
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        
        # Verify we got the right genre and filter out mismatches
        parts = []
        sources = []
        
        for d, m in zip(docs, metas):
            # Double-check genre matches
            doc_genre = m.get('genre', '').lower()
            if genre and genre_normalized not in doc_genre:
                continue  # Skip documents that don't match genre
                
            src = f"{m['title']} ({m['year']}) - {m['genre']}"
            parts.append(f"--- {src} ---\n{d}")
            sources.append(src)
        
        # If no genre matches found, do a broader search without genre filter
        if not parts and genre:
            st.warning(f"⚠️ No {genre} references found. Doing broader search...")
            results = collection.query(
                query_texts=[query],
                n_results=RETRIEVAL_K
            )
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            
            for d, m in zip(docs, metas):
                src = f"{m['title']} ({m['year']}) - {m['genre']}"
                parts.append(f"--- {src} ---\n{d}")
                sources.append(src)
        
        # Build actor cast list with enriched data
        cast_list = ""
        if actors:
            cast_parts = []
            for a in actors:
                actor_line = f"- {a['name']} (Age: {a['age']}, {a['gender']})"
                
                # Add specialty if present
                if a.get('specialty'):
                    actor_line += f", Specialty: {a['specialty']}"
                
                # Add enriched data if available
                if a.get('enriched_data'):
                    ed = a['enriched_data']
                    if ed.get('known_for'):
                        actor_line += f"\n  Known for: {', '.join(ed['known_for'][:3])}"
                    if ed.get('personality'):
                        actor_line += f"\n  Acting style: {ed['personality']}"
                    if ed.get('typical_roles'):
                        actor_line += f"\n  Typical roles: {ed['typical_roles']}"
                
                cast_parts.append(actor_line)
            
            cast_list = "\n\nCAST:\n" + "\n".join(cast_parts)
        
        prompt = f"""
You are a professional Hollywood screenwriter. Your task is to write an original {genre or ''} screenplay based EXACTLY on the user's story request below.

=== USER'S STORY REQUEST (FOLLOW THIS EXACTLY) ===
{query}
===

GENRE: {genre or 'Any'}{cast_list}

CRITICAL INSTRUCTIONS:
1. Write EXACTLY the story the user requested above - do NOT change the plot or characters
2. This MUST be a {genre.upper() if genre else ''} film - stay true to this genre
3. Use the reference scenes below ONLY for formatting style and dialogue techniques, NOT for story ideas
4. The story MUST match what the user asked for: {query}

--- REFERENCE SCENES (FOR STYLE ONLY - DO NOT COPY PLOTS) ---
{chr(10).join(parts)}
--- END REFERENCES ---

SCREENPLAY REQUIREMENTS:
- Follow proper Hollywood script format (INT./EXT., character names in CAPS, action lines)
- Write 3-5 pages
- Stay 100% faithful to the user's requested story: {query}
- Match the {genre} genre conventions
- If actors are specified, write roles that fit their age and persona
- DO NOT invent a completely different story - write what the user asked for!

NOW WRITE THE {genre.upper() if genre else ''} SCREENPLAY FOR: {query}
""".strip()
        
        return prompt, sources
    except Exception as e:
        st.error(f"Retrieval error: {e}")
        return "", []

# ===================================================================
# MAIN APP
# ===================================================================

# Title and Header
st.title("🎬 Movie Script Generator & Actor Booking System")
st.markdown("""
### Powered by RAG + Langchain + Mistral (MCP Integration)
Generate professional movie scripts using AI with context from classic films!
""")

# Sidebar for system status
with st.sidebar:
    st.header("⚙️ System Status")
    
    # Check Ollama connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            st.success("✅ Mistral (Ollama) Connected")
        else:
            st.error("❌ Mistral Not Available")
    except:
        st.error("❌ Mistral Not Running")
        st.info("Start Ollama with: `ollama serve`")
    
    # Initialize RAG
    if not st.session_state.rag_initialized:
        with st.spinner("Initializing RAG system..."):
            st.session_state.collection = initialize_rag_system()
            if st.session_state.collection:
                st.session_state.rag_initialized = True
                st.success(f"✅ RAG Initialized ({st.session_state.collection.count()} chunks)")
                
                # Show available genres in database
                try:
                    # Get a sample to see what genres exist
                    sample = st.session_state.collection.get(limit=100)
                    genres = set()
                    if sample and 'metadatas' in sample:
                        for meta in sample['metadatas']:
                            if 'genre' in meta:
                                genres.add(meta['genre'])
                    if genres:
                        st.caption(f"📂 Genres in DB: {', '.join(sorted(genres))}")
                except:
                    pass
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
    
    # HuggingFace token input
    with st.expander("🤗 Hugging Face API (Optional)"):
        hf_token = st.text_input(
            "HF Token",
            type="password",
            help="For HF Inference API",
            key="hf_token_input"
        )
        if hf_token:
            st.session_state.hf_token = hf_token
            st.success("✅ HF Token saved")

# ===================================================================
# ACTOR BOOKING SECTION
# ===================================================================

st.header("🎭 Actor Booking System")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Add New Actor")
    with st.form("actor_form", clear_on_submit=True):
        actor_name = st.text_input("Actor Name", placeholder="e.g., Scarlett Johansson")
        
        auto_enrich = st.checkbox(
            "🤖 Auto-fetch info from web",
            value=st.session_state.actor_enrichment_enabled,
            disabled=not st.session_state.actor_enrichment_enabled,
            help="Uses OpenAI to fetch actor details automatically"
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
                
                # Try to enrich if enabled
                if auto_enrich and st.session_state.openai_client:
                    with st.spinner(f"🔍 Fetching info about {actor_name}..."):
                        enriched_data = enrich_actor_with_openai(actor_name, st.session_state.openai_client)
                        
                        if enriched_data:
                            st.success(f"✅ Found detailed info for {actor_name}!")
                            # Override with enriched data if available
                            if 'age' in enriched_data:
                                actor_age = enriched_data['age']
                            if 'typical_roles' in enriched_data:
                                actor_specialty = enriched_data['typical_roles']
                
                st.session_state.actors[actor_name] = {
                    "age": actor_age,
                    "gender": actor_gender,
                    "specialty": actor_specialty,
                    "enriched_data": enriched_data
                }
                st.success(f"✅ Added {actor_name} to roster!")
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
                
                with col_b:
                    if st.button("🗑️", key=f"del_{name}"):
                        del st.session_state.actors[name]
                        st.rerun()
                st.divider()
    else:
        st.info("No actors booked yet. Add some actors to get started!")

st.divider()

# ===================================================================
# SCRIPT GENERATION SECTION
# ===================================================================

st.header("🎬 Generate Movie Script")

with st.form("script_form"):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        movie_title = st.text_input("🎥 Movie Title", placeholder="e.g., The Last Mission")
        query = st.text_area(
            "📝 Story Idea / Prompt",
            placeholder="Describe your movie concept...",
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
        
        with st.spinner("🎬 Generating screenplay with Mistral + RAG..."):
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
                cast_list = ""
                if actor_list:
                    cast_list = "\n\nCAST:\n" + "\n".join(
                        [f"- {a['name']} (Age: {a['age']}, {a['gender']})" for a in actor_list]
                    )
                prompt = f"""Write a {genre} movie script titled '{movie_title}'.

This MUST be a {genre.upper()} screenplay. Stay true to the {genre} genre.

Story: {query}{cast_list}

Write in proper Hollywood script format with scene headings, character names in CAPS, and action descriptions appropriate for the {genre} genre."""
            
            # Step 2: Generate Script
            status_text.text(f"✍️ Generating {genre} script with Mistral...")
            progress_bar.progress(50)
            
            screenplay = call_mistral_local(prompt)
            
            progress_bar.progress(70)
            status_text.text("✅ Script generated!")
            
            # Step 3: Generate Images (if requested)
            poster_image = None
            scene_images = []
            
            if generate_images:
                status_text.text("🎨 Generating movie poster...")
                progress_bar.progress(80)
                
                method = IMAGE_API_OPTIONS[image_method]
                
                try:
                    poster_image = generate_movie_poster(movie_title, genre, screenplay, method)
                    if poster_image:
                        st.success("✅ Poster generated!")
                    
                    # Generate scene images
                    status_text.text("🎨 Generating scene images...")
                    progress_bar.progress(90)
                    scene_images = generate_scene_images(screenplay, genre, movie_title, method)
                    
                except Exception as e:
                    st.warning(f"⚠️ Image generation failed: {e}")
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            time.sleep(0.5)
            
            # Store in session
            st.session_state.generated_scripts.append({
                "title": movie_title,
                "genre": genre,
                "query": query,
                "actors": actor_selection,
                "screenplay": screenplay,
                "sources": sources,
                "poster": poster_image,
                "scene_images": scene_images,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            progress_bar.empty()
            status_text.empty()

# ===================================================================
# DISPLAY GENERATED SCRIPTS
# ===================================================================

if st.session_state.generated_scripts:
    st.divider()
    st.header("📜 Generated Scripts")
    
    for idx, script in enumerate(reversed(st.session_state.generated_scripts)):
        with st.expander(f"🎬 {script['title']} ({script['genre']}) - {script['timestamp']}", expanded=(idx==0)):
            
            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Genre:** {script['genre']}")
            with col2:
                st.write(f"**Actors:** {', '.join(script['actors']) if script['actors'] else 'None'}")
            with col3:
                st.write(f"**Generated:** {script['timestamp']}")
            
            if script['sources']:
                st.caption(f"**References:** {', '.join(script['sources'])}")
            
            st.divider()
            
            # Display Images if available
            if script.get('poster') or script.get('scene_images'):
                st.subheader("🎨 Generated Images")
                
                img_cols = st.columns([1, 2])
                
                # Movie Poster
                if script.get('poster'):
                    with img_cols[0]:
                        st.image(script['poster'], caption=f"{script['title']} - Movie Poster", use_container_width=True)
                
                # Scene Images
                if script.get('scene_images'):
                    with img_cols[1]:
                        for i, scene_data in enumerate(script['scene_images']):
                            st.image(
                                scene_data['image'],
                                caption=f"Scene {i+1}: {scene_data['scene']}",
                                use_container_width=True
                            )
                            with st.expander(f"View prompt for Scene {i+1}"):
                                st.code(scene_data['prompt'])
                
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
                # Download script button
                st.download_button(
                    label="💾 Download Script",
                    data=script['screenplay'],
                    file_name=f"{script['title'].replace(' ', '_')}.txt",
                    mime="text/plain",
                    key=f"download_{idx}",
                    use_container_width=True
                )
                
                # Download poster button
                if script.get('poster'):
                    buf = io.BytesIO()
                    script['poster'].save(buf, format='PNG')
                    st.download_button(
                        label="🖼️ Download Poster",
                        data=buf.getvalue(),
                        file_name=f"{script['title'].replace(' ', '_')}_poster.png",
                        mime="image/png",
                        key=f"download_poster_{idx}",
                        use_container_width=True
                    )
                
                # Regenerate images button
                if st.button("🔄 Regenerate Images", key=f"regen_{idx}", use_container_width=True):
                    with st.spinner("Generating images..."):
                        try:
                            method = "stable-diffusion"  # default
                            poster = generate_movie_poster(script['title'], script['genre'], script['screenplay'], method)
                            scenes = generate_scene_images(script['screenplay'], script['genre'], script['title'], method)
                            
                            # Update script with new images
                            script['poster'] = poster
                            script['scene_images'] = scenes
                            st.success("✅ Images regenerated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to regenerate: {e}")

# Footer
st.divider()
st.caption("🎬 Movie Script Generator | Powered by Mistral + RAG + MCP")