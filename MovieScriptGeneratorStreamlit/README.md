# 🎬 Movie Script Generator & Actor Booking System

A powerful Streamlit application that integrates **MCP (Model Context Protocol)**, **Langchain**, and **Mistral AI** with **RAG (Retrieval-Augmented Generation)** to generate professional movie scripts.

## Features

- 🎭 **Actor Booking System**: Manage your cast with detailed actor profiles
- 🤖 **AI-Powered Actor Enrichment**: Automatically fetch actor info using OpenAI (filmography, personality, typical roles)
- 🎬 **Script Generation**: Generate original movie scripts using AI
- 📚 **RAG Integration**: Reference classic film scripts for context and inspiration
- 🤖 **Mistral AI**: Local LLM via Ollama for cost-free, private generation
- 🎨 **Image Generation**: Create movie posters and scene images
- 💾 **Export Scripts**: Download generated scripts as text files
- 📊 **Statistics Tracking**: Monitor actors booked and scripts generated

## Architecture

```
┌─────────────────┐
│  Streamlit UI   │ ← User Interface
└────────┬────────┘
         │
         ├──────────────────┐
         │                  │
┌────────▼────────┐  ┌─────▼──────┐
│  MCP Server     │  │  RAG System │
│  (FastAPI)      │  │  (ChromaDB) │
└────────┬────────┘  └─────┬──────┘
         │                  │
         └────────┬─────────┘
                  │
         ┌────────▼────────┐
         │ Mistral (Ollama)│
         │  Local LLM      │
         └─────────────────┘
```

## Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running
3. **Mistral model** pulled in Ollama

## Installation

### 1. Install Ollama and Mistral

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama
ollama serve

# Pull Mistral model (in another terminal)
ollama pull mistral
```

### 2. Install Python Dependencies

```bash
cd MovieScriptGeneratorStreamlit
conda activate huggingface_env
pip install -r requirements.txt
```

### 3. Verify RAG Database

Make sure your ChromaDB is initialized with movie scripts:

```bash
# From the parent directory
cd ..
python langchain+mistral.py
```

## Usage

### Option 1: Streamlit Only (Recommended)

Run the Streamlit app directly (includes embedded RAG):

```bash
streamlit run streamlit_app.py
```

Then open your browser to: `http://localhost:8501`

### Option 2: With MCP Server Backend

If you want to use the FastAPI backend:

**Terminal 1 - Start MCP Server:**
```bash
python mcp_server.py
```

**Terminal 2 - Start Streamlit:**
```bash
streamlit run streamlit_app.py
```

## How to Use

### 1. Add Actors
- Navigate to the "Actor Booking System" section
- Fill in actor details (name, age, gender, specialty)
- Click "Add Actor"

### 2. Generate Scripts
- Enter a movie title
- Describe your story idea in the prompt
- Select a genre
- (Optional) Select actors from your roster
- Check "Use RAG" to reference classic films
- Click "Generate Script"

### 3. View & Download
- Generated scripts appear in the "Generated Scripts" section
- Review the screenplay
- Download as a text file

## Configuration

Edit these settings in `streamlit_app.py`:

```python
SCRIPTS_ROOT = Path("../GenAIMovie/scripts")  # Path to script database
DB_PATH = Path("../chroma_db")                # ChromaDB location
EMBEDDING_MODEL = "all-MiniLM-L6-v2"          # Embedding model
MISTRAL_MODEL = "mistral"                     # Ollama model name
RETRIEVAL_K = 6                               # Number of reference chunks
```

## API Endpoints (MCP Server)

- `GET /` - Health check
- `POST /actors` - Add actor
- `GET /actors` - List all actors
- `DELETE /actors/{name}` - Remove actor
- `POST /generate-script` - Generate screenplay
- `GET /health` - System health status

## Troubleshooting

### Mistral Not Connected
```bash
# Make sure Ollama is running
ollama serve

# Verify model is available
ollama list
```

### RAG Initialization Failed
```bash
# Rebuild the vector database
cd ..
python langchain+mistral.py
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

## Example Prompts

- "A detective discovers that reality is a simulation, in the style of The Matrix and Blade Runner"
- "A chef must save the world using only cooking skills, comedy style"
- "Space marines fight alien creatures on a distant colony, horror/action blend"

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI (MCP Server)
- **LLM**: Mistral via Ollama
- **Actor Enrichment**: OpenAI GPT-4o-mini
- **Vector DB**: ChromaDB
- **Embeddings**: SentenceTransformers
- **Image Generation**: Stable Diffusion / DALL-E / HuggingFace
- **RAG**: Custom implementation from langchain+mistral.py

## File Structure

```
MovieScriptGeneratorStreamlit/
├── streamlit_app.py           # Main Streamlit application
├── mcp_server.py              # FastAPI MCP server (optional)
├── requirements.txt           # Python dependencies
├── README.md                  # Main documentation
├── QUICKSTART.md              # Quick start guide
├── ACTOR_ENRICHMENT.md        # Actor enrichment guide (NEW!)
├── IMAGE_GENERATION_SETUP.md  # Image generation setup
└── run.sh                     # Launch script
```

## License

© 2025 Movie Script Generator

## Support

For issues or questions, check:
- Ollama documentation: https://ollama.ai
- Streamlit docs: https://docs.streamlit.io
- ChromaDB docs: https://docs.trychroma.com
