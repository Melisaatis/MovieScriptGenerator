# 🎬 Quick Start Guide

## Get Started in 3 Steps

### Step 1: Start Ollama & Mistral
```bash
# Terminal 1
ollama serve

# Terminal 2
ollama pull mistral  # Only needed first time
```

### Step 2: Install Dependencies
```bash
cd MovieScriptGeneratorStreamlit
conda activate huggingface_env
pip install -r requirements.txt
```

### Step 3: Run the App
```bash
# Easy way
./run.sh

# Or manually
streamlit run streamlit_app.py
```

## First Time Setup

If you haven't built the RAG database yet:

```bash
cd /Users/melisaatis/Desktop/MovieScriptGenerator
python langchain+mistral.py
```

This will create the ChromaDB vector database from your movie scripts.

## Using the App

### Book Actors
1. Go to "Actor Booking System"
2. Add actor name, age, gender, specialty
3. Click "➕ Add Actor"

### Generate Scripts
1. Enter a movie title
2. Write your story idea
3. Select genre
4. Choose actors (optional)
5. Enable "Use RAG" for better results
6. Click "🎬 Generate Script"

### Example Story Ideas

**Sci-Fi:**
"A rogue AI on a spaceship begins eliminating crew members one by one"

**Action:**
"An ex-special forces soldier must rescue hostages from a skyscraper"

**Comedy:**
"A time traveler accidentally changes history and must fix it before disappearing"

**Horror:**
"A group of friends discover an ancient evil in a cabin in the woods"

## Tips for Best Results

✅ **DO:**
- Use RAG for better quality scripts
- Be specific in your prompts
- Reference specific movie styles you like
- Select appropriate actors for your genre

❌ **DON'T:**
- Make prompts too vague
- Forget to check Ollama is running
- Generate without selecting a genre

## Troubleshooting

**Problem:** "Mistral Not Connected"
```bash
# Solution
ollama serve
```

**Problem:** "RAG Failed to Initialize"
```bash
# Solution
cd ..
python langchain+mistral.py
```

**Problem:** Scripts are low quality
```bash
# Solution
- Enable "Use RAG"
- Make prompts more specific
- Try different genres
- Reference classic films in your prompt
```

## Need Help?

Check the full README.md for detailed documentation.

---
🎬 Happy Scriptwriting!
