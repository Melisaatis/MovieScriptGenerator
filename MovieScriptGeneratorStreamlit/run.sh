#!/bin/bash
# 🎬 Movie Script Generator - Launch Script

echo "🎬 Starting Movie Script Generator..."
echo ""

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama is not running!"
    echo "Please start Ollama in another terminal with: ollama serve"
    echo ""
    read -p "Press Enter to continue anyway or Ctrl+C to exit..."
fi

# Check if mistral model is available
echo "🤖 Checking for Mistral model..."
if ollama list | grep -q "mistral"; then
    echo "✅ Mistral model found"
else
    echo "⚠️  Mistral model not found. Pulling now..."
    ollama pull mistral
fi

echo ""
echo "🚀 Launching Streamlit app..."
echo "📍 App will open at http://localhost:8501"
echo ""

# Activate conda environment
echo "Activating conda environment: huggingface_env"
eval "$(conda shell.bash hook)"
conda activate huggingface_env

# Run Streamlit
streamlit run streamlit_app.py
