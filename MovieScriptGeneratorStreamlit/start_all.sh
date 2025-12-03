#!/bin/bash
# 🎬 Movie Script Generator - Complete Startup Script
# Starts Ollama, MCP Server, and Streamlit App

set -e  # Exit on error

echo "🎬 Movie Script Generator - Complete Startup"
echo "=============================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate conda environment first
echo -e "${YELLOW}🔧 Activating conda environment: huggingface_env${NC}"
eval "$(conda shell.bash hook)"
conda activate huggingface_env

# Step 1: Check Ollama
echo ""
echo -e "${YELLOW}📡 Step 1: Checking Ollama...${NC}"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ollama is running${NC}"
else
    echo -e "${RED}❌ Ollama is NOT running${NC}"
    echo ""
    echo "To start Ollama, run in a separate terminal:"
    echo "  ollama serve"
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
fi

# Step 2: Check Mistral model
echo ""
echo -e "${YELLOW}🤖 Step 2: Checking Mistral model...${NC}"
if ollama list 2>/dev/null | grep -q "mistral"; then
    echo -e "${GREEN}✅ Mistral model found${NC}"
else
    echo -e "${YELLOW}⚠️  Mistral model not found. Would you like to pull it? (y/n)${NC}"
    read -p "> " pull_mistral
    if [[ "$pull_mistral" =~ ^[Yy]$ ]]; then
        ollama pull mistral
    fi
fi

# Step 3: Check Stable Diffusion (optional)
echo ""
echo -e "${YELLOW}🎨 Step 3: Checking Stable Diffusion (optional for images)...${NC}"
if curl -s http://127.0.0.1:7860/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Stable Diffusion (Automatic1111) is running${NC}"
else
    echo -e "${YELLOW}ℹ️  Stable Diffusion not detected (optional - for image generation)${NC}"
    echo "   To enable image generation, start Automatic1111 in separate terminal:"
    echo "   cd ~/stable-diffusion-webui && ./webui.sh --api"
fi

# Step 4: Ask which mode to run
echo ""
echo -e "${YELLOW}🚀 Step 4: Choose startup mode:${NC}"
echo "  1) Streamlit Only (recommended - easiest)"
echo "  2) MCP Server + Streamlit (advanced)"
echo "  3) Streamlit with debug mode"
echo ""
read -p "Select option (1-3): " mode

case $mode in
    1)
        echo ""
        echo -e "${GREEN}🚀 Starting Streamlit App...${NC}"
        echo "📍 App will open at http://localhost:8501"
        echo ""
        streamlit run streamlit_app.py
        ;;
    
    2)
        echo ""
        echo -e "${GREEN}🚀 Starting MCP Server + Streamlit...${NC}"
        echo ""
        
        # Start MCP server in background
        echo "Starting MCP Server on port 8000..."
        python mcp_server.py > mcp_server.log 2>&1 &
        MCP_PID=$!
        echo "MCP Server PID: $MCP_PID"
        
        # Wait for MCP to start
        sleep 3
        
        # Check if MCP is running
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ MCP Server running at http://localhost:8000${NC}"
        else
            echo -e "${RED}❌ MCP Server failed to start. Check mcp_server.log${NC}"
        fi
        
        echo ""
        echo "Starting Streamlit App on port 8501..."
        streamlit run streamlit_app.py
        
        # Cleanup on exit
        trap "echo 'Stopping MCP Server...'; kill $MCP_PID 2>/dev/null" EXIT
        ;;
    
    3)
        echo ""
        echo -e "${GREEN}🔍 Starting Streamlit in Debug Mode...${NC}"
        echo "📍 App will open at http://localhost:8501"
        echo ""
        streamlit run streamlit_app.py --server.runOnSave=true --logger.level=debug
        ;;
    
    *)
        echo -e "${RED}Invalid option. Exiting.${NC}"
        exit 1
        ;;
esac
