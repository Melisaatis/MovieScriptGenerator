# 🎨 Image Generation Setup Guide

Your Movie Script Generator now supports generating movie posters and scene images! Here are your options:

## Option 1: Stable Diffusion (Local) - RECOMMENDED ⭐

### Using Automatic1111 WebUI

**Pros:** Free, fast, runs locally, no API limits
**Cons:** Requires GPU with 4GB+ VRAM

1. **Install Automatic1111:**
```bash
# Clone the repository
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# Install (Mac)
./webui.sh --api

# Install (Linux/Windows)
./webui.sh --api  # or webui-user.bat on Windows
```

2. **Start with API enabled:**
```bash
./webui.sh --api --listen
```

3. **Verify it's running:**
- Open browser: http://127.0.0.1:7860
- API will be at: http://127.0.0.1:7860/docs

4. **The Streamlit app will automatically detect it!**

---

## Option 2: Hugging Face Inference API

**Pros:** No setup, cloud-based, good quality
**Cons:** Rate limited (free tier), requires internet

1. **Get HuggingFace API Token:**
   - Go to: https://huggingface.co/settings/tokens
   - Create a new token (Read access)
   - Copy the token

2. **In Streamlit App:**
   - Open sidebar
   - Find "🤗 Hugging Face API" section
   - Paste your token
   - Select "Hugging Face Inference" as image method

---

## Option 3: Diffusers Library (Local Python)

**Pros:** Pure Python, good control
**Cons:** Slower than A1111, requires setup

1. **Install dependencies:**
```bash
conda activate huggingface_env
pip install diffusers torch torchvision transformers accelerate
```

2. **Download model (first time only):**
```python
from diffusers import StableDiffusionPipeline

# This will download ~4GB
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
```

3. **Enable in Streamlit (coming soon)**

---

## Option 4: Replicate API (Cloud)

**Pros:** High quality SDXL, no local setup
**Cons:** Costs money, requires API key

1. **Get API Key:**
   - Sign up: https://replicate.com
   - Get API token from account settings

2. **Set environment variable:**
```bash
export REPLICATE_API_TOKEN="your-token-here"
```

---

## Option 5: OpenAI DALL-E

**Pros:** Very high quality, reliable
**Cons:** Costs money (~$0.02-0.04 per image)

1. **Get OpenAI API Key:**
   - https://platform.openai.com/api-keys

2. **Set environment variable:**
```bash
export OPENAI_API_KEY="your-key-here"
```

---

## Quick Start (Recommended Path)

### For Mac Users (M1/M2/M3):

```bash
# 1. Install Automatic1111
cd ~
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# 2. Start with API
./webui.sh --api --skip-torch-cuda-test

# 3. In another terminal, start Streamlit
cd /Users/melisaatis/Desktop/MovieScriptGenerator/MovieScriptGeneratorStreamlit
conda activate huggingface_env
streamlit run streamlit_app.py

# 4. Generate a script and check "Generate Images"!
```

### For Users Without GPU:

Use Hugging Face Inference API (Option 2) - it's free and works great!

---

## Troubleshooting

### "Stable Diffusion not detected"
- Make sure A1111 is running with `--api` flag
- Check http://127.0.0.1:7860 is accessible
- Verify no firewall is blocking port 7860

### "Image generation timeout"
- First generation is slow (model loading)
- Increase timeout in code or use faster model
- Consider using Hugging Face API instead

### "Out of memory"
- Reduce image size (512x512 instead of 768x1024)
- Use CPU instead of GPU (slower but works)
- Close other applications

### Images look bad
- Adjust the prompts in the code
- Try different Stable Diffusion models
- Increase steps (30-50 is good balance)

---

## What Gets Generated

### Movie Poster (1 image)
- Professional theatrical poster style
- Title treatment
- Genre-appropriate mood
- Cinematic composition

### Scene Images (3 images)
- Key scenes from your screenplay
- Extracted from INT./EXT. headers
- Genre-matched atmosphere
- Dramatic lighting

---

## Tips for Best Results

1. **Use descriptive prompts** - The script quality affects image quality
2. **Be specific about genre** - Helps with mood and styling
3. **Local is faster** - Automatic1111 beats cloud APIs
4. **Experiment with models** - Different SD models have different styles
5. **Save good images** - Download and keep your favorites!

---

## Model Recommendations

### For Movie Posters:
- Stable Diffusion 2.1 (good all-around)
- SDXL (highest quality, slower)
- DreamShaper (artistic style)

### For Scene Images:
- Realistic Vision (photorealistic)
- Deliberate (cinematic look)
- SDXL (best quality)

### For Speed:
- SD 1.5 based models (fastest)
- Lower resolution (512x512)
- Fewer steps (20-25)

---

## Cost Comparison

| Method | Cost | Speed | Quality |
|--------|------|-------|---------|
| A1111 Local | Free | Fast | Great |
| HF Inference | Free* | Medium | Good |
| Replicate | ~$0.01/img | Fast | Excellent |
| DALL-E 3 | ~$0.04/img | Fast | Excellent |
| Diffusers Local | Free | Slow | Great |

*Rate limited on free tier

---

## Support

For issues with:
- **A1111 Setup**: https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues
- **HF API**: https://huggingface.co/docs/api-inference
- **Streamlit App**: Check the main README.md

---

Happy Image Generation! 🎨🎬
