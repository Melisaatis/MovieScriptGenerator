Project Summary

The MovieScriptGenerator is a comprehensive AI-powered screenplay generation system that combines Retrieval-Augmented Generation (RAG) with large language model to produce movie scripts. The project implements a microservices architecture with a FastAPI backend server and an interactive Streamlit web application, integrating actor enrichment, vector-based semantic search, and image generation capabilities. 

The MovieScriptGenerator aims to: 

•	Generate original, genre-specific movie scripts referenced by old movies. Generated scripts are tailored to user provided prompts and actor selections. 
•	Allow users to select actors for their scripts, with enriched actor profiles fetched via OpenAI or TMDB.
•	Generate movie posters and scene images using diffusion model. 
•	Use Retrieval Augmented Generation (RAG) to pull reference scenes from a database of older movie scripts for stylistic inspiration.
•	Offer a user-friendly Streamlit interface for script generation, actor booking, and image generation.
•	Scrape and organize older movie scripts by genre to build a dataset for training or fine-tuning AI models.
•	Enable integration with external APIs (e.g., TMDB, OpenAI) and local models (e.g., Stable Diffusion) for flexible and scalable functionality.

