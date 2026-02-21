# NITDA RAG Assistant

This is a Gradio-based Retrieval-Augmented Generation (RAG) app that answers questions about NITDA based on official documents.

## How it works
- Loads the Mistral GGUF model via `llama-cpp-python`.
- Splits the NITDA Act PDF into chunks.
- Embeds chunks using `sentence-transformers`.
- Stores embeddings in a Chroma vector database.
- Retrieves relevant context and generates answers.

## Files
- `app.py` → Main Gradio app script.
- `requirements.txt` → Dependencies list.
- `NITDA-ACT-2007-2019-Edition1.pdf` → Official NITDA Act document.

## Running locally
```bash
pip install -r requirements.txt
python app.py
