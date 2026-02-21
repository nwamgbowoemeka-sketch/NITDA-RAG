import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import gradio as gr

# -----------------------------
# 1. Load Mistral GGUF model
# -----------------------------

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_basename = "mistral-7b-instruct-v0.2.Q6_K.gguf"

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

lcpp_llm = Llama(
    model_path=model_path,
    n_threads=4,
    n_batch=256,
    n_gpu_layers=32,
    n_ctx=4096
)

# -----------------------------
# 2. Load NITDA Act PDF
# -----------------------------

# Uncomment and run the below code snippets if the dataset is present in the Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define the path to your PDF file in Google Drive
pdf_path = "/content/drive/MyDrive/NITDA-ACT-2007-2019-Edition1.pdf"
#manual_pdf_path = "user_manual.pdf"   # just the file name
# Load the PDF using PyMuPDFLoader
pdf_loader = PyMuPDFLoader(pdf_path)
documents = pdf_loader.load()

#pdf_path = "NITDA-ACT-2007-2019-Edition1.pdf"  # Upload this file to your Hugging Face Space
#pdf_loader = PyMuPDFLoader(pdf_path)
#documents = pdf_loader.load()

# -----------------------------
# 3. Split into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name='cl100k_base',
    chunk_size=1000,
    chunk_overlap=150
)
document_chunks = pdf_loader.load_and_split(text_splitter)

# -----------------------------
# 4. Embeddings + Vectorstore
# -----------------------------
embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

out_dir = "nitda_db"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

vectorstore = Chroma.from_documents(
    documents=document_chunks,
    embedding=embedding_model,
    persist_directory=out_dir
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# -----------------------------
# 5. RAG Response Function
# -----------------------------
system_message = (
    "You are an AI assistant specialized in NITDA's information retrieval. "
    "Provide accurate, evidence-based answers strictly from authoritative NITDA sources. "
    "Do not give personal information. If documents do not contain the answer, say so."
)

qna_template = """Context:
{context}

Question:
{question}

Answer:"""

def generate_rag_response(user_input, max_tokens=256, temperature=0.01, top_p=0.95, top_k=50):
    relevant_docs = retriever.get_relevant_documents(user_input)
    context = " ".join([doc.page_content for doc in relevant_docs])

    prompt = system_message + "\n" + qna_template.format(context=context, question=user_input)

    try:
        response = lcpp_llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"Error: {e}"

# -----------------------------
# 6. Gradio Interface
# -----------------------------
def chat_fn(query):
    return generate_rag_response(query)

iface = gr.Interface(
    fn=chat_fn,
    inputs=gr.Textbox(lines=2, placeholder="Ask about NITDA..."),
    outputs="text",
    title="NITDA RAG Assistant",
    description="Ask questions about NITDA based on official documents."
)

if __name__ == "__main__":
    iface.launch()


gradio
huggingface_hub
llama-cpp-python
langchain
sentence-transformers
chromadb
pymupdf



