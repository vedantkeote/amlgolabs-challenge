# src/ingest.py
import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_db():
    print("Loading document...")
    file_path = os.path.abspath(os.path.join(os.getcwd(), ".", "data", "AITrainingDocument.pdf"))
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    #saving the chunks for inspection
    print("Saving processed text segments to /chunks...")
    chunks_dir = os.path.abspath(os.path.join(os.getcwd(), ".", "chunks"))
    chunks_file_path = os.path.join(chunks_dir, "document_chunks.json")
    
    chunks_data = [
        {"chunk_id": i, "page_content": chunk.page_content, "metadata": chunk.metadata} 
        for i, chunk in enumerate(chunks)
    ]
    
    with open(chunks_file_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=4)
        
    print(f"Successfully saved {len(chunks)} chunks to {chunks_file_path}")

    print("Generating embeddings and building FAISS index (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)

    save_path = os.path.abspath(os.path.join(os.getcwd(), ".", "vectordb"))
    vector_db.save_local(save_path)
    print(f"Success! Vector database saved to {save_path}")

if __name__ == "__main__":
    build_vector_db()