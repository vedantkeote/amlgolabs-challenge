
# Local RAG Chatbot with Streaming Responses

This project implements a local, CPU-friendly Retrieval-Augmented Generation (RAG) chatbot designed to answer user queries based on a provided legal document. It features a completely local pipeline using FAISS, Hugging Face embeddings, and Ollama, served through a real-time Streamlit interface.

![Recording 2026-04-03 164625](https://github.com/user-attachments/assets/bf3ab039-589d-4dc8-8d8e-5aeff193b3ce)

## Project Architecture and Flow

The pipeline is structured into two main phases: Data Ingestion and the Inference (RAG) Pipeline.

1. **Document Ingestion (`src/ingest.py`)**
   * **Loading:** The PDF document is parsed using LangChain's `PyPDFLoader`.
   * **Chunking:** The extracted text is split into segments of 100-300 words using a `RecursiveCharacterTextSplitter` with an overlap of 50 characters to preserve sentence boundaries and context.
   * **Embedding & Storage:** The chunks are converted into dense vector embeddings and stored locally using a FAISS vector database. The raw chunks are also saved as a JSON file for easy inspection.

2. **Inference Pipeline (`src/rag_pipeline.py` & `app.py`)**
   * **Retrieval:** When a user submits a query, it is embedded using the same model, and FAISS performs a similarity search to retrieve the top 3 most relevant document chunks.
   * **Augmentation:** The retrieved chunks are injected into a strict prompt template alongside the user's query to ensure grounded, factual answers.
   * **Generation & Streaming:** The augmented prompt is sent to a local LLM, which generates the answer. The response is streamed token-by-token back to the Streamlit frontend in real-time, accompanied by the source text passages used.

## Model and Embedding Choices

This project is optimized to run locally on a standard machine (e.g., 16GB RAM, CPU-only) without requiring cloud APIs or dedicated GPUs.

* **Embedding Model:** `all-MiniLM-L6-v2` (via Hugging Face)
  * *Reasoning:* This model is highly efficient, extremely fast on CPUs, and specifically trained for semantic search and mapping sentences to a dense vector space. It is much more suited for building a search index than a generative LLM.
* **Generative LLM:** `Mistral-7B-Instruct` (via Ollama, 4-bit Quantized)
  * *Reasoning:* Mistral-7B offers exceptional reasoning capabilities for its size. By running a quantized version through Ollama, the memory footprint is reduced to under 8GB, allowing it to run comfortably alongside the operating system on a 16GB RAM machine while maintaining high instruction-following accuracy.

## Prerequisites and Setup

1. **Install Ollama:** Download and install Ollama from [ollama.com](https://ollama.com).
2. **Pull the Model:** Open your terminal and run:
   ```bash
   ollama run mistral
   ```
   *Note: Keep Ollama running in the background while using this application.*
3. **Clone github repository:**
   You can clone the repository by running
   ```bash
   git clone https://github.com/vedantkeote/amlgolabs-challenge
   ```
4. **Install Python Dependencies:**
   Ensure you are using Python 3.10+ and run:
   ```bash
   pip install -r requirements.txt
   ```

## Steps to Run the Pipeline

### Step 1: Preprocessing and Vector Database Creation
Before running the chatbot, you must process the provided document and build the search index.

1. Place your source document in the `data/` folder and name it `AITrainingDocument.pdf`.
2. Run the ingestion script:
   ```bash
   python ./src/ingest.py
   ```
   *This will generate a `document_chunks.json` file in the `chunks/` folder and build the index in the `vectordb/` folder.*

### Step 2: Run the Chatbot Interface
Once the database is built, launch the Streamlit application.

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. The interface will open in your default web browser.

## Sample Queries

Here are a few sample queries you can use to test the chatbot's retrieval and grounding capabilities:

* **Query 1:** "What are the main obligations of the user mentioned in the terms?"
* **Query 2:** "Under what conditions can the contract be terminated?"
* **Query 3:** "How is user data handled and stored according to the privacy policy?"

*(Note to Reviewer: Please see the accompanying PDF report for a detailed breakdown of success and failure cases, along with screenshots/video links demonstrating the real-time streaming output).*
