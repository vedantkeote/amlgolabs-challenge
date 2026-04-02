import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

class RAGPipeline:
    def __init__(self):
        # 1. Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Load Vector DB
        db_path = os.path.abspath(os.path.join(os.getcwd(), "..", "vectordb"))
        self.vector_db = FAISS.load_local(
            db_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True # Required for local FAISS loading
        )
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        
        self.llm = OllamaLLM(model="mistral", num_ctx=2048)
        
        template = """You are a helpful AI assistant. Use the following context to answer the user's question. 
If you don't know the answer based on the context, just say that you don't know. Do not make up information.

Context:
{context}

Question: {question}
Answer:"""
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    def stream_response(self, query):
        """
        Retrieves documents, formats the prompt, and yields the response chunk by chunk.
        Also returns the source documents for display.
        """
        retrieved_docs = self.retriever.invoke(query)
        
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        formatted_prompt = self.prompt.format(context=context, question=query)
        
        response_stream = self.llm.stream(formatted_prompt)
        
        return response_stream, retrieved_docs