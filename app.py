import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="Legal Document Chatbot", page_icon="⚖️", layout="wide")


@st.cache_resource
def load_pipeline():
    return RAGPipeline()

with st.spinner("Loading AI Model and Vector Database..."):
    pipeline = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.title("Chatbot Settings")
st.sidebar.markdown("**Current Model:** Mistral-7B (Local)")
st.sidebar.markdown("---")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.title("Document QA Assistant")
st.markdown("Ask questions based on the provided legal document. The AI will stream answers directly from the text.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.info(f"**Source {i+1}:**\n{source.page_content}")

if prompt := st.chat_input("Enter your query here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        response_stream, sources = pipeline.stream_response(prompt)
        
        for chunk in response_stream:
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
            
        response_placeholder.markdown(full_response)
        
        with st.expander("View Sources"):
            for i, source in enumerate(sources):
                st.info(f"**Source {i+1}:**\n{source.page_content}")
                
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "sources": sources
    })