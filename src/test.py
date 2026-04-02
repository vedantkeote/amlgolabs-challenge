from rag_pipeline import RAGPipeline

def main():
    print("Initializing RAG Pipeline. Please wait...")
    pipeline = RAGPipeline()
    print("Pipeline ready. Type 'exit' to quit.\n")

    while True:
        query = input("\nYour Query: ")
        
        if query.lower() in ['exit', 'quit']:
            print("Exiting.")
            break
            
        if not query.strip():
            continue

        print("\nResponse: ", end="", flush=True)
        
        response_stream, sources = pipeline.stream_response(query)
        
        for chunk in response_stream:
            print(chunk, end="", flush=True)
            
        print("\n\n--- Retrieved Sources ---")
        for i, doc in enumerate(sources):
            print(f"Source {i+1}:\n{doc.page_content}\n")

if __name__ == "__main__":
    main()