import os
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import pipeline

# Use the same single directory
PERSIST_DIR = "./ingres_storage"

def run_chat():
    if not os.path.exists(PERSIST_DIR):
        print(f"Error: Storage directory '{PERSIST_DIR}' not found.")
        print("Please run 'python ingres_index.py' first.")
        return

    try:
        # Initialize ChromaDB client pointing to the stored path
        db = chromadb.PersistentClient(path=os.path.join(PERSIST_DIR, "chroma_db"))
        chroma_collection = db.get_collection("ingres_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Setup embedding model
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Setup local pipeline for text generation
        llm = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device="cpu"  # or "cuda" if you have GPU
        )

        # Create service context with our models
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model
        )

        # Load the index from the vector store with our service context
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            service_context=service_context
        )
        
        query_engine = index.as_query_engine()

        print("💬 Chatbot ready! Type 'exit' to quit.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye!")
                break

            response = query_engine.query(user_input)
            print(f"Bot: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Ensure you have run 'ingres_loader.py' and 'ingres_index.py' successfully.")

if __name__ == "__main__":
    run_chat()