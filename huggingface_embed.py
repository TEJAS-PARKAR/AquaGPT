import os
from huggingface_hub import InferenceClient
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Custom Hugging Face embedding wrapper
class HuggingFaceEmbedding:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.client = InferenceClient(api_key=os.getenv("HUGGINGFACE_API_KEY"))
        self.model_name = model_name

    def get_text_embedding(self, text: str):
        """Return embedding vector for a given text"""
        result = self.client.feature_extraction(text, model=self.model_name)
        return result

# Example: building index
def build_index():
    # Initialize ChromaDB
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("huggingface_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Example documents
    documents = [
        Document(text="PCCOE Ingres is a student-driven tech event."),
        Document(text="It includes workshops, hackathons, and seminars."),
    ]

    # Use Hugging Face embeddings
    embedder = HuggingFaceEmbedding()
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context,
        embed_model=embedder.get_text_embedding
    )

    index.storage_context.persist()
    print("✅ Index built with Hugging Face embeddings.")

if __name__ == "__main__":
    build_index()
