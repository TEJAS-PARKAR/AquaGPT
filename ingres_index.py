import os
import json
import chromadb
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Use a single directory for all storage
PERSIST_DIR = "./ingres_storage"

def build_index():
    # Initialize ChromaDB client pointing to a path inside the persist_dir
    db = chromadb.PersistentClient(path=os.path.join(PERSIST_DIR, "chroma_db"))
    chroma_collection = db.get_or_create_collection("ingres_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create the storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load documents
    with open("ingres_docs.json", "r") as f:
        raw_docs = json.load(f)
    documents = [Document(text=d["text"], metadata={"id": d["id"]}) for d in raw_docs]

    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build the index from documents
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )

    # The index is already linked to the persistent ChromaDB, no separate persist call is needed.
    print(f"✅ Index built and stored in {PERSIST_DIR}")

if __name__ == "__main__":
    build_index()