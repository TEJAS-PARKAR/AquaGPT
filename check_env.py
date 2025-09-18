import sys
import pkg_resources

print("Checking Python environment...\n")

# Show which Python interpreter is being used
print(f"Python executable: {sys.executable}")
print(f"Python version   : {sys.version}\n")

# Packages we care about
packages = [
    "llama-index",
    "llama-index-core",
    "llama-index-embeddings-huggingface",
    "llama-index-vector-stores-chroma",
    "chromadb",
    "sentence-transformers",
    "flask",
]

for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f"✅ {pkg}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"❌ {pkg}: NOT INSTALLED")

print("\n🎉 Environment check complete!")
