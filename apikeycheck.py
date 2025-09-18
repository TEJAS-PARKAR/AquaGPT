import os
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")

headers = {"Authorization": f"Bearer {api_key}"}
response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)

if response.status_code == 200:
    print("✅ HuggingFace API key is valid!")
    print("User info:", response.json())
else:
    print("❌ Invalid HuggingFace API key or network error.")
    print("Status code:", response.status_code)
    print("Response:", response.text)