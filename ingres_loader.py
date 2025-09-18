import json

def load_ingres_data():
    docs = [
        {"id": "1", "text": "INGRES is an organization focusing on research and student development."},
        {"id": "2", "text": "INGRES organizes events, workshops, and hackathons."}
    ]
    with open("ingres_docs.json", "w") as f:
        json.dump(docs, f, indent=2)
    print("✅ Data saved to ingres_docs.json")

if __name__ == "__main__":
    load_ingres_data()