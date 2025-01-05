import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import requests
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Constants
PERSIST_DIRECTORY = "./output/co-designer_db"
COLLECTION_NAME = "co-designer_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Same as your collection
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama API endpoint
MODEL_NAME = "llama3.2:latest"  # Replace with the Ollama model you're using (e.g., "llama2", "gpt-4")

# Initialize Chroma client
def initialize_chroma_client(persist_directory: str) -> chromadb.Client:
    return chromadb.PersistentClient(path=persist_directory)

# Query ChromaDB for relevant context
def query_chromadb(collection, query_text: str, top_k: int = 5):
    """
    Query ChromaDB to fetch top-k relevant documents.
    """
    logging.info("Querying ChromaDB with text: %s", query_text)
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
    )
    return results

# Use Ollama for response generation
def generate_response_with_ollama(model_name: str, prompt: str):
    """
    Send a prompt to the Ollama API and get the response.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
    }
    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response from the model.")
    except requests.RequestException as e:
        logging.error("Error communicating with Ollama API: %s", e)
        return "Error generating response."

def generate_response(collection, user_query: str):
    """
    Generate a response using Ollama and context retrieved from ChromaDB.
    """
    # Query the database for relevant context
    results = query_chromadb(collection, user_query)

    # Extract documents and format them for the prompt
    documents = results.get("documents", [[]])[0]  # Get top-k docs
    context = "\n".join(documents)

    # Construct the prompt
    prompt = f"""
    You are an expert assistant with access to the following context. Use it to answer the user's query as accurately as possible:
    
    Context:
    {context}

    User Query:
    {user_query}
    """
    logging.info("Generated prompt for Ollama:\n%s", prompt)

    # Generate response using Ollama
    response = generate_response_with_ollama(MODEL_NAME, prompt)
    return response

def main():
    # Initialize the persistent Chroma client
    client = initialize_chroma_client(PERSIST_DIRECTORY)

    # Get the ChromaDB collection
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)

    # User input
    user_query = input("Enter your query: ")

    # Generate response
    response = generate_response(collection, user_query)

    # Output the response
    print("\nOllama Response:")
    print(response)

if __name__ == "__main__":
    main()
