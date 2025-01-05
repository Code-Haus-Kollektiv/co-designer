import os
import chromadb
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings


# Constants
PERSIST_DIRECTORY = "./output/co-designer_db"  # Update this path to your SQLite database location
COLLECTION_NAME = "co-designer_collection"
LLM_MODEL = "llama3.2:latest"
OLLAMA_BASE_URL = "http://localhost:11434"  # Adjust based on your Ollama server configuration

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Check if the collection exists
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' found.")
except chromadb.errors.InvalidCollectionException:
    # If the collection does not exist, create it with the embedding function
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    print(f"Collection '{COLLECTION_NAME}' created.")

# Function to query the ChromaDB collection
def query_chromadb(query_text, n_results=1):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results["documents"], results["metadatas"]

# Function to interact with the Ollama LLM
def query_ollama(prompt):
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    return llm.invoke(prompt)

# RAG pipeline: Combine ChromaDB and Ollama for Retrieval-Augmented Generation
def rag_pipeline(query_text):
    # Step 1: Retrieve relevant documents from ChromaDB
    retrieved_docs, metadata = query_chromadb(query_text)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."

    # Step 2: Send the query along with the context to Ollama
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    print("######## Augmented Prompt ########")
    # print(augmented_prompt)

    response = query_ollama(augmented_prompt)
    return response

# Example usage
query = "What should the area component connect to?"
response = rag_pipeline(query)
print("######## Response from LLM ########\n", response)
