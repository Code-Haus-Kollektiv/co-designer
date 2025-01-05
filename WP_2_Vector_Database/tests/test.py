import json
import os
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling - Take attention mask into account for correct averaging
    """
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    # Expand the attention mask
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Sum the embeddings, then divide by the sum of attention mask
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

def test_chroma_with_json(json_file_path: str, persist_directory: str = "./my_chroma_db"):
    """
    Demonstration of using ChromaDB to embed one specific JSON file.
    """

    # 1. Initialize a persistent client (stores data on disk)
    client = chromadb.PersistentClient(path=persist_directory)

    # 2. Define the embedding function
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    # 3. Get or create a collection
    collection_name = "test_collection2"
    try:
        collection = client.get_collection(
            name=collection_name, 
            embedding_function=default_ef
        )
        print(f"Collection '{collection_name}' found.")
    except Exception:
        collection = client.create_collection(
            name=collection_name, 
            embedding_function=default_ef
        )
        print(f"Collection '{collection_name}' created.")

    # 4. Load the single JSON file
    if not os.path.isfile(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded JSON data from '{json_file_path}':")
    
    # 5. Prepare the data and generate embeddings
    doc_id = os.path.basename(json_file_path)  # Use the filename as the ID
    doc_text = json.dumps(data, ensure_ascii=False)

    # Initialize tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Tokenize
    encoded_input = tokenizer(
        doc_text,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    # Compute embeddings with no_grad
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1).numpy()

    print(sentence_embeddings)
    # 6. Add to the collection
    # Note: Because doc_text is a single string, we pass a list with one element:
    collection.add(
        documents=[doc_text],
        ids=[doc_id],
        embeddings=sentence_embeddings

    )
    

    
    # 7. Retrieve collection contents to verify
    results = collection.get(include=["embeddings"]
)
    print("\nCollection contents after insertion:")
    print(results)

if __name__ == "__main__":
    # Update with your actual JSON file path to test
    json_file_path = r"json_chunks\object_1.json"
    test_chroma_with_json(json_file_path)
