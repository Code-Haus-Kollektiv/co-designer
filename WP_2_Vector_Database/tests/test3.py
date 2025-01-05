import json
import logging
import os
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.api.models import Collection
from chromadb.errors import InvalidCollectionException
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Constants
PERSIST_DIRECTORY = "./my_chroma_db"
COLLECTION_NAME = "my_collection"
JSON_FOLDER = "./json_chunks"
EXPORT_FILE = "my_collection_export.json"

# Example: Using a SentenceTransformer-based embedding function from Chroma’s helpers
DEFAULT_EF = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


logging.info("Initializing persistent Chroma client with directory: %s", PERSIST_DIRECTORY)
client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# Try getting or creating the collection
logging.info("Getting or creating collection: %s", COLLECTION_NAME)
try:
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=DEFAULT_EF)
    logging.info("Collection '%s' found.", COLLECTION_NAME)
except InvalidCollectionException:
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=DEFAULT_EF)
    logging.info("Collection '%s' created.", COLLECTION_NAME)

# Load all JSON files from the JSON_FOLDER
logging.info("Loading JSON files from folder: %s", JSON_FOLDER)
json_data_list = []
for file_name in os.listdir(JSON_FOLDER):
    if file_name.endswith(".json"):
        file_path = os.path.join(JSON_FOLDER, file_name)
        logging.debug("Loading file: %s", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                data["__source_file"] = file_name  # store filename in data
                json_data_list.append(data)
            except json.JSONDecodeError as e:
                logging.warning("Failed to decode JSON from %s: %s", file_path, e)

logging.info("Loaded %d JSON items.", len(json_data_list))

# Add each JSON item to the collection
for data_item in json_data_list:
    # Create an ID from "@index" or fallback to filename
    doc_id = str(data_item.get("@index", "")) or data_item["__source_file"]
    # Convert dict to JSON string
    doc_text = json.dumps(data_item, ensure_ascii=False)
    collection.add(
        documents=[doc_text],
        ids=[doc_id],
    )
    logging.debug("Added doc_id '%s' to the collection.", doc_id)

# Export the entire collection (optional)
logging.info("Exporting the collection to %s", EXPORT_FILE)
# collection_data = collection.get(include=["embeddings"])
collection_data = collection.get(include=["embeddings", "documents"])

# If embeddings exist, convert NumPy arrays to lists
if "embeddings" in collection_data:
    collection_data["embeddings"] = [
        embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        for embedding in collection_data["embeddings"]
    ]

with open(EXPORT_FILE, "w", encoding="utf-8") as out_file:
    json.dump(collection_data, out_file, indent=2)

logging.info("Collection exported to %s", EXPORT_FILE)
logging.info("Done.")
