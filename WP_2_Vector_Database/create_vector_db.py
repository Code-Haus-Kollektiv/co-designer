import json
import logging
import os
from typing import Any, Dict, List
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.api.models import Collection
from chromadb.errors import InvalidCollectionException
from chromadb.utils import embedding_functions
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Configure logging with color
class CustomFormatter(logging.Formatter):
    def format(self, record):
        level_colors = {
            "INFO": Fore.GREEN,
            "DEBUG": Fore.CYAN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.MAGENTA,
        }
        level_color = level_colors.get(record.levelname, "")
        message = super().format(record)
        return f"{level_color}{message}{Style.RESET_ALL}"

formatter = CustomFormatter("%(asctime)s [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO, handlers=[handler])

# Constants
NAME = "co-designer"
PERSIST_DIRECTORY = f"./output/{NAME}_db"
COLLECTION_NAME = f"{NAME}_collection"
JSON_FOLDER = "./json_chunks"
EXPORT_FILE = f"./output/{COLLECTION_NAME}_export.json"

# Embedding function
DEFAULT_EF = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def initialize_persistent_client(persist_directory: str) -> chromadb.Client:
    logging.info("Initializing persistent Chroma client with directory: %s", persist_directory)
    return chromadb.PersistentClient(path=persist_directory)

def get_or_create_collection(client: chromadb.Client, collection_name: str, embedding_fn) -> Collection:
    try:
        collection = client.get_collection(name=collection_name, embedding_function=embedding_fn)
        logging.info("Collection '%s' found.", collection_name)
    except InvalidCollectionException:
        collection = client.create_collection(name=collection_name, embedding_function=embedding_fn)
        logging.info("Collection '%s' created.", collection_name)
    return collection

def load_json_files(folder_path: str) -> List[Dict[str, Any]]:
    json_data_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            logging.debug("Loading file: %s", file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    data["__source_file"] = file_name
                    json_data_list.append(data)
                except json.JSONDecodeError as e:
                    logging.warning("Failed to decode JSON from %s: %s", file_path, e)
    return json_data_list

def add_to_collection(collection: Collection, data_item: Dict[str, Any]) -> None:
    doc_id: str = str(data_item.get("@index", "")) or data_item["__source_file"]
    doc_text = json.dumps(data_item, ensure_ascii=False)
    collection.add(documents=[doc_text], ids=[doc_id])
    logging.debug("Added doc_id '%s' to the collection.", doc_id)

def export_collection(collection: Collection, export_path: str) -> None:
    logging.info("Exporting collection to %s", export_path)
    collection_data = collection.get(include=["embeddings", "documents", "metadatas"])
    if "embeddings" in collection_data:
        collection_data["embeddings"] = [
            embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            for embedding in collection_data["embeddings"]
        ]
    with open(export_path, "w", encoding="utf-8") as out_file:
        json.dump(collection_data, out_file, indent=2)
    logging.info("Collection exported to %s", export_path)

def main() -> None:
    client = initialize_persistent_client(PERSIST_DIRECTORY)
    collection = get_or_create_collection(client, COLLECTION_NAME, DEFAULT_EF)
    json_data = load_json_files(JSON_FOLDER)
    logging.info("Loaded %d JSON items.", len(json_data))
    for item in json_data:
        add_to_collection(collection, item)
    export_collection(collection, EXPORT_FILE)
    logging.info("Done.")

if __name__ == "__main__":
    main()
