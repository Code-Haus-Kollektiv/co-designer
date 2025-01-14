import json
import logging
import os
from typing import Any, Dict, List
import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from colorama import Fore, Style, init
import chromadb
import pprint

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
NAME = "co-designer2"
PERSIST_DIRECTORY = f"./output/{NAME}_db"
COLLECTION_NAME = f"{NAME}_collection"
JSON_FOLDER = "./json_chunks/Results"
EXPORT_FILE = f"./output/{COLLECTION_NAME}_export.json"

# Initialize embedding function
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    multi_process = True,
    show_progress = True
    )

def load_json_files(folder_path: str) -> List[Dict[str, Any]]:
    """Load JSON files from a specified folder."""
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

def extract_components(json_data: List[Dict[str, Any]]) -> List[Document]:
    """Extract components from JSON data and create documents."""
    documents = []
    for item in json_data:
        author_likes = item.get("AuthorLikes", 0)
        components = item.get("Components", [])
        for component in components:
            doc_id = component.get("Id", "")
            component["AuthorLikes"] = author_likes
            logging.debug("Extracting component is \n\n '%s", pprint.pformat(component))
            doc_text = json.dumps(component, ensure_ascii=False)
            doc = Document(page_content=doc_text, metadata={"id": doc_id})
            documents.append(doc)
            logging.debug("Extracted component with ID '%s'", doc_id)
    return documents

def main() -> None:
    """Main function for processing JSON files and storing embeddings."""
    json_data = load_json_files(JSON_FOLDER)
    logging.info("Loaded %d JSON files from '%s'.", len(json_data), JSON_FOLDER)

    texts = extract_components(json_data)
    logging.info("Extracted %d components as documents.", len(texts))

    documents = extract_components(json_data)
    logging.info("Extracted %d components as documents.", len(documents))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1001, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    client_settings = chromadb.config.Settings(
        is_persistent=True,
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False,
    )
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=client_settings)

    logging.info("Vector database created and persisted at '%s'.", PERSIST_DIRECTORY)

if __name__ == "__main__":
    main()
