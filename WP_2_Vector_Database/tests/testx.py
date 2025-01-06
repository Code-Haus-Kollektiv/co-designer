import os
import logging
import json
from pprint import pformat
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configure colorful logging
class ColorfulFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",   # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
        "CRITICAL": "\033[95m"  # Magenta
    }

    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)

logger = logging.getLogger("VectorDBInspector")
handler = logging.StreamHandler()
formatter = ColorfulFormatter("[%(levelname)s] %(asctime)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Constants
PERSIST_DIRECTORY = "./WP_2_Vector_Database/output/co-designer5_db"

# Embedding Model
logger.info("Loading embedding model...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# Vector Store
logger.info(f"Initializing vector database with persistence directory: {PERSIST_DIRECTORY}")
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
try:
    logger.info("Fetching the first entry in the vector database...")
    entries = vectordb.similarity_search("curve")  # Fetch the first entry
    first_entry = entries[0].page_content if entries else None
    if first_entry:
        logger.info(f"First Entry:\n{pformat(json.dumps(first_entry), indent=4)}")
    else:
        logger.warning("The vector database is empty.")
except Exception as e:
    logger.error(f"Error fetching the first entry: {e}")
