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
PERSIST_DIRECTORY = "./output/co-designer5_db"


# Vector Store
logger.info(f"Initializing vector database with persistence directory: {PERSIST_DIRECTORY}")
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, )
logger.debug(vectordb.get(ids="e31b1b18-8aab-4e71-a168-f5238e4525f9"))
