import os
import json
import logging
import time
from uuid import UUID
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from typing import List, Any, Optional
from pydantic import BaseModel, Field
from pprint import pformat
from langgraph.graph import START, StateGraph
import jsonpickle

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

logger = logging.getLogger("DesignPredictor")
handler = logging.StreamHandler()
formatter = ColorfulFormatter("[%(levelname)s] %(asctime)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Constants
PERSIST_DIRECTORY = "./WP_2_Vector_Database/output/co-designer5_db"

# Query definition
search_item = """
    "Name": "Vector 2Pt",
    "Nickname": "Vec2Pt",
    """

# Define GrasshopperComponent model
class GrasshopperComponent(BaseModel):
    """
    Represents a Grasshopper component, its properties, parameters, and plugin association.
    """
    Name: str
    Nickname: str
    Description: str

# Load prompt
logger.info("Loading prompt template...")
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert assistant for Rhino Grasshopper, specialized in analyzing the selected component and suggesting the next component in the visual coding node graph of Grasshopper."),
        ("human", "Based on the given current component information, Your task is to predict the next component in the node graph based on the current component's parameter outputs."),
        ("human", "Current Component:{context}"),
    ]
)

# Chat Model
logger.info("Initializing chat model...")
ollama = ChatOllama(model="llama3.2:latest")
structured_ollama = ollama.with_structured_output(GrasshopperComponent)

# Embedding Model
logger.info("Loading embedding model...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # multi_process = True,
    show_progress = True,
)

# Vector Store
logger.info(f"Initializing vector database with persistence directory: {PERSIST_DIRECTORY}")
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
logger.info("Vector Database Initialized")

try:
    start_time = time.time()
    logger.info("Invoking retriever...")
    retrieved_docs = vectordb.similarity_search(search_item, k=5)  # Retrieve the closest document, sorted by author_likes
    logger.info(f"Retrieved {len(retrieved_docs)} documents.")
    if len(retrieved_docs) > 0:
        # Sort documents by AuthorLikes
        
        sorted_docs = sorted(retrieved_docs, key=lambda doc: json.loads(doc.page_content).get("AuthorLikes", 0), reverse=True)
        logger.debug("Sorted Docs")
        # Extract output parameters
        output_parameters = []
        logger.debug("Getting Output Parameters")
        for doc in sorted_docs:
            content = json.loads(doc.page_content)
            author_likes = content.get("AuthorLikes", 0)
            output_params = [
                {
                    **param,
                    "AuthorLikes": author_likes
                }
                for param in content.get("Parameters", [])
                if param["ParameterType"] == "Output"
            ]
            output_parameters.extend(output_params)

        logger.debug("Getting Most Common Parameter Subset")
        # Find the most common output parameter
        param_counts = {}
        for param in output_parameters:
            param_name = param.get("Name")
            if param_name:
                param_counts[param_name] = param_counts.get(param_name, 0) + 1

        most_common_param = max(param_counts, key=param_counts.get, default=None)

        if most_common_param:
            logger.info(f"Most Common Output Parameter: {most_common_param}")
            # Filter the subset of parameters matching the most common parameter
            common_param_subset = [param for param in output_parameters if param.get("Name") == most_common_param]
            # Select the parameter with the most AuthorLikes
            best_param = max(common_param_subset, key=lambda x: x.get("AuthorLikes", 0), default=None)

            if best_param:
                logger.info(f"Best Parameter with Most AuthorLikes: {json.dumps(best_param, indent=2)}")
            else:
                logger.warning("No parameter found in the common subset with AuthorLikes.")
        else:
            logger.warning("No common parameter found.")
    else:
        logger.warning("No documents retrieved for similarity search.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total Search to Result Time: {elapsed_time:.2f} seconds")
except Exception as e:
    logger.error(f"Error during similarity search: {e}")
