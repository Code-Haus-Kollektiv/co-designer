import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from uuid import UUID
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel

# Configure colorful logging
class ColorfulFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",
        "INFO": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[95m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)

# Setup logging
logger = logging.getLogger("DesignPredictor")
handler = logging.StreamHandler()
handler.setFormatter(ColorfulFormatter("[%(levelname)s] %(asctime)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Constants
PERSIST_DIRECTORY = "./WP_2_Vector_Database/output/co-designer_nonLang_db"

# GrasshopperComponent model
class GrasshopperComponent(BaseModel):
    Name: str
    Nickname: str
    Description: str

def load_prompt_template() -> ChatPromptTemplate:
    logger.info("Loading prompt template...")
    return ChatPromptTemplate.from_messages([
        ("system", "You are an expert assistant for Rhino Grasshopper, specialized in analyzing the selected component and suggesting the next component in the visual coding node graph of Grasshopper."),
        ("human", "Based on the given current component information, your task is to predict the next component in the node graph based on the current component's parameter outputs."),
        ("human", "Current Component:{context}"),
    ])

def initialize_vector_database(dir: str, embedding_function) -> Chroma:
    logger.info(f"Initializing vector database with persistence directory: {dir}")
    return Chroma(persist_directory=dir, embedding_function=embedding_function)

def retrieve_documents(vectordb: Chroma, query: str, top_k: int = 5) -> List[Any]:
    logger.info("Invoking retriever...")
    return vectordb.similarity_search(query, k=top_k)

def process_documents(retrieved_docs: List[Any]) -> Optional[Dict[str, Any]]:
    if not retrieved_docs:
        logger.warning("No documents retrieved for similarity search.")
        return None

    sorted_docs = sorted(
        retrieved_docs,
        key=lambda doc: json.loads(doc.page_content).get("AuthorLikes", 0),
        reverse=True
    )

    output_parameters = []
    for doc in sorted_docs:
        content = json.loads(doc.page_content)
        author_likes = content.get("AuthorLikes", 0)
        output_parameters.extend(
            {
                **param,
                "AuthorLikes": author_likes
            }
            for param in content.get("Parameters", [])
            if param.get("ParameterType") == "Output"
        )

    param_counts = {}
    for param in output_parameters:
        param_name = param.get("Name")
        if param_name:
            param_counts[param_name] = param_counts.get(param_name, 0) + 1

    most_common_param = max(param_counts, key=param_counts.get, default=None)
    if not most_common_param:
        logger.warning("No common parameter found.")
        return None

    logger.info(f"Most Common Output Parameter: {most_common_param}")
    common_param_subset = [
        param for param in output_parameters if param.get("Name") == most_common_param
    ]
    best_param = max(
        common_param_subset,
        key=lambda x: x.get("AuthorLikes", 0),
        default=None
    )

    if best_param:
        logger.debug(f"Best Parameter with Most AuthorLikes: {json.dumps(best_param, indent=2)}")
        return best_param

    logger.warning("No parameter found in the common subset with AuthorLikes.")
    return None

def main():
    search_item = "ba80fd98-91a1-4958-b6a7-a94e40e52bdb"

    prompt_template = load_prompt_template()
    ollama = ChatOllama(model="llama3.2:latest")
    structured_ollama = ollama.with_structured_output(GrasshopperComponent)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        show_progress=True,
    )

    vectordb = initialize_vector_database(PERSIST_DIRECTORY, embedding)

    try:
        start_time = time.time()
        retrieved_docs = retrieve_documents(vectordb, search_item)
        best_param = process_documents(retrieved_docs)

        if best_param:
            logger.info(f"Best parameter successfully identified: {best_param}")

        elapsed_time = time.time() - start_time
        logger.info(f"Total Search to Result Time: {elapsed_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error during similarity search: {e}")

if __name__ == "__main__":
    main()
