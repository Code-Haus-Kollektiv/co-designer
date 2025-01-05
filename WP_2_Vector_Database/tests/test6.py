import os
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from chromadb.utils import embedding_functions

# Configuration Section
CHROMADB_DIR = r"WP_2_Vector_Database\output\co-designer_db"  # Update this to your project's base directory
COLLECTION_NAME = "co-designer_collection"  # Update with your collection name
OLLAMA_HOST = "http://localhost:11434/"

# Define the embedding function using SentenceTransformers
DEFAULT_EF = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Configure the language model
config_list = [
    {
        "model": "llama3.2:latest",
        "api_type": "ollama",
        "client_host": OLLAMA_HOST
    }
]

# Initialize the Assistant Agent
assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={"config_list": config_list},
)

# Initialize the Retrieve User Proxy Agent
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "qa",
        "vector_db": "chroma",
        "db_config": {
            "persist_directory": CHROMADB_DIR
        },
        "embedding_function": DEFAULT_EF,
        "get_or_create": True,
        "collection_name": COLLECTION_NAME,
    },
    code_execution_config=False,
)

# Define the query or problem statement
qa_problem = "Your question here"

# Initiate the chat
ragproxyagent.initiate_chat(
    assistant,
    problem=qa_problem
)
