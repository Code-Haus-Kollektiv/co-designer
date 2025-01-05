import os
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from chromadb.utils import embedding_functions

# Configuration Section
BASE_DIR = os.path.abspath("WP_2_Vector_Database/output/co-designer_db")  # Update this to your project's base directory
COLLECTION_NAME = "co-designer_collection"  # Update with your collection name
OLLAMA_HOST = "http://localhost:11434/"

# Configure the language model
llm_config = {
    "config_list": [
        {
            "model": "llama3.2:latest",
            "api_type": "ollama",
            "client_host": OLLAMA_HOST
        }
    ]
}

# Initialize the Assistant Agent
assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)

# Define the embedding function using SentenceTransformers
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Initialize the RetrieveUserProxyAgent with ChromaDB
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    retrieve_config={
        "task": "qa",
        "vector_db": "chroma",
        "db_config": {
            "persist_directory": BASE_DIR
        },
        "embedding_function": embedding_function,
        "collection_name": COLLECTION_NAME,
        "get_or_create": True,  # Set to False if you don't want to reuse an existing collection
    },
)

# Reset the Assistant Agent
assistant.reset()

# Initialize the UserProxyAgent and start the interaction
userproxyagent = UserProxyAgent(name="userproxyagent")
userproxyagent.initiate_chat(assistant, message="What is autogen?")
