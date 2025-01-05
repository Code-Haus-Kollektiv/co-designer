import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from typing import Optional
from pydantic import BaseModel, Field
from langgraph.graph import START, StateGraph


# Constants
PERSIST_DIRECTORY = "./output/co-designer_db"

# Query definition
query = "You are a copilot for Rhino Grasshopper. Your task is to predict the next component in the design process. Give me the next component in the design process if my current component is a 'Curve'. Return only a JSON"

# Define the NextComponent schema
class NextComponent(BaseModel):
    """
    Represents the next connected component.
    """
    name: str
    """Name of the component."""

    description: str
    """Description of the component."""

    id: str
    """GUID of the component."""

# Load prompt

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a copilot for Rhino Grasshopper."),
        ("human", "question:{question}"),
        ("human", "context:{context}"),
    ]
)

# Chat Model
ollama = ChatOllama(model="llama3.2:latest")
structured_ollama = ollama.with_structured_output(NextComponent)

# Embedding Model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector Store
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
retriever = vectordb.as_retriever()


retrieved_docs = vectordb.similarity_search(query)
docs_content = "\n\n".join(retrieved_docs)
print(docs_content)
messages = prompt_template.format_messages(question=query, context=docs_content)
result = structured_ollama.invoke(messages)

# Print results
print(result)