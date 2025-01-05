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

# Define JSON schema
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


# # Create system prompt
# system_prompt = (
#     "You are a copilot for Rhino Grasshopper. Your task is to predict the next component in the design process. "
#     "\n\n"
#     "{context}"
# )

# # Create prompt template for input and context
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )
# query = "Give me the next component in the design process if my current component is a 'Curve' Return only a JSON"

query = "You are a copilot for Rhino Grasshopper. Your task is to predict the next component in the design process. Give me the next component in the design process if my current component is a 'Curve' Return only a JSON"


# Initialize Ollama Chat model
ollama = ChatOllama(model="llama3.2:latest")

# Set structured output using the updated JSON schema
structured_ollama = ollama.with_structured_output(NextComponent)

# result = structured_ollama.invoke(query)
# print(result)

# Define the directory for persistent storage
PERSIST_DIRECTORY = "./output/co-designer_db"

# Initialize the embedding function with HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Set up Chroma vector store for retrieval
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)

# Fetch more documents for the MMR algorithm to consider
# But only return the top 5
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 5, 'fetch_k': 50}
)

# Create the question-answer chain using the structured output model
question_answer_chain = create_stuff_documents_chain(structured_ollama, prompt)

# Create the retrieval chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Test the RAG pipeline with a query
result = rag_chain.invoke({"input": query})
print(result["answer"]) 
