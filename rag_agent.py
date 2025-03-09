import pandas as pd
import faiss
import numpy as np
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

os.environ[
    "OPENAI_API_KEY"] = "your_api_key_here"

# Ensure OpenAI API Key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ OpenAI API Key is missing. Set it as an environment variable!")

# Load CVE Data from CSV
csv_file = "cve_data.csv"

if not os.path.exists(csv_file):
    raise FileNotFoundError(f"❌ CSV file '{csv_file}' not found. Please fetch CVE data first!")

df = pd.read_csv(csv_file)

if df.empty:
    raise ValueError("❌ CSV file is empty. Ensure CVE data is available!")

# Initialize OpenAI Embedding Model
embedding_model = OpenAIEmbeddings()

# Extract Text Data for Embeddings
text_data = df["Description"].dropna().tolist()

if not text_data:
    raise ValueError("❌ No valid descriptions found in CSV.")

# Create FAISS Vector Store using LangChain
try:
    vector_store = FAISS.from_texts(text_data, embedding_model)
except Exception as e:
    raise RuntimeError(f"❌ Error initializing FAISS: {e}")


# RAG-based Retrieval Function
def rag_retrieve(query: str) -> str:
    """Retrieves the most relevant CVEs using FAISS and summarizes using GPT-4."""

    if not query:
        return "❌ Query cannot be empty."

    try:
        # Retrieve top 5 matches using LangChain FAISS Retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

        # Run query
        response = qa_chain.run(query)

        return response

    except Exception as e:
        return f"❌ Error in RAG retrieval: {e}"


# Run the AI Agent
if __name__ == "__main__":
    query = "Show me critical unpatched vulnerabilities"
    print("🔍 RAG AI Agent Response:")
    print(rag_retrieve(query))
