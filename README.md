CVE RAG AI Agent

Overview

This project is a Retrieval-Augmented Generation (RAG) AI Agent that fetches real-time CVE data from the National Vulnerability Database (NVD) API, processes it, and retrieves high-priority unpatched vulnerabilities using FAISS and GPT-4.

Features

Real-time CVE Fetching: Retrieves the latest vulnerabilities from NVD.

Data Storage: Stores vulnerabilities in a CSV file (cve_data.csv).

RAG-based AI Agent: Uses FAISS for similarity search and GPT-4 for analysis.

Prioritization of Critical Vulnerabilities: Identifies high-risk, unpatched CVEs.

Installation

Prerequisites

Ensure you have Python 3.8+ installed. Install dependencies using:

pip install requests pandas langchain faiss-cpu openai

OpenAI API Key

This project uses OpenAI for LLM-based summarization. Set your API key:

export OPENAI_API_KEY="your_api_key_here"

For Windows:

$env:OPENAI_API_KEY="your_api_key_here"

Usage

1️⃣ Fetch Latest CVE Data

Run the following command to fetch the latest CVEs and store them in cve_data.csv:

python cve_fetcher.py

2️⃣ Retrieve High-Priority Unpatched CVEs

Run the RAG AI agent to find and analyze the most critical unpatched vulnerabilities:

python rag_agent.py

3️⃣ Deploy as an API (Optional)

Want to serve this as an API? Install FastAPI:

pip install fastapi uvicorn

Run the API:

uvicorn main:app --host 0.0.0.0 --port 8000

Then, open http://localhost:8000/docs for API documentation.

How FAISS Works in This Project

FAISS (Facebook AI Similarity Search) is a tool for fast vector-based similarity search. In this project, it is used to store and retrieve CVE descriptions efficiently.

How FAISS Works

Embedding Creation:

Each CVE description is converted into a vector embedding using OpenAI’s embedding model.

These embeddings are stored in FAISS.

Indexing:

FAISS organizes these embeddings for fast similarity search.

Querying:

When a query is provided (e.g., “Critical unpatched vulnerabilities”), FAISS retrieves the most relevant CVEs based on vector similarity.

Retrieval & Augmentation:

The retrieved CVEs are then passed to GPT-4 for summarization and analysis.

Example Usage of FAISS

# Convert CVE descriptions into embeddings
embedding_model = OpenAIEmbeddings()
text_data = df["Description"].tolist()
text_embeddings = [embedding_model.embed_query(text) for text in text_data]

# Create FAISS index
import faiss
dimension = 1536  # OpenAI embedding size
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(text_embeddings, dtype=np.float32))

# Search FAISS for closest matches
query_embedding = embedding_model.embed_query("Critical unpatched vulnerabilities")
_, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), k=5)
retrieved_cves = [text_data[idx] for idx in indices[0]]

print("Top 5 Matching CVEs:", retrieved_cves)

Project Structure

CVE-RAG-AI/
│── cve_fetcher.py      # Fetches CVE data from NVD API
│── rag_agent.py        # RAG AI Agent for CVE Retrieval
│── cve_data.csv        # Stored CVE data
│── main.py (optional)  # FastAPI Server (if deployed)
│── README.md           # Project Documentation

Future Enhancements

✅ Automated CVE updates with cron jobs.
✅ Enhanced filtering by vendor or product.
✅ Real-time alerts for new vulnerabilities.
✅ Integration with security platforms (Splunk, SIEMs).

Contributions

Feel free to contribute by submitting PRs or reporting issues!

License

This project is licensed under the MIT License.
