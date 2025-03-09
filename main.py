import cve_fetcher
import rag_agent
import analysis_agent
import os

if __name__ == "__main__":
    print("\n🚀 Fetching CVE Data...")
    cve_fetcher.save_cve_to_csv()

    print("\n🔍 Running RAG AI Agent...")
    print(rag_agent.rag_retrieve("Show me high-priority CVEs"))

    print("\n📊 Running Analysis AI Agent...")
    print(analysis_agent.analyze_cves())
