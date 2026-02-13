"""
DeFrog Dashboard - Simple DeFi RAG Query Interface
"""
import streamlit as st
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Streamlit
st.set_page_config(
    page_title="DeFrog - DeFi RAG",
    page_icon="🐸",
    layout="wide"
)

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")


def fetch_api(endpoint: str, method="GET", json=None):
    """Fetch data from API"""
    try:
        if method == "GET":
            response = requests.get(f"{API_URL}{endpoint}", timeout=30)
        else:
            response = requests.post(f"{API_URL}{endpoint}", json=json, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# Title
st.title("🐸 DeFrog - DeFi Protocol RAG")
st.markdown("Query DeFi protocol whitepapers and documentation using RAG")

# Main content
tab1, tab2 = st.tabs(["🔍 Query", "ℹ️ System Info"])

with tab1:
    st.header("Ask about DeFi Protocols")
    
    # Example queries
    with st.expander("Example Questions"):
        st.markdown("""
        - How does Uniswap V3 concentrated liquidity work?
        - What is the difference between Uniswap V2 and V3?
        - How does Aave's liquidation mechanism work?
        - Explain Compound's interest rate model
        - What are the risks of providing liquidity in Curve?
        - How does MakerDAO maintain DAI's peg?
        """)
    
    # Query input
    query = st.text_area(
        "Enter your DeFi question:",
        placeholder="e.g., How does Uniswap liquidity provision work?",
        height=100
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        top_k = st.slider("Number of documents to retrieve", 3, 10, 5)
    
    # Query button
    if st.button("🚀 Ask Question", type="primary"):
        if query:
            with st.spinner("Searching documentation and generating answer..."):
                # Call API
                result = fetch_api("/query", method="POST", json={
                    "query": query,
                    "top_k": top_k
                })
                
                if result:
                    # Display answer
                    st.success("Answer generated!")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model", result.get("model_used", "Unknown"))
                    with col2:
                        st.metric("Latency", f"{result.get('latency_ms', 0)}ms")
                    with col3:
                        st.metric("Documents Retrieved", result.get("documents_retrieved", 0))
                    
                    # Answer
                    st.subheader("Answer")
                    st.markdown(result.get("answer", "No answer generated"))
                    
                    # Sources
                    if result.get("sources"):
                        st.subheader("Sources")
                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(f"Source {i}: {source.get('protocol', 'Unknown')} - {source.get('doc_type', 'document')}"):
                                st.write(f"**Title:** {source.get('title', 'Unknown')}")
                                st.write(f"**Similarity:** {source.get('similarity', 0):.3f}")
                                if source.get('url'):
                                    st.write(f"**URL:** {source['url']}")
        else:
            st.warning("Please enter a question")

with tab2:
    st.header("System Information")
    
    # Health check
    health = fetch_api("/health")
    if health:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_icon = "✅" if health["status"] == "healthy" else "⚠️"
            st.metric("System Status", f"{status_icon} {health['status'].title()}")
        
        with col2:
            pg_icon = "✅" if health["postgres"] else "❌"
            st.metric("PostgreSQL", pg_icon)
        
        with col3:
            openai_icon = "✅" if health["openai"] else "❌"
            st.metric("OpenAI API", openai_icon)
    
    st.divider()
    
    # Available protocols
    protocols = fetch_api("/protocols")
    if protocols:
        st.subheader("Available Protocols")
        cols = st.columns(3)
        for i, protocol in enumerate(protocols.get("protocols", [])):
            cols[i % 3].write(f"• {protocol}")
    
    st.divider()
    
    # Instructions
    st.subheader("How to Use")
    st.markdown("""
    1. **Ask a Question**: Enter any question about DeFi protocols in the query box
    2. **Get Answer**: The system searches through protocol documentation and generates an answer
    3. **Check Sources**: Review which protocols and documents were used to answer your question
    
    The system has ingested whitepapers and documentation from major DeFi protocols including:
    - Uniswap (V2 & V3)
    - Aave
    - Compound
    - MakerDAO
    - Curve Finance
    - And more...
    """)

# Sidebar
with st.sidebar:
    st.header("🐸 About DeFrog")
    st.markdown("""
    **DeFrog** is a RAG (Retrieval Augmented Generation) system for querying DeFi protocol documentation.
    
    **Features:**
    - Vector search through DeFi whitepapers
    - LLM-powered answer generation
    - Source attribution
    - Simple, clean interface
    
    **Tech Stack:**
    - PostgreSQL with pgvector
    - OpenAI embeddings & GPT
    - FastAPI backend
    - Streamlit frontend
    """)
    
    st.divider()
    
    # Refresh button
    if st.button("🔄 Refresh"):
        st.rerun()