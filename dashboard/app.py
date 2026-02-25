"""
DeFrog Dashboard - Simple DeFi RAG Query Interface
"""
import json
import os

import requests
import streamlit as st
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
API_KEY = os.getenv("API_KEY", "").strip()

_AUTH_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


def fetch_api(endpoint: str, method="GET", json_body=None):
    """Fetch data from API (non-streaming)"""
    try:
        if method == "GET":
            response = requests.get(f"{API_URL}{endpoint}", headers=_AUTH_HEADERS, timeout=30)
        else:
            response = requests.post(
                f"{API_URL}{endpoint}", json=json_body, headers=_AUTH_HEADERS, timeout=30
            )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def stream_query(query: str, top_k: int):
    """
    Generator that yields text chunks from the /query/stream SSE endpoint.
    Stores final metadata (sources, model, latency) in st.session_state["stream_meta"].
    """
    try:
        with requests.post(
            f"{API_URL}/query/stream",
            json={"query": query, "top_k": top_k},
            headers={**_AUTH_HEADERS, "Accept": "text/event-stream"},
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                if raw_line.startswith(b"data: "):
                    try:
                        event = json.loads(raw_line[6:])
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "chunk":
                        yield event["content"]
                    elif event.get("type") == "done":
                        st.session_state["stream_meta"] = event
                    elif event.get("type") == "error":
                        st.error(f"Stream error: {event.get('message')}")
    except Exception as e:
        st.error(f"Streaming error: {e}")


# Title
st.title("🐸 DeFrog - DeFi Protocol RAG")
st.markdown("Query DeFi protocol whitepapers and documentation using RAG")

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Query", "💬 Feedback", "ℹ️ System Info"])

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
        use_streaming = st.checkbox("Stream answer token-by-token", value=True)

    # Query button
    if st.button("🚀 Ask Question", type="primary"):
        if not query:
            st.warning("Please enter a question")
        elif use_streaming:
            st.session_state.pop("stream_meta", None)
            st.subheader("Answer")
            # st.write_stream consumes the generator and renders tokens live
            answer = st.write_stream(stream_query(query, top_k))

            meta = st.session_state.get("stream_meta", {})
            if meta:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model", meta.get("model", "Unknown"))
                with col2:
                    st.metric("Documents Retrieved", meta.get("documents_retrieved", 0))
                with col3:
                    cache_label = "Yes" if meta.get("cache_hit") else "No"
                    st.metric("Cache Hit", cache_label)

                if meta.get("sources"):
                    st.subheader("Sources")
                    for i, source in enumerate(meta["sources"], 1):
                        with st.expander(
                            f"Source {i}: {source.get('protocol', 'Unknown')} — {source.get('doc_type', 'document')}"
                        ):
                            st.write(f"**Title:** {source.get('title', 'Unknown')}")
                            st.write(f"**Similarity:** {source.get('similarity', 0):.3f}")
                            if source.get("url"):
                                st.write(f"**URL:** {source['url']}")

            # Persist for feedback tab
            if answer:
                st.session_state["last_query"] = query
                st.session_state["last_answer"] = answer
        else:
            with st.spinner("Searching documentation and generating answer..."):
                result = fetch_api("/query", method="POST", json_body={"query": query, "top_k": top_k})

            if result:
                st.success("Answer generated!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model", result.get("model_used", "Unknown"))
                with col2:
                    st.metric("Latency", f"{result.get('latency_ms', 0)}ms")
                with col3:
                    st.metric("Documents Retrieved", result.get("documents_retrieved", 0))

                st.subheader("Answer")
                st.markdown(result.get("answer", "No answer generated"))

                if result.get("sources"):
                    st.subheader("Sources")
                    for i, source in enumerate(result["sources"], 1):
                        with st.expander(
                            f"Source {i}: {source.get('protocol', 'Unknown')} — {source.get('doc_type', 'document')}"
                        ):
                            st.write(f"**Title:** {source.get('title', 'Unknown')}")
                            st.write(f"**Similarity:** {source.get('similarity', 0):.3f}")
                            if source.get("url"):
                                st.write(f"**URL:** {source['url']}")

                st.session_state["last_query"] = query
                st.session_state["last_answer"] = result.get("answer", "")


with tab2:
    st.header("Rate the Last Answer")

    last_query = st.session_state.get("last_query", "")
    last_answer = st.session_state.get("last_answer", "")

    if not last_query:
        st.info("Ask a question in the Query tab first, then come back here to rate the answer.")
    else:
        st.markdown(f"**Question:** {last_query}")
        with st.expander("Answer"):
            st.markdown(last_answer)

        rating = st.slider("Rating (1 = poor, 5 = excellent)", 1, 5, 3)
        comment = st.text_area("Optional comment", height=80)

        if st.button("Submit Feedback"):
            result = fetch_api(
                "/feedback",
                method="POST",
                json_body={
                    "query": last_query,
                    "answer": last_answer,
                    "rating": rating,
                    "comment": comment or None,
                },
            )
            if result and result.get("status") == "accepted":
                st.success(f"Feedback submitted (id={result['id']}). Thank you!")
            else:
                st.error("Failed to submit feedback.")


with tab3:
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

    # Document sources
    sources_data = fetch_api("/sources")
    if sources_data:
        st.subheader("Document Sources")
        sources = sources_data.get("sources", [])
        if sources:
            for src in sources:
                enabled_badge = "🟢" if src.get("enabled") else "🔴"
                st.write(
                    f"{enabled_badge} **{src['protocol_name']}** — {src['doc_type']} — {src['url']}"
                )
        else:
            st.info("No sources configured yet.")

    st.divider()

    # Instructions
    st.subheader("How to Use")
    st.markdown("""
    1. **Ask a Question**: Enter any question about DeFi protocols in the query box
    2. **Stream or batch**: Toggle streaming to see tokens appear in real time
    3. **Rate Answers**: Use the Feedback tab to rate answer quality
    4. **Check Sources**: Review which protocols and documents were used

    The system has ingested whitepapers and documentation from major DeFi protocols including:
    - Uniswap (V2 & V3), Aave, Compound, MakerDAO, Curve Finance, and more
    """)

# Sidebar
with st.sidebar:
    st.header("🐸 About DeFrog")
    st.markdown("""
    **DeFrog** is a RAG (Retrieval Augmented Generation) system for querying DeFi protocol documentation.

    **Features:**
    - Hybrid vector + keyword search
    - Cross-encoder reranking
    - Streaming answers (SSE)
    - Query caching
    - Answer feedback
    - LLM-agnostic (OpenAI, DeepSeek, Qwen…)

    **Tech Stack:**
    - PostgreSQL with pgvector
    - FastAPI backend
    - Streamlit frontend
    """)

    st.divider()

    if st.button("🔄 Refresh"):
        st.rerun()
