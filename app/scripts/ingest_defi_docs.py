"""
Simple ingestion script for DeFi whitepapers
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.crawler import DeFiCrawler
from src.ingestion.chunking import ChunkingPipeline
from src.ingestion.vector_store import VectorStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main ingestion pipeline"""
    # Parse args early so --clear is available before we need it
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Clear existing documents before ingesting")
    args, _ = parser.parse_known_args()
    
    logger.info("Starting DeFi document ingestion...")
    
    # Initialize components
    crawler = DeFiCrawler()
    chunker = ChunkingPipeline(chunk_size=800, chunk_overlap=200)
    vector_store = VectorStore()
    
    # Step 1: Crawl documents
    logger.info("Step 1: Crawling DeFi documents...")
    documents = crawler.crawl_all()
    logger.info(f"Crawled {len(documents)} documents")
    
    if not documents:
        logger.error("No documents crawled. Exiting.")
        return
    
    # Step 2: Chunk documents
    logger.info("Step 2: Chunking documents...")
    all_chunks = []
    
    for doc in documents:
        logger.info(f"Chunking {doc.title}...")
        
        # Create metadata for chunks
        metadata = {
            "protocol": doc.protocol,
            "title": doc.title,
            "url": doc.url,
            "doc_type": doc.doc_type
        }
        
        # Chunk the document
        chunks = chunker.smart_chunk_text(doc.content, metadata)
        
        # Convert to format for vector store
        for chunk in chunks:
            all_chunks.append({
                "content": chunk.content,
                "metadata": chunk.metadata
            })
        
        logger.info(f"Created {len(chunks)} chunks from {doc.title}")
    
    logger.info(f"Total chunks created: {len(all_chunks)}")
    
    # Step 3: Generate embeddings and store
    logger.info("Step 3: Generating embeddings and storing in vector database...")
    
    # Clear existing documents (use --clear flag for non-interactive/Docker)
    if args.clear:
        vector_store.clear_all()
        logger.info("Cleared existing documents")
    
    # Process in batches to avoid overwhelming the API
    batch_size = 10
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
        
        try:
            vector_store.store_documents(batch)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            continue
    
    # Step 4: Verify
    doc_count = vector_store.get_document_count()
    logger.info(f"✅ Ingestion complete! Total documents in database: {doc_count}")
    
    # Step 5: Test search
    test_query = "How does Uniswap liquidity provision work?"
    logger.info(f"\nTest search: '{test_query}'")
    
    results = vector_store.hybrid_search(test_query, top_k=3)
    for i, result in enumerate(results):
        logger.info(f"\nResult {i+1} (similarity: {result.get('similarity', 0):.3f}):")
        logger.info(f"Protocol: {result['metadata'].get('protocol', 'Unknown')}")
        logger.info(f"Content preview: {result['content'][:200]}...")


if __name__ == "__main__":
    main()