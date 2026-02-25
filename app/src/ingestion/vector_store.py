"""
Simple vector store for DeFi documents using pgvector
"""
import os
import logging
from typing import List, Dict, Optional
import psycopg2
import psycopg2.pool
import psycopg2.extensions
from psycopg2.extras import RealDictCursor
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _PooledConnection:
    """
    Proxy around a psycopg2 connection checked out from a ThreadedConnectionPool.
    Intercepts close() to return the connection to the pool rather than destroying
    the underlying TCP socket, making it a transparent drop-in for callers that
    already call conn.close() in their finally blocks.
    """

    def __init__(self, conn, pool: psycopg2.pool.ThreadedConnectionPool):
        # Use object.__setattr__ to avoid triggering __getattr__ during init
        object.__setattr__(self, '_conn', conn)
        object.__setattr__(self, '_pool', pool)

    # Delegate every attribute/method lookup to the real connection
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_conn'), name)

    # Explicit delegation for the most-used methods (avoids __getattr__ overhead)
    def cursor(self, *args, **kwargs):
        return object.__getattribute__(self, '_conn').cursor(*args, **kwargs)

    def commit(self):
        return object.__getattribute__(self, '_conn').commit()

    def rollback(self):
        return object.__getattribute__(self, '_conn').rollback()

    def close(self):
        """Return the connection to the pool instead of closing it."""
        conn = object.__getattribute__(self, '_conn')
        pool = object.__getattribute__(self, '_pool')
        if not conn.closed:
            # Roll back any open transaction before returning to pool
            if conn.status != psycopg2.extensions.STATUS_READY:
                try:
                    conn.rollback()
                except Exception:
                    pass
        pool.putconn(conn)


class VectorStore:
    """Simple vector store using PostgreSQL with pgvector"""
    
    def __init__(self):
        self.conn_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'defrog'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
        }

        # Connection pool: keep 2 connections warm, allow up to 10 under load.
        # Sizes are configurable via env so production can tune without code changes.
        min_conn = int(os.getenv('DB_POOL_MIN', '2'))
        max_conn = int(os.getenv('DB_POOL_MAX', '10'))
        self._pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=min_conn,
            maxconn=max_conn,
            **self.conn_params
        )
        logger.info(f"DB connection pool initialised (min={min_conn}, max={max_conn})")

        # Support different embedding providers
        api_key = os.getenv('EMBEDDING_API_KEY') or os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('EMBEDDING_BASE_URL')  # For custom providers

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

    def get_connection(self) -> '_PooledConnection':
        """Check out a connection from the pool.

        Callers should still call conn.close() when done — the _PooledConnection
        proxy intercepts that call and returns the connection to the pool instead
        of destroying it.
        """
        return _PooledConnection(self._pool.getconn(), self._pool)

    def close(self):
        """Close the connection pool (call on app shutdown)."""
        self._pool.closeall()
        logger.info("DB connection pool closed")
    
    def store_documents(self, documents: List[Dict]):
        """Store documents with embeddings in the database"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            for doc in documents:
                # Generate embedding
                embedding = self._generate_embedding(doc['content'])
                
                # Store in database
                cur.execute("""
                    INSERT INTO documents (content, metadata, embedding)
                    VALUES (%s, %s, %s)
                """, (
                    doc['content'],
                    doc.get('metadata', {}),
                    embedding
                ))
                
                logger.info(f"Stored document: {doc.get('metadata', {}).get('title', 'Unknown')}")
            
            conn.commit()
            logger.info(f"Successfully stored {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents using vector similarity"""
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Vector similarity search
            cur.execute("""
                SELECT 
                    id,
                    content,
                    metadata,
                    1 - (embedding <=> %s::vector) as similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))
            
            results = cur.fetchall()
            return results
            
        finally:
            cur.close()
            conn.close()
    
    def hybrid_search(self, query: str, top_k: int = 5,
                      vector_weight: float = 0.7, keyword_weight: float = 0.3) -> List[Dict]:
        """Hybrid search combining vector and keyword search"""
        query_embedding = self._generate_embedding(query)

        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT * FROM hybrid_search(%s, %s::vector, %s, %s, %s)
            """, (query, query_embedding, vector_weight, keyword_weight, top_k))

            results = cur.fetchall()
            return results

        finally:
            cur.close()
            conn.close()

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Pure keyword/full-text search using PostgreSQL ts_rank"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT
                    id,
                    content,
                    metadata,
                    ts_rank(
                        to_tsvector('english', content),
                        plainto_tsquery('english', %s)
                    ) as score
                FROM documents
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
            """, (query, query, top_k))

            results = cur.fetchall()
            return results

        finally:
            cur.close()
            conn.close()
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def clear_all(self):
        """Clear all documents from the database"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("TRUNCATE TABLE documents CASCADE")
            conn.commit()
            logger.info("Cleared all documents from database")
        finally:
            cur.close()
            conn.close()
    
    def get_document_count(self) -> int:
        """Get total number of documents"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("SELECT COUNT(*) FROM documents")
            return cur.fetchone()[0]
        finally:
            cur.close()
            conn.close()