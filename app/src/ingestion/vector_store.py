"""
Simple vector store for DeFi documents using pgvector
"""
import os
import logging
from typing import List, Dict, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        # Support different embedding providers
        api_key = os.getenv('EMBEDDING_API_KEY') or os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('EMBEDDING_BASE_URL')  # For custom providers
        
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.conn_params)
    
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
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid search combining vector and keyword search"""
        query_embedding = self._generate_embedding(query)
        
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Use the hybrid search function we defined in init.sql
            cur.execute("""
                SELECT * FROM hybrid_search(%s, %s::vector, 0.7, 0.3, %s)
            """, (query, query_embedding, top_k))
            
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