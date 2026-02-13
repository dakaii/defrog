"""
Simple RAG engine for DeFi queries
"""
import os
import logging
from typing import List, Dict
from openai import OpenAI
from src.ingestion.vector_store import VectorStore
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """Simple RAG implementation for DeFi queries"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Process a query using RAG
        1. Retrieve relevant documents
        2. Build context
        3. Generate answer using LLM
        """
        
        # Step 1: Retrieve relevant documents
        logger.info(f"Retrieving documents for: {question}")
        documents = self.vector_store.hybrid_search(question, top_k=top_k)
        
        if not documents:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "model": self.model,
                "documents_retrieved": 0
            }
        
        # Step 2: Build context from retrieved documents
        context = self._build_context(documents)
        sources = self._extract_sources(documents)
        
        # Step 3: Generate answer using LLM
        logger.info("Generating answer with LLM...")
        answer = self._generate_answer(question, context)
        
        return {
            "answer": answer,
            "sources": sources,
            "model": self.model,
            "documents_retrieved": len(documents)
        }
    
    def _build_context(self, documents: List[Dict]) -> str:
        """Build context from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            protocol = metadata.get('protocol', 'Unknown')
            doc_type = metadata.get('doc_type', 'document')
            
            context_parts.append(f"[Source {i} - {protocol} {doc_type}]")
            context_parts.append(doc['content'])
            context_parts.append("")  # Empty line between sources
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, documents: List[Dict]) -> List[Dict]:
        """Extract source information from documents"""
        sources = []
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            sources.append({
                "protocol": metadata.get('protocol', 'Unknown'),
                "title": metadata.get('title', 'Unknown'),
                "doc_type": metadata.get('doc_type', 'document'),
                "url": metadata.get('url', ''),
                "similarity": doc.get('similarity', 0)
            })
        
        return sources
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM with context"""
        
        system_prompt = """You are a DeFi expert assistant. Answer questions about DeFi protocols 
        based on the provided context from official whitepapers and documentation. 
        Be accurate, concise, and cite which protocol's documentation you're referencing when relevant.
        If the context doesn't contain enough information to fully answer the question, say so."""
        
        user_prompt = f"""Context from DeFi documentation:
{context}

Question: {question}

Please provide a clear and accurate answer based on the context above."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"