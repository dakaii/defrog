"""
Smart chunking and embedding pipeline for DeFi documents
Uses semantic chunking for better context preservation
"""
import os
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import hashlib
import logging
import numpy as np
from openai import OpenAI
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata"""
    chunk_id: str
    content: str
    embedding: Optional[List[float]]
    metadata: Dict
    token_count: int


class ChunkingPipeline:
    """Intelligent chunking with overlap and semantic boundaries"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Track costs
        self.total_tokens_embedded = 0
        self.embedding_costs = []
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def smart_chunk_text(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """
        Smart chunking that respects semantic boundaries
        Tries to break at paragraph/sentence boundaries
        """
        chunks = []
        
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            # If single paragraph is too large, split by sentences
            if para_tokens > self.chunk_size:
                sentences = self._split_into_sentences(para)
                for sent in sentences:
                    sent_tokens = self.count_tokens(sent)
                    
                    if current_tokens + sent_tokens > self.chunk_size:
                        if current_chunk:
                            # Create chunk
                            chunk_text = ' '.join(current_chunk)
                            chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))
                            
                            # Keep overlap
                            overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) > 2 else ' '.join(current_chunk)
                            overlap_tokens = self.count_tokens(overlap_text)
                            
                            if overlap_tokens < self.chunk_overlap:
                                current_chunk = [overlap_text, sent]
                                current_tokens = overlap_tokens + sent_tokens
                            else:
                                current_chunk = [sent]
                                current_tokens = sent_tokens
                    else:
                        current_chunk.append(sent)
                        current_tokens += sent_tokens
            
            # Paragraph fits in current chunk
            elif current_tokens + para_tokens <= self.chunk_size:
                current_chunk.append(para)
                current_tokens += para_tokens
            
            # Need to start new chunk
            else:
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))
                    
                    # Create overlap
                    overlap_text = current_chunk[-1] if current_chunk else ""
                    overlap_tokens = self.count_tokens(overlap_text)
                    
                    if overlap_tokens < self.chunk_overlap:
                        current_chunk = [overlap_text, para]
                        current_tokens = overlap_tokens + para_tokens
                    else:
                        current_chunk = [para]
                        current_tokens = para_tokens
        
        # Don't forget last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk) if len(current_chunk) > 1 else current_chunk[0]
            chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter"""
        # Basic sentence splitting (can be improved with spacy/nltk)
        sentences = []
        current = []
        
        for char in text:
            current.append(char)
            if char in '.!?' and len(current) > 20:
                sentences.append(''.join(current).strip())
                current = []
        
        if current:
            sentences.append(''.join(current).strip())
        
        return sentences
    
    def _create_chunk(self, text: str, metadata: Dict, chunk_index: int) -> DocumentChunk:
        """Create a chunk with metadata"""
        chunk_id = hashlib.sha256(f"{text}{chunk_index}".encode()).hexdigest()[:16]
        
        chunk_metadata = {
            **metadata,
            "chunk_index": chunk_index,
            "chunk_method": "semantic",
            "chunk_size": self.chunk_size,
            "overlap": self.chunk_overlap
        }
        
        return DocumentChunk(
            chunk_id=chunk_id,
            content=text,
            embedding=None,  # Will be set by embed_chunks
            metadata=chunk_metadata,
            token_count=self.count_tokens(text)
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed_chunks(self, chunks: List[DocumentChunk], batch_size: int = 20) -> List[DocumentChunk]:
        """
        Generate embeddings for chunks in batches
        Tracks costs for optimization metrics
        """
        embedded_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk.content for chunk in batch]
            
            try:
                # Call OpenAI embedding API
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )
                
                # Track tokens and costs
                usage = response.usage
                self.total_tokens_embedded += usage.total_tokens
                
                # Cost calculation (as of 2026 pricing)
                cost = self._calculate_embedding_cost(usage.total_tokens)
                self.embedding_costs.append({
                    "batch_size": len(batch),
                    "tokens": usage.total_tokens,
                    "cost": cost
                })
                
                # Assign embeddings to chunks
                for chunk, embedding_data in zip(batch, response.data):
                    chunk.embedding = embedding_data.embedding
                    embedded_chunks.append(chunk)
                
                logger.info(f"Embedded batch {i//batch_size + 1}: {len(batch)} chunks, {usage.total_tokens} tokens, ${cost:.6f}")
                
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                raise
        
        return embedded_chunks
    
    def _calculate_embedding_cost(self, tokens: int) -> float:
        """Calculate embedding cost based on model pricing"""
        # Pricing as of 2026 (adjust as needed)
        pricing = {
            "text-embedding-3-small": 0.00002,  # per 1K tokens
            "text-embedding-3-large": 0.00013,  # per 1K tokens
            "text-embedding-ada-002": 0.00010   # per 1K tokens
        }
        
        price_per_1k = pricing.get(self.embedding_model, 0.00002)
        return (tokens / 1000) * price_per_1k
    
    def get_cost_summary(self) -> Dict:
        """Get summary of embedding costs"""
        if not self.embedding_costs:
            return {"total_cost": 0, "total_tokens": 0, "batches": 0}
        
        return {
            "total_cost": sum(c["cost"] for c in self.embedding_costs),
            "total_tokens": self.total_tokens_embedded,
            "batches": len(self.embedding_costs),
            "avg_cost_per_chunk": sum(c["cost"] for c in self.embedding_costs) / sum(c["batch_size"] for c in self.embedding_costs)
        }


class OptimizedChunkingPipeline(ChunkingPipeline):
    """
    Optimized version with caching and deduplication
    This is what makes the cost reduction claim real
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_cache = {}  # Simple in-memory cache
        self.cache_hits = 0
        self.cache_misses = 0
    
    def embed_chunks(self, chunks: List[DocumentChunk], batch_size: int = 20) -> List[DocumentChunk]:
        """Embed with caching to reduce API calls"""
        chunks_to_embed = []
        embedded_chunks = []
        
        # Check cache first
        for chunk in chunks:
            cache_key = hashlib.sha256(chunk.content.encode()).hexdigest()
            
            if cache_key in self.embedding_cache:
                # Cache hit!
                chunk.embedding = self.embedding_cache[cache_key]
                embedded_chunks.append(chunk)
                self.cache_hits += 1
                logger.debug(f"Cache hit for chunk {chunk.chunk_id}")
            else:
                # Need to embed this one
                chunks_to_embed.append((chunk, cache_key))
                self.cache_misses += 1
        
        # Embed uncached chunks
        if chunks_to_embed:
            logger.info(f"Embedding {len(chunks_to_embed)} new chunks (cache hits: {self.cache_hits})")
            
            for i in range(0, len(chunks_to_embed), batch_size):
                batch = chunks_to_embed[i:i + batch_size]
                texts = [chunk[0].content for chunk in batch]
                
                try:
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=texts
                    )
                    
                    # Track costs
                    usage = response.usage
                    self.total_tokens_embedded += usage.total_tokens
                    cost = self._calculate_embedding_cost(usage.total_tokens)
                    self.embedding_costs.append({
                        "batch_size": len(batch),
                        "tokens": usage.total_tokens,
                        "cost": cost
                    })
                    
                    # Store in cache and assign to chunks
                    for (chunk, cache_key), embedding_data in zip(batch, response.data):
                        embedding = embedding_data.embedding
                        self.embedding_cache[cache_key] = embedding
                        chunk.embedding = embedding
                        embedded_chunks.append(chunk)
                    
                except Exception as e:
                    logger.error(f"Error embedding batch: {e}")
                    raise
        
        cache_ratio = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        logger.info(f"Cache hit ratio: {cache_ratio:.2%}")
        
        return embedded_chunks
    
    def get_optimization_metrics(self) -> Dict:
        """Get metrics showing optimization effectiveness"""
        base_metrics = self.get_cost_summary()
        
        # Calculate savings from caching
        avg_cost_per_chunk = base_metrics.get("avg_cost_per_chunk", 0)
        cache_savings = self.cache_hits * avg_cost_per_chunk
        
        return {
            **base_metrics,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "estimated_savings": cache_savings,
            "effective_cost": base_metrics["total_cost"] - cache_savings
        }


if __name__ == "__main__":
    # Test the chunking pipeline
    pipeline = OptimizedChunkingPipeline()
    
    test_text = """
    Decentralized Finance (DeFi) represents a fundamental shift in how financial services operate.
    
    By leveraging blockchain technology and smart contracts, DeFi protocols eliminate traditional intermediaries and enable peer-to-peer financial transactions. This paradigm shift has profound implications for accessibility, transparency, and financial inclusion.
    
    The core innovation of DeFi lies in its composability. Protocols can interact seamlessly with each other, creating complex financial products from simple building blocks. This 'money lego' approach enables rapid innovation and experimentation.
    
    However, DeFi also faces significant challenges. Smart contract risks, scalability limitations, and regulatory uncertainty all pose threats to the ecosystem's growth. Understanding these trade-offs is crucial for anyone participating in DeFi.
    """
    
    metadata = {
        "protocol": "Test Protocol",
        "doc_type": "whitepaper",
        "source": "test"
    }
    
    chunks = pipeline.smart_chunk_text(test_text, metadata)
    print(f"Created {len(chunks)} chunks")
    
    # Test embedding (requires OPENAI_API_KEY)
    if os.getenv("OPENAI_API_KEY"):
        embedded = pipeline.embed_chunks(chunks)
        print(f"Embedded {len(embedded)} chunks")
        print(f"Optimization metrics: {pipeline.get_optimization_metrics()}")