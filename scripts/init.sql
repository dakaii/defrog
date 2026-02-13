-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create main documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create full-text search index
CREATE INDEX idx_documents_content_gin ON documents USING gin(to_tsvector('english', content));

-- Create vector index (HNSW for better performance)
CREATE INDEX idx_documents_embedding ON documents USING hnsw (embedding vector_cosine_ops);

-- Create metadata index for filtering
CREATE INDEX idx_documents_metadata ON documents USING gin(metadata);

-- Hybrid search function combining vector and keyword search
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector(1536),
    vector_weight FLOAT DEFAULT 0.7,
    keyword_weight FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 10
)
RETURNS TABLE(
    id INT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT 
            d.id,
            d.content,
            d.metadata,
            (1 - (d.embedding <=> query_embedding)) AS vector_similarity
        FROM documents d
        ORDER BY d.embedding <=> query_embedding
        LIMIT match_count * 2
    ),
    keyword_results AS (
        SELECT 
            d.id,
            d.content,
            d.metadata,
            ts_rank(to_tsvector('english', d.content), plainto_tsquery('english', query_text)) AS keyword_similarity
        FROM documents d
        WHERE to_tsvector('english', d.content) @@ plainto_tsquery('english', query_text)
        LIMIT match_count * 2
    ),
    combined AS (
        SELECT 
            COALESCE(v.id, k.id) AS id,
            COALESCE(v.content, k.content) AS content,
            COALESCE(v.metadata, k.metadata) AS metadata,
            (COALESCE(v.vector_similarity, 0) * vector_weight + 
             COALESCE(k.keyword_similarity, 0) * keyword_weight) AS similarity
        FROM vector_results v
        FULL OUTER JOIN keyword_results k ON v.id = k.id
    )
    SELECT * FROM combined
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Cost tracking table
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    query_text TEXT,
    model_used VARCHAR(50),
    input_tokens INT,
    output_tokens INT,
    cost DECIMAL(10, 6),
    latency_ms INT,
    cache_hit BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Evaluation results table
CREATE TABLE IF NOT EXISTS evaluation_results (
    id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(100),
    query_text TEXT,
    ground_truth TEXT,
    predicted_answer TEXT,
    faithfulness FLOAT,
    answer_relevancy FLOAT,
    context_recall FLOAT,
    cost DECIMAL(10, 6),
    created_at TIMESTAMP DEFAULT NOW()
);