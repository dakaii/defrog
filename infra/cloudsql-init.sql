-- CloudSQL initialization script for DeFrog
-- This script sets up the database with pgvector extension and necessary tables

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create main documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_content_gin 
    ON documents USING gin(to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS idx_documents_embedding 
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_documents_metadata 
    ON documents USING gin(metadata);

CREATE INDEX IF NOT EXISTS idx_documents_created_at 
    ON documents(created_at DESC);

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
        WHERE d.embedding IS NOT NULL
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
    WHERE similarity > 0
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Query logs table for cost tracking
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    query_text TEXT NOT NULL,
    model_used VARCHAR(50),
    input_tokens INT,
    output_tokens INT,
    cost DECIMAL(10, 6),
    latency_ms INT,
    cache_hit BOOLEAN DEFAULT FALSE,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_created_at 
    ON query_logs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_query_logs_user_id 
    ON query_logs(user_id);

CREATE INDEX IF NOT EXISTS idx_query_logs_session_id 
    ON query_logs(session_id);

-- Evaluation results table
CREATE TABLE IF NOT EXISTS evaluation_results (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    experiment_name VARCHAR(100) NOT NULL,
    query_text TEXT NOT NULL,
    ground_truth TEXT,
    predicted_answer TEXT,
    faithfulness FLOAT,
    answer_relevancy FLOAT,
    context_recall FLOAT,
    cost DECIMAL(10, 6),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evaluation_results_experiment 
    ON evaluation_results(experiment_name);

CREATE INDEX IF NOT EXISTS idx_evaluation_results_created_at 
    ON evaluation_results(created_at DESC);

-- Ingestion tracking table
CREATE TABLE IF NOT EXISTS ingestion_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    protocol VARCHAR(100) NOT NULL,
    document_title TEXT NOT NULL,
    document_url TEXT,
    doc_type VARCHAR(50),
    chunks_created INT,
    status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ingestion_logs_protocol 
    ON ingestion_logs(protocol);

CREATE INDEX IF NOT EXISTS idx_ingestion_logs_status 
    ON ingestion_logs(status);

-- Create update trigger for updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create materialized view for protocol statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS protocol_stats AS
SELECT 
    metadata->>'protocol' AS protocol,
    COUNT(*) AS document_count,
    AVG(LENGTH(content)) AS avg_content_length,
    MAX(created_at) AS last_updated
FROM documents
WHERE metadata->>'protocol' IS NOT NULL
GROUP BY metadata->>'protocol';

CREATE UNIQUE INDEX IF NOT EXISTS idx_protocol_stats_protocol 
    ON protocol_stats(protocol);

-- Function to refresh protocol stats
CREATE OR REPLACE FUNCTION refresh_protocol_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY protocol_stats;
END;
$$ LANGUAGE plpgsql;

-- Grant necessary permissions (adjust based on your user setup)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO defrog;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO defrog;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO defrog;