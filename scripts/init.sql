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

-- User feedback table (ratings for RAG responses)
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating INT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment TEXT,
    query_log_id INT REFERENCES query_logs(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Dynamic document sources (editable list of URLs to crawl)
CREATE TABLE IF NOT EXISTS document_sources (
    id SERIAL PRIMARY KEY,
    protocol_name VARCHAR(100) NOT NULL,
    doc_type VARCHAR(50) NOT NULL,  -- whitepaper, docs, litepaper
    url TEXT NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Seed default DeFi sources (skipped if already populated)
INSERT INTO document_sources (protocol_name, doc_type, url)
SELECT * FROM (VALUES
    ('Aave V2',       'whitepaper', 'https://github.com/aave/protocol-v2/blob/master/aave-v2-whitepaper.pdf'),
    ('Aave V2',       'docs',       'https://docs.aave.com'),
    ('Uniswap V2',    'whitepaper', 'https://uniswap.org/whitepaper.pdf'),
    ('Uniswap V2',    'docs',       'https://docs.uniswap.org'),
    ('Uniswap V3',    'whitepaper', 'https://uniswap.org/whitepaper-v3.pdf'),
    ('Uniswap V3',    'docs',       'https://docs.uniswap.org/protocol/V3/'),
    ('Compound',      'whitepaper', 'https://compound.finance/documents/Compound.Whitepaper.pdf'),
    ('Compound',      'docs',       'https://docs.compound.finance'),
    ('MakerDAO',      'whitepaper', 'https://makerdao.com/en/whitepaper'),
    ('MakerDAO',      'docs',       'https://docs.makerdao.com'),
    ('Curve Finance', 'whitepaper', 'https://curve.fi/files/stableswap-paper.pdf'),
    ('Curve Finance', 'docs',       'https://resources.curve.fi'),
    ('Balancer',      'whitepaper', 'https://balancer.fi/whitepaper.pdf'),
    ('Balancer',      'docs',       'https://docs.balancer.fi'),
    ('Synthetix',     'litepaper',  'https://docs.synthetix.io/litepaper'),
    ('Synthetix',     'docs',       'https://docs.synthetix.io')
) AS v(protocol_name, doc_type, url)
WHERE NOT EXISTS (SELECT 1 FROM document_sources LIMIT 1);