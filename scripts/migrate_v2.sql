-- Migration v2: add feedback and document_sources tables
-- Safe to run multiple times (idempotent)

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating INT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment TEXT,
    query_log_id INT REFERENCES query_logs(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_sources (
    id SERIAL PRIMARY KEY,
    protocol_name VARCHAR(100) NOT NULL,
    doc_type VARCHAR(50) NOT NULL,
    url TEXT NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

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
