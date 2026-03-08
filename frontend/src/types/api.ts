export interface QueryRequest {
  query: string
  top_k: number
}

export interface DocumentSource {
  protocol: string
  doc_type: string
  title: string
  similarity: number
  url?: string
  content?: string
}

export interface QueryResponse {
  answer: string
  model_used: string
  latency_ms: number
  documents_retrieved: number
  cache_hit?: boolean
  sources: DocumentSource[]
}

export interface FeedbackRequest {
  query: string
  answer: string
  rating: number
  comment?: string
}

export interface FeedbackResponse {
  id: number
  status: 'accepted' | 'rejected'
}

export interface HealthResponse {
  status: 'healthy' | 'degraded'
  postgres: boolean
  openai: boolean
}

export interface ProtocolsResponse {
  protocols: string[]
}

export interface DocumentSourceInfo {
  id: number
  protocol_name: string
  doc_type: string
  url: string
  enabled: boolean
}

export interface SourcesResponse {
  sources: DocumentSourceInfo[]
}

export interface StreamEvent {
  type: 'chunk' | 'done' | 'error'
  content?: string
  message?: string
  model?: string
  documents_retrieved?: number
  cache_hit?: boolean
  sources?: DocumentSource[]
}
