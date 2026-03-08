import type {
  FeedbackRequest,
  FeedbackResponse,
  HealthResponse,
  ProtocolsResponse,
  QueryRequest,
  QueryResponse,
  SourcesResponse,
  StreamEvent,
} from '@/types/api'
import axios, { type AxiosInstance, type AxiosResponse } from 'axios'

const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const apiKey = import.meta.env.VITE_API_KEY || ''

const headers: Record<string, string> = {
  'Content-Type': 'application/json',
}
if (apiKey) {
  headers['X-API-Key'] = apiKey
}

class ApiService {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers,
    })

    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message)
        return Promise.reject(error)
      }
    )
  }

  async query(request: QueryRequest): Promise<QueryResponse> {
    const response: AxiosResponse<QueryResponse> = await this.client.post('/query', request)
    return response.data
  }

  async *streamQuery(request: QueryRequest): AsyncGenerator<StreamEvent> {
    const response = await fetch(`${baseURL}/query/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        ...(apiKey && { 'X-API-Key': apiKey }),
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('Failed to get response reader')
    }

    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event: StreamEvent = JSON.parse(line.slice(6))
              yield event
            } catch {
              // ignore parse errors
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  }

  async submitFeedback(request: FeedbackRequest): Promise<FeedbackResponse> {
    const response: AxiosResponse<FeedbackResponse> = await this.client.post('/feedback', request)
    return response.data
  }

  async getHealth(): Promise<HealthResponse> {
    const response: AxiosResponse<HealthResponse> = await this.client.get('/health')
    return response.data
  }

  async getProtocols(): Promise<ProtocolsResponse> {
    const response: AxiosResponse<ProtocolsResponse> = await this.client.get('/protocols')
    return response.data
  }

  async getSources(): Promise<SourcesResponse> {
    const response: AxiosResponse<SourcesResponse> = await this.client.get('/sources')
    return response.data
  }
}

export const apiService = new ApiService()
