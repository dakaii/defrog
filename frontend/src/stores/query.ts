import { apiService } from '@/services/api'
import type { DocumentSource, QueryRequest, QueryResponse } from '@/types/api'
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { useFeedbackStore } from './feedback'

export const useQueryStore = defineStore('query', () => {
  const currentQuery = ref('')
  const topK = ref(5)
  const useStreaming = ref(true)
  const isLoading = ref(false)
  const currentAnswer = ref('')
  const queryMetadata = ref<Partial<QueryResponse>>({})
  const sources = ref<DocumentSource[]>([])
  const error = ref<string | null>(null)

  const hasCurrentQuery = computed(() => currentQuery.value.trim().length > 0)
  const hasAnswer = computed(() => currentAnswer.value.length > 0)

  async function executeQuery() {
    if (!hasCurrentQuery.value) return

    isLoading.value = true
    error.value = null
    currentAnswer.value = ''
    sources.value = []
    queryMetadata.value = {}

    const request: QueryRequest = {
      query: currentQuery.value,
      top_k: topK.value,
    }

    const feedbackStore = useFeedbackStore()

    try {
      if (useStreaming.value) {
        await executeStreamingQuery(request)
      } else {
        await executeBatchQuery(request)
      }

      feedbackStore.setLastQuery(currentQuery.value, currentAnswer.value)
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'An error occurred'
    } finally {
      isLoading.value = false
    }
  }

  async function executeStreamingQuery(request: QueryRequest) {
    for await (const event of apiService.streamQuery(request)) {
      if (event.type === 'chunk' && event.content) {
        currentAnswer.value += event.content
      } else if (event.type === 'done') {
        queryMetadata.value = {
          model_used: event.model,
          documents_retrieved: event.documents_retrieved,
          cache_hit: event.cache_hit,
        }
        sources.value = event.sources || []
      } else if (event.type === 'error') {
        throw new Error(event.message || 'Streaming error')
      }
    }
  }

  async function executeBatchQuery(request: QueryRequest) {
    const response = await apiService.query(request)
    currentAnswer.value = response.answer
    queryMetadata.value = response
    sources.value = response.sources
  }

  function clearQuery() {
    currentQuery.value = ''
    currentAnswer.value = ''
    sources.value = []
    queryMetadata.value = {}
    error.value = null
  }

  return {
    currentQuery,
    topK,
    useStreaming,
    isLoading,
    currentAnswer,
    queryMetadata,
    sources,
    error,
    hasCurrentQuery,
    hasAnswer,
    executeQuery,
    clearQuery,
  }
})
