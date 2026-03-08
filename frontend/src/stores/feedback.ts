import { apiService } from '@/services/api'
import type { FeedbackRequest } from '@/types/api'
import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useFeedbackStore = defineStore('feedback', () => {
  const lastQuery = ref('')
  const lastAnswer = ref('')

  function setLastQuery(query: string, answer: string) {
    lastQuery.value = query
    lastAnswer.value = answer
  }

  async function submitFeedback(request: FeedbackRequest) {
    return apiService.submitFeedback(request)
  }

  return {
    lastQuery,
    lastAnswer,
    setLastQuery,
    submitFeedback,
  }
})
