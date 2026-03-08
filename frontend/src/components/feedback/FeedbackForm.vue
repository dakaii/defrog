<template>
  <div class="space-y-6">
    <QueryContext
      v-if="feedbackStore.lastQuery"
      :query="feedbackStore.lastQuery"
      :answer="feedbackStore.lastAnswer"
    />
    <div v-else class="card">
      <p class="py-8 text-center text-gray-500">
        Ask a question in the Query tab first, then come back here to rate the answer.
      </p>
    </div>

    <div v-if="feedbackStore.lastQuery" class="card space-y-6">
      <h3 class="text-lg font-medium text-gray-900">Rate the Answer</h3>
      <div class="form-group">
        <label class="form-label">Rating (1 = poor, 5 = excellent)</label>
        <div class="flex items-center space-x-2">
          <button
            v-for="star in 5"
            :key="star"
            type="button"
            @click="rating = star"
            :class="[
              'rounded p-1 transition-colors',
              star <= rating ? 'text-yellow-400' : 'text-gray-300',
            ]"
          >
            ★
          </button>
          <span class="ml-2 text-sm text-gray-600">({{ rating }}/5)</span>
        </div>
      </div>
      <div class="form-group">
        <label for="comment" class="form-label">Optional comment</label>
        <textarea
          id="comment"
          v-model="comment"
          class="form-input"
          rows="3"
          placeholder="Share your thoughts about the answer quality..."
        />
      </div>
      <button
        type="button"
        @click="submitFeedback"
        :disabled="isSubmitting"
        class="btn-primary flex items-center"
      >
        <LoadingSpinner v-if="isSubmitting" class="mr-2" size="sm" />
        Submit Feedback
      </button>
      <div
        v-if="submissionResult"
        class="rounded-md border border-green-200 bg-green-50 p-4"
      >
        <p class="text-green-800">
          ✅ Feedback submitted successfully! (ID: {{ submissionResult.id }})
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import LoadingSpinner from '@/components/shared/LoadingSpinner.vue'
import { useFeedbackStore } from '@/stores/feedback'
import { ref } from 'vue'
import QueryContext from './QueryContext.vue'

const feedbackStore = useFeedbackStore()

const rating = ref(3)
const comment = ref('')
const isSubmitting = ref(false)
const submissionResult = ref<{ id: number } | null>(null)

async function submitFeedback() {
  if (!feedbackStore.lastQuery) return

  isSubmitting.value = true
  submissionResult.value = null

  try {
    const result = await feedbackStore.submitFeedback({
      query: feedbackStore.lastQuery,
      answer: feedbackStore.lastAnswer,
      rating: rating.value,
      comment: comment.value || undefined,
    })
    submissionResult.value = result
    rating.value = 3
    comment.value = ''
  } catch (error) {
    console.error('Failed to submit feedback:', error)
  } finally {
    isSubmitting.value = false
  }
}
</script>
