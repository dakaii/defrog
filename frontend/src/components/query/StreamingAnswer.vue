<template>
  <div class="space-y-4">
    <h3 class="text-lg font-medium text-gray-900">Answer</h3>
    <div class="card min-h-[200px]">
      <div
        v-if="queryStore.isLoading && !queryStore.currentAnswer"
        class="flex h-32 items-center justify-center"
      >
        <LoadingSpinner />
        <span class="ml-2 text-gray-500">Generating answer...</span>
      </div>
      <div v-else-if="queryStore.currentAnswer" class="prose max-w-none">
        <div
          v-html="formattedAnswer"
          :class="{ 'streaming-text': queryStore.isLoading }"
        />
        <div
          v-if="queryStore.isLoading"
          class="ml-1 inline-block h-4 w-2 animate-pulse bg-primary-500"
        />
      </div>
      <div v-else-if="queryStore.error" class="text-red-600">
        Error: {{ queryStore.error }}
      </div>
      <div v-else class="py-8 text-center text-gray-500">
        Ask a question to see the answer here
      </div>
    </div>
    <QueryMetadata v-if="queryStore.queryMetadata" :metadata="queryStore.queryMetadata" />
  </div>
</template>

<script setup lang="ts">
import LoadingSpinner from '@/components/shared/LoadingSpinner.vue'
import { useQueryStore } from '@/stores/query'
import { marked } from 'marked'
import { computed } from 'vue'
import QueryMetadata from './QueryMetadata.vue'

const queryStore = useQueryStore()

const formattedAnswer = computed(() => {
  if (!queryStore.currentAnswer) return ''
  return marked(queryStore.currentAnswer)
})
</script>
