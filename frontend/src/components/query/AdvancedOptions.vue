<template>
  <div class="card">
    <button
      type="button"
      @click="isOpen = !isOpen"
      class="flex w-full items-center justify-between text-left"
    >
      <h3 class="font-medium text-gray-900">Advanced Options</h3>
      <span :class="['text-lg transition-transform', isOpen && 'rotate-180']">▼</span>
    </button>
    <Transition name="fade">
      <div v-if="isOpen" class="mt-4 space-y-4">
        <div class="form-group">
          <label class="form-label">Number of documents to retrieve</label>
          <input
            v-model.number="topK"
            type="range"
            min="3"
            max="10"
            class="w-full"
          />
          <span class="text-sm text-gray-600">{{ topK }}</span>
        </div>
        <div class="flex items-center">
          <input
            v-model="useStreaming"
            type="checkbox"
            id="streaming"
            class="h-4 w-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
          />
          <label for="streaming" class="ml-2 text-sm text-gray-700">
            Stream answer token-by-token
          </label>
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { useQueryStore } from '@/stores/query'
import { ref, watch } from 'vue'

const queryStore = useQueryStore()
const isOpen = ref(false)
const topK = ref(queryStore.topK)
const useStreaming = ref(queryStore.useStreaming)

watch(topK, (v) => {
  queryStore.topK = v
})
watch(useStreaming, (v) => {
  queryStore.useStreaming = v
})
</script>
