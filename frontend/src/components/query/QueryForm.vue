<template>
  <div class="space-y-6">
    <ExampleQuestions @select-example="handleExampleSelect" />
    <div class="form-group">
      <label for="query" class="form-label">Enter your DeFi question:</label>
      <textarea
        id="query"
        v-model="queryStore.currentQuery"
        class="form-input"
        rows="4"
        placeholder="e.g., How does Uniswap liquidity provision work?"
      />
    </div>
    <AdvancedOptions />
    <button
      type="button"
      @click="handleSubmit"
      :disabled="!queryStore.hasCurrentQuery || queryStore.isLoading"
      class="btn-primary flex w-full items-center justify-center"
    >
      <LoadingSpinner v-if="queryStore.isLoading" class="mr-2" size="sm" />
      🚀 Ask Question
    </button>
  </div>
</template>

<script setup lang="ts">
import LoadingSpinner from '@/components/shared/LoadingSpinner.vue'
import { useQueryStore } from '@/stores/query'
import AdvancedOptions from './AdvancedOptions.vue'
import ExampleQuestions from './ExampleQuestions.vue'

const queryStore = useQueryStore()

function handleExampleSelect(example: string) {
  queryStore.currentQuery = example
}

async function handleSubmit() {
  await queryStore.executeQuery()
}
</script>
