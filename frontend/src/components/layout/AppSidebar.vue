<template>
  <aside class="w-64 shrink-0 border-l border-gray-200 bg-gray-50 p-6">
    <h2 class="text-lg font-semibold text-gray-900">🐸 About DeFrog</h2>
    <p class="mt-2 text-sm text-gray-600">
      DeFrog is a RAG (Retrieval Augmented Generation) system for querying DeFi protocol
      documentation.
    </p>
    <ul class="mt-4 space-y-2 text-sm text-gray-600">
      <li>• Hybrid vector + keyword search</li>
      <li>• Streaming answers (SSE)</li>
      <li>• Query caching</li>
      <li>• Answer feedback</li>
    </ul>
    <p class="mt-4 text-xs text-gray-500">
      PostgreSQL + pgvector • FastAPI • Vue 3
    </p>
    <button
      type="button"
      @click="$emit('refresh')"
      class="mt-6 w-full btn-secondary"
    >
      🔄 Refresh
    </button>
  </aside>
</template>

<script setup lang="ts">
defineEmits<{
  refresh: []
}>()
</script>
