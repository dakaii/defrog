<template>
  <div class="grid grid-cols-1 gap-8 lg:grid-cols-2">
    <div>
      <h2 class="mb-6 text-xl font-semibold text-gray-900">Ask about DeFi Protocols</h2>
      <QueryForm />
    </div>
    <div>
      <StreamingAnswer />
      <SourceDocuments :sources="queryStore.sources" />
    </div>
  </div>
</template>

<script setup lang="ts">
import QueryForm from '@/components/query/QueryForm.vue'
import SourceDocuments from '@/components/query/SourceDocuments.vue'
import StreamingAnswer from '@/components/query/StreamingAnswer.vue'
import { useQueryStore } from '@/stores/query'

const queryStore = useQueryStore()
</script>
