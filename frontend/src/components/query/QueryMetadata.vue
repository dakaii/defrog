<template>
  <div v-if="Object.keys(metadata).length" class="grid grid-cols-1 gap-4 md:grid-cols-3">
    <MetricCard
      title="Model"
      :value="metadata.model_used || 'Unknown'"
    />
    <MetricCard
      title="Documents Retrieved"
      :value="String(metadata.documents_retrieved ?? 0)"
    />
    <MetricCard
      title="Cache Hit"
      :value="metadata.cache_hit ? 'Yes' : 'No'"
    />
    <MetricCard
      v-if="metadata.latency_ms"
      title="Latency"
      :value="`${metadata.latency_ms}ms`"
    />
  </div>
</template>

<script setup lang="ts">
import MetricCard from '@/components/shared/MetricCard.vue'
import type { QueryResponse } from '@/types/api'

defineProps<{
  metadata: Partial<QueryResponse>
}>()
</script>
