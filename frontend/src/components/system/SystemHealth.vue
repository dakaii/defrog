<template>
  <div class="space-y-6">
    <h3 class="text-lg font-medium text-gray-900">System Health</h3>
    <div v-if="systemStore.isLoading" class="flex justify-center py-8">
      <LoadingSpinner />
    </div>
    <div
      v-else-if="systemStore.health"
      class="grid grid-cols-1 gap-4 md:grid-cols-3"
    >
      <MetricCard
        title="System Status"
        :value="formatStatus(systemStore.health.status)"
        :color="systemStore.health.status === 'healthy' ? 'green' : 'red'"
      />
      <MetricCard
        title="PostgreSQL"
        :value="formatBooleanStatus(systemStore.health.postgres)"
        :color="systemStore.health.postgres ? 'green' : 'red'"
      />
      <MetricCard
        title="OpenAI API"
        :value="formatBooleanStatus(systemStore.health.openai)"
        :color="systemStore.health.openai ? 'green' : 'red'"
      />
    </div>
    <div v-else class="card">
      <p class="py-4 text-center text-red-600">
        Failed to load system health information
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import LoadingSpinner from '@/components/shared/LoadingSpinner.vue'
import MetricCard from '@/components/shared/MetricCard.vue'
import { useSystemStore } from '@/stores/system'
import { onMounted } from 'vue'

const systemStore = useSystemStore()

onMounted(() => {
  systemStore.loadHealth()
})

function formatStatus(status: string): string {
  const icon = status === 'healthy' ? '✅' : '⚠️'
  return `${icon} ${status.charAt(0).toUpperCase() + status.slice(1)}`
}

function formatBooleanStatus(value: boolean): string {
  return value ? '✅ Online' : '❌ Offline'
}
</script>
