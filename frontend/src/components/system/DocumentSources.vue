<template>
  <div class="space-y-4">
    <h3 class="text-lg font-medium text-gray-900">Document Sources</h3>
    <div v-if="systemStore.sources.length" class="space-y-2">
      <div
        v-for="src in systemStore.sources"
        :key="src.id"
        class="rounded border border-gray-200 bg-white px-4 py-2 text-sm"
      >
        <span :class="src.enabled ? 'text-green-600' : 'text-red-600'">
          {{ src.enabled ? '🟢' : '🔴' }}
        </span>
        <strong>{{ src.protocol_name }}</strong> — {{ src.doc_type }} — {{ src.url }}
      </div>
    </div>
    <p v-else class="text-sm text-gray-500">No sources configured yet.</p>
  </div>
</template>

<script setup lang="ts">
import { useSystemStore } from '@/stores/system'
import { onMounted } from 'vue'

const systemStore = useSystemStore()

onMounted(() => {
  systemStore.loadSources()
})
</script>
