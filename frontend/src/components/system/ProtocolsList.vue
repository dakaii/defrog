<template>
  <div class="space-y-4">
    <h3 class="text-lg font-medium text-gray-900">Available Protocols</h3>
    <div class="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
      <div
        v-for="protocol in systemStore.protocols"
        :key="protocol"
        class="rounded border border-gray-200 bg-white px-4 py-2"
      >
        • {{ protocol }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useSystemStore } from '@/stores/system'
import { onMounted } from 'vue'

const systemStore = useSystemStore()

onMounted(() => {
  systemStore.loadProtocols()
})
</script>
