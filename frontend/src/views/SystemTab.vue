<template>
  <div class="space-y-8">
    <h2 class="text-xl font-semibold text-gray-900">System Information</h2>
    <SystemHealth />
    <hr class="border-gray-200" />
    <ProtocolsList />
    <hr class="border-gray-200" />
    <DocumentSources />
    <hr class="border-gray-200" />
    <UsageInstructions />
  </div>
</template>

<script setup lang="ts">
import DocumentSources from '@/components/system/DocumentSources.vue'
import ProtocolsList from '@/components/system/ProtocolsList.vue'
import SystemHealth from '@/components/system/SystemHealth.vue'
import UsageInstructions from '@/components/system/UsageInstructions.vue'
</script>
