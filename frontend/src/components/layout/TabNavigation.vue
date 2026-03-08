<template>
  <div class="border-b border-gray-200">
    <nav class="-mb-px flex space-x-8">
      <button
        v-for="tab in tabs"
        :key="tab.name"
        type="button"
        @click="$emit('tab-change', tab.name)"
        :class="[
          'whitespace-nowrap border-b-2 px-1 py-4 text-sm font-medium',
          activeTab === tab.name
            ? 'border-primary-500 text-primary-600'
            : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700',
        ]"
      >
        <span class="mr-2">{{ tab.icon }}</span>
        {{ tab.label }}
      </button>
    </nav>
  </div>
</template>

<script setup lang="ts">
interface Tab {
  name: string
  label: string
  icon: string
}

defineProps<{
  activeTab: string
}>()

defineEmits<{
  'tab-change': [tabName: string]
}>()

const tabs: Tab[] = [
  { name: 'query', label: 'Query', icon: '🔍' },
  { name: 'feedback', label: 'Feedback', icon: '💬' },
  { name: 'system', label: 'System Info', icon: 'ℹ️' },
]
</script>
