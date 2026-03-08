<template>
  <div class="metric-card">
    <p class="text-sm font-medium text-gray-600">{{ title }}</p>
    <p :class="['text-2xl font-semibold', colorClasses]">{{ value }}</p>
    <p v-if="subtitle" class="mt-1 text-xs text-gray-500">{{ subtitle }}</p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    title: string
    value: string | number
    subtitle?: string
    color?: 'default' | 'green' | 'red' | 'yellow' | 'blue'
  }>(),
  { color: 'default' }
)

const colorClasses = computed(() => {
  const colors = {
    default: 'text-gray-900',
    green: 'text-green-600',
    red: 'text-red-600',
    yellow: 'text-yellow-600',
    blue: 'text-blue-600',
  }
  return colors[props.color]
})
</script>
