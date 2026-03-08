<template>
  <div
    :class="[
      'animate-spin rounded-full border-2 border-gray-300 border-t-primary-600',
      sizeClasses,
    ]"
    role="status"
    aria-label="Loading"
  />
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    size?: 'sm' | 'md' | 'lg'
  }>(),
  { size: 'md' }
)

const sizeClasses = computed(() => {
  const sizes = { sm: 'w-4 h-4', md: 'w-6 h-6', lg: 'w-8 h-8' }
  return sizes[props.size]
})
</script>
