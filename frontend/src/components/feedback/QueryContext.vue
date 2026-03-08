<template>
  <div class="card space-y-2">
    <p class="font-medium text-gray-900">Question:</p>
    <p class="text-gray-700">{{ query }}</p>
    <details class="mt-2">
      <summary class="cursor-pointer text-sm text-primary-600 hover:text-primary-800">
        View answer
      </summary>
      <div class="prose mt-2 max-w-none text-sm" v-html="formattedAnswer" />
    </details>
  </div>
</template>

<script setup lang="ts">
import { marked } from 'marked'
import { computed } from 'vue'

const props = defineProps<{
  query: string
  answer: string
}>()

const formattedAnswer = computed(() => marked(props.answer || ''))
</script>
