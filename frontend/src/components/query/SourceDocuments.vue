<template>
  <div v-if="sources.length" class="space-y-4">
    <h3 class="text-lg font-medium text-gray-900">Sources</h3>
    <div class="space-y-3">
      <div
        v-for="(source, index) in sources"
        :key="index"
        class="overflow-hidden rounded-lg border"
      >
        <button
          type="button"
          @click="toggleSource(index)"
          class="flex w-full items-center justify-between bg-gray-50 px-4 py-3 text-left transition-colors hover:bg-gray-100"
        >
          <span class="font-medium">
            Source {{ index + 1 }}: {{ source.protocol }} — {{ source.doc_type }}
          </span>
          <span :class="['text-lg transition-transform', openSources.has(index) && 'rotate-180']">
            ▼
          </span>
        </button>
        <Transition name="fade">
          <div v-if="openSources.has(index)" class="border-t bg-white p-4">
            <div class="space-y-2">
              <p><strong>Title:</strong> {{ source.title }}</p>
              <p><strong>Similarity:</strong> {{ (source.similarity ?? 0).toFixed(3) }}</p>
              <p v-if="source.url">
                <strong>URL:</strong>
                <a
                  :href="source.url"
                  target="_blank"
                  rel="noopener noreferrer"
                  class="ml-1 text-primary-600 underline hover:text-primary-800"
                >
                  {{ source.url }}
                </a>
              </p>
            </div>
          </div>
        </Transition>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { DocumentSource } from '@/types/api'
import { ref } from 'vue'

defineProps<{
  sources: DocumentSource[]
}>()

const openSources = ref(new Set<number>())

function toggleSource(index: number) {
  const next = new Set(openSources.value)
  if (next.has(index)) {
    next.delete(index)
  } else {
    next.add(index)
  }
  openSources.value = next
}
</script>
