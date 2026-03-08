<template>
  <div class="card">
    <button
      type="button"
      @click="isOpen = !isOpen"
      class="flex w-full items-center justify-between text-left"
    >
      <h3 class="font-medium text-gray-900">Example Questions</h3>
      <span :class="['text-lg transition-transform', isOpen && 'rotate-180']">▼</span>
    </button>
    <Transition name="fade">
      <div v-if="isOpen" class="mt-4 space-y-2">
        <button
          v-for="example in examples"
          :key="example"
          type="button"
          @click="$emit('select-example', example)"
          class="block w-full rounded px-3 py-2 text-left text-sm text-gray-600 transition-colors hover:bg-gray-100"
        >
          {{ example }}
        </button>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

defineEmits<{
  'select-example': [example: string]
}>()

const isOpen = ref(false)

const examples = [
  'How does Uniswap V3 concentrated liquidity work?',
  'What is the difference between Uniswap V2 and V3?',
  "How does Aave's liquidation mechanism work?",
  "Explain Compound's interest rate model",
  'What are the risks of providing liquidity in Curve?',
  "How does MakerDAO maintain DAI's peg?",
]
</script>
