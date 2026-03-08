<template>
  <div v-if="message" class="rounded-md bg-red-50 p-4 text-red-700">
    {{ message }}
  </div>
</template>

<script setup lang="ts">
defineProps<{
  message: string | null
}>()
</script>
