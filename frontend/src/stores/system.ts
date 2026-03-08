import { apiService } from '@/services/api'
import type { DocumentSourceInfo, HealthResponse } from '@/types/api'
import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSystemStore = defineStore('system', () => {
  const health = ref<HealthResponse | null>(null)
  const protocols = ref<string[]>([])
  const sources = ref<DocumentSourceInfo[]>([])
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  async function loadHealth() {
    isLoading.value = true
    error.value = null
    try {
      health.value = await apiService.getHealth()
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to load health'
    } finally {
      isLoading.value = false
    }
  }

  async function loadProtocols() {
    try {
      const res = await apiService.getProtocols()
      protocols.value = res.protocols
    } catch {
      protocols.value = []
    }
  }

  async function loadSources() {
    try {
      const res = await apiService.getSources()
      sources.value = res.sources
    } catch {
      sources.value = []
    }
  }

  async function loadAll() {
    await Promise.all([loadHealth(), loadProtocols(), loadSources()])
  }

  return {
    health,
    protocols,
    sources,
    isLoading,
    error,
    loadHealth,
    loadProtocols,
    loadSources,
    loadAll,
  }
})
