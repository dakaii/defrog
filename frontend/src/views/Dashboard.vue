<template>
  <div class="flex min-h-screen bg-gray-50">
    <div class="flex flex-1 flex-col">
      <AppHeader />
      <main class="flex-1 px-4 py-8 sm:px-6 lg:px-8">
        <div class="mx-auto max-w-7xl">
          <TabNavigation :active-tab="activeTab" @tab-change="activeTab = $event" />
          <div class="mt-6">
            <QueryTab v-show="activeTab === 'query'" />
            <FeedbackTab v-show="activeTab === 'feedback'" />
            <SystemTab v-show="activeTab === 'system'" />
          </div>
        </div>
      </main>
    </div>
    <AppSidebar @refresh="handleRefresh" />
  </div>
</template>

<script setup lang="ts">
import AppHeader from '@/components/layout/AppHeader.vue'
import AppSidebar from '@/components/layout/AppSidebar.vue'
import TabNavigation from '@/components/layout/TabNavigation.vue'
import { useSystemStore } from '@/stores/system'
import { ref } from 'vue'
import FeedbackTab from './FeedbackTab.vue'
import QueryTab from './QueryTab.vue'
import SystemTab from './SystemTab.vue'

const activeTab = ref('query')
const systemStore = useSystemStore()

function handleRefresh() {
  systemStore.loadAll()
}
</script>
