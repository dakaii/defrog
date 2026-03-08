# Component Mapping & Implementation Guide

## Overview

This document provides a detailed mapping between Streamlit components and their Vue.js equivalents, along with implementation examples for each component in the DeFrog migration.

## Streamlit to Vue.js Component Mapping

### Layout Components

#### 1. Main Application Structure

**Streamlit (app.py:14-18)**
```python
st.set_page_config(
    page_title="DeFrog - DeFi RAG",
    page_icon="🐸",
    layout="wide"
)
```

**Vue.js Equivalent: `AppLayout.vue`**
```vue
<template>
  <div class="min-h-screen bg-gray-50">
    <AppHeader />
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <router-view />
    </div>
    <AppSidebar v-if="showSidebar" @close="showSidebar = false" />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useHead } from '@vueuse/head'
import AppHeader from './AppHeader.vue'
import AppSidebar from './AppSidebar.vue'

const showSidebar = ref(false)

useHead({
  title: 'DeFrog - DeFi RAG',
  meta: [
    { name: 'description', content: 'Query DeFi protocol whitepapers using RAG' }
  ]
})
</script>
```

#### 2. Tab Navigation

**Streamlit (app.py:80)**
```python
tab1, tab2, tab3 = st.tabs(["🔍 Query", "💬 Feedback", "ℹ️ System Info"])
```

**Vue.js Equivalent: `TabNavigation.vue`**
```vue
<template>
  <div class="border-b border-gray-200">
    <nav class="-mb-px flex space-x-8">
      <button
        v-for="tab in tabs"
        :key="tab.name"
        @click="$emit('tab-change', tab.name)"
        :class="[
          'whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm',
          activeTab === tab.name
            ? 'border-primary-500 text-primary-600'
            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
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
  { name: 'system', label: 'System Info', icon: 'ℹ️' }
]
</script>
```

### Query Tab Components

#### 3. Query Input Form

**Streamlit (app.py:97-106)**
```python
query = st.text_area(
    "Enter your DeFi question:",
    placeholder="e.g., How does Uniswap liquidity provision work?",
    height=100
)

with st.expander("Advanced Options"):
    top_k = st.slider("Number of documents to retrieve", 3, 10, 5)
    use_streaming = st.checkbox("Stream answer token-by-token", value=True)
```

**Vue.js Equivalent: `QueryForm.vue`**
```vue
<template>
  <div class="space-y-6">
    <!-- Example Questions -->
    <ExampleQuestions @select-example="handleExampleSelect" />
    
    <!-- Query Input -->
    <div class="form-group">
      <label for="query" class="form-label">
        Enter your DeFi question:
      </label>
      <textarea
        id="query"
        v-model="queryStore.currentQuery"
        class="form-input"
        rows="4"
        placeholder="e.g., How does Uniswap liquidity provision work?"
      />
    </div>

    <!-- Advanced Options -->
    <AdvancedOptions />

    <!-- Submit Button -->
    <button
      @click="handleSubmit"
      :disabled="!queryStore.hasCurrentQuery || queryStore.isLoading"
      class="btn-primary w-full"
    >
      <LoadingSpinner v-if="queryStore.isLoading" class="mr-2" />
      🚀 Ask Question
    </button>
  </div>
</template>

<script setup lang="ts">
import { useQueryStore } from '@/stores/query'
import ExampleQuestions from './ExampleQuestions.vue'
import AdvancedOptions from './AdvancedOptions.vue'
import LoadingSpinner from '@/components/shared/LoadingSpinner.vue'

const queryStore = useQueryStore()

function handleExampleSelect(example: string) {
  queryStore.currentQuery = example
}

async function handleSubmit() {
  await queryStore.executeQuery()
}
</script>
```

#### 4. Example Questions

**Streamlit (app.py:86-94)**
```python
with st.expander("Example Questions"):
    st.markdown("""
    - How does Uniswap V3 concentrated liquidity work?
    - What is the difference between Uniswap V2 and V3?
    - How does Aave's liquidation mechanism work?
    """)
```

**Vue.js Equivalent: `ExampleQuestions.vue`**
```vue
<template>
  <div class="card">
    <button
      @click="isOpen = !isOpen"
      class="flex items-center justify-between w-full text-left"
    >
      <h3 class="font-medium text-gray-900">Example Questions</h3>
      <ChevronDownIcon 
        :class="['w-5 h-5 transition-transform', isOpen && 'rotate-180']" 
      />
    </button>
    
    <Transition name="fade">
      <div v-if="isOpen" class="mt-4 space-y-2">
        <button
          v-for="example in examples"
          :key="example"
          @click="$emit('select-example', example)"
          class="block w-full text-left px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded transition-colors"
        >
          {{ example }}
        </button>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ChevronDownIcon } from '@heroicons/vue/24/outline'

defineEmits<{
  'select-example': [example: string]
}>()

const isOpen = ref(false)

const examples = [
  'How does Uniswap V3 concentrated liquidity work?',
  'What is the difference between Uniswap V2 and V3?',
  'How does Aave\'s liquidation mechanism work?',
  'Explain Compound\'s interest rate model',
  'What are the risks of providing liquidity in Curve?',
  'How does MakerDAO maintain DAI\'s peg?'
]
</script>
```

#### 5. Streaming Answer Display

**Streamlit (app.py:115-116)**
```python
st.subheader("Answer")
answer = st.write_stream(stream_query(query, top_k))
```

**Vue.js Equivalent: `StreamingAnswer.vue`**
```vue
<template>
  <div class="space-y-4">
    <h3 class="text-lg font-medium text-gray-900">Answer</h3>
    
    <div class="card min-h-[200px]">
      <div 
        v-if="queryStore.isLoading && !queryStore.currentAnswer"
        class="flex items-center justify-center h-32"
      >
        <LoadingSpinner />
        <span class="ml-2 text-gray-500">Generating answer...</span>
      </div>
      
      <div 
        v-else-if="queryStore.currentAnswer"
        class="prose max-w-none"
      >
        <div 
          v-html="formattedAnswer"
          :class="{ 'streaming-text': queryStore.isLoading }"
        />
        
        <div 
          v-if="queryStore.isLoading"
          class="inline-block w-2 h-4 bg-primary-500 animate-pulse ml-1"
        />
      </div>
      
      <div 
        v-else-if="queryStore.error"
        class="text-red-600"
      >
        Error: {{ queryStore.error }}
      </div>
      
      <div 
        v-else
        class="text-gray-500 text-center py-8"
      >
        Ask a question to see the answer here
      </div>
    </div>
    
    <QueryMetadata v-if="queryStore.queryMetadata" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { marked } from 'marked'
import { useQueryStore } from '@/stores/query'
import LoadingSpinner from '@/components/shared/LoadingSpinner.vue'
import QueryMetadata from './QueryMetadata.vue'

const queryStore = useQueryStore()

const formattedAnswer = computed(() => {
  if (!queryStore.currentAnswer) return ''
  return marked(queryStore.currentAnswer)
})
</script>
```

#### 6. Query Metadata

**Streamlit (app.py:120-127)**
```python
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model", meta.get("model", "Unknown"))
with col2:
    st.metric("Documents Retrieved", meta.get("documents_retrieved", 0))
with col3:
    cache_label = "Yes" if meta.get("cache_hit") else "No"
    st.metric("Cache Hit", cache_label)
```

**Vue.js Equivalent: `QueryMetadata.vue`**
```vue
<template>
  <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
    <MetricCard
      title="Model"
      :value="metadata.model_used || 'Unknown'"
    />
    <MetricCard
      title="Documents Retrieved"
      :value="metadata.documents_retrieved?.toString() || '0'"
    />
    <MetricCard
      title="Cache Hit"
      :value="metadata.cache_hit ? 'Yes' : 'No'"
    />
    <MetricCard
      v-if="metadata.latency_ms"
      title="Latency"
      :value="`${metadata.latency_ms}ms`"
    />
  </div>
</template>

<script setup lang="ts">
import type { QueryResponse } from '@/types/api'
import MetricCard from '@/components/shared/MetricCard.vue'

defineProps<{
  metadata: Partial<QueryResponse>
}>()
</script>
```

#### 7. Source Documents

**Streamlit (app.py:129-139)**
```python
if meta.get("sources"):
    st.subheader("Sources")
    for i, source in enumerate(meta["sources"], 1):
        with st.expander(
            f"Source {i}: {source.get('protocol', 'Unknown')} — {source.get('doc_type', 'document')}"
        ):
            st.write(f"**Title:** {source.get('title', 'Unknown')}")
            st.write(f"**Similarity:** {source.get('similarity', 0):.3f}")
            if source.get("url"):
                st.write(f"**URL:** {source['url']}")
```

**Vue.js Equivalent: `SourceDocuments.vue`**
```vue
<template>
  <div v-if="sources.length" class="space-y-4">
    <h3 class="text-lg font-medium text-gray-900">Sources</h3>
    
    <div class="space-y-3">
      <div
        v-for="(source, index) in sources"
        :key="index"
        class="border rounded-lg overflow-hidden"
      >
        <button
          @click="toggleSource(index)"
          class="w-full px-4 py-3 text-left bg-gray-50 hover:bg-gray-100 transition-colors flex justify-between items-center"
        >
          <span class="font-medium">
            Source {{ index + 1 }}: {{ source.protocol }} — {{ source.doc_type }}
          </span>
          <ChevronDownIcon 
            :class="[
              'w-5 h-5 transition-transform',
              openSources.has(index) && 'rotate-180'
            ]"
          />
        </button>
        
        <Transition name="fade">
          <div v-if="openSources.has(index)" class="p-4 bg-white border-t">
            <div class="space-y-2">
              <p><strong>Title:</strong> {{ source.title }}</p>
              <p><strong>Similarity:</strong> {{ source.similarity.toFixed(3) }}</p>
              <p v-if="source.url">
                <strong>URL:</strong> 
                <a 
                  :href="source.url" 
                  target="_blank"
                  rel="noopener noreferrer"
                  class="text-primary-600 hover:text-primary-800 underline ml-1"
                >
                  {{ source.url }}
                  <ExternalLinkIcon class="w-4 h-4 inline ml-1" />
                </a>
              </p>
              <div v-if="source.content" class="mt-3">
                <strong>Content Preview:</strong>
                <div class="mt-1 p-3 bg-gray-50 rounded text-sm text-gray-700">
                  {{ source.content.substring(0, 200) }}
                  <span v-if="source.content.length > 200">...</span>
                </div>
              </div>
            </div>
          </div>
        </Transition>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ChevronDownIcon, ExternalLinkIcon } from '@heroicons/vue/24/outline'
import type { DocumentSource } from '@/types/api'

defineProps<{
  sources: DocumentSource[]
}>()

const openSources = ref(new Set<number>())

function toggleSource(index: number) {
  if (openSources.value.has(index)) {
    openSources.value.delete(index)
  } else {
    openSources.value.add(index)
  }
}
</script>
```

### Feedback Tab Components

#### 8. Feedback Form

**Streamlit (app.py:190-207)**
```python
rating = st.slider("Rating (1 = poor, 5 = excellent)", 1, 5, 3)
comment = st.text_area("Optional comment", height=80)

if st.button("Submit Feedback"):
    result = fetch_api(
        "/feedback",
        method="POST",
        json_body={
            "query": last_query,
            "answer": last_answer,
            "rating": rating,
            "comment": comment or None,
        },
    )
```

**Vue.js Equivalent: `FeedbackForm.vue`**
```vue
<template>
  <div class="space-y-6">
    <QueryContext 
      v-if="feedbackStore.lastQuery"
      :query="feedbackStore.lastQuery"
      :answer="feedbackStore.lastAnswer"
    />
    
    <div v-else class="card">
      <p class="text-gray-500 text-center py-8">
        Ask a question in the Query tab first, then come back here to rate the answer.
      </p>
    </div>

    <div v-if="feedbackStore.lastQuery" class="card space-y-6">
      <h3 class="text-lg font-medium text-gray-900">Rate the Answer</h3>
      
      <!-- Rating -->
      <div class="form-group">
        <label class="form-label">
          Rating (1 = poor, 5 = excellent)
        </label>
        <div class="flex items-center space-x-2">
          <button
            v-for="star in 5"
            :key="star"
            @click="rating = star"
            :class="[
              'w-8 h-8 transition-colors',
              star <= rating ? 'text-yellow-400' : 'text-gray-300'
            ]"
          >
            <StarIcon class="w-full h-full" :class="star <= rating ? 'fill-current' : ''" />
          </button>
          <span class="ml-2 text-sm text-gray-600">({{ rating }}/5)</span>
        </div>
      </div>

      <!-- Comment -->
      <div class="form-group">
        <label for="comment" class="form-label">
          Optional comment
        </label>
        <textarea
          id="comment"
          v-model="comment"
          class="form-input"
          rows="3"
          placeholder="Share your thoughts about the answer quality..."
        />
      </div>

      <!-- Submit Button -->
      <button
        @click="submitFeedback"
        :disabled="isSubmitting"
        class="btn-primary"
      >
        <LoadingSpinner v-if="isSubmitting" class="mr-2" />
        Submit Feedback
      </button>

      <!-- Success Message -->
      <div v-if="submissionResult" class="p-4 bg-green-50 border border-green-200 rounded-md">
        <p class="text-green-800">
          ✅ Feedback submitted successfully! (ID: {{ submissionResult.id }})
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { StarIcon } from '@heroicons/vue/24/solid'
import { useFeedbackStore } from '@/stores/feedback'
import QueryContext from './QueryContext.vue'
import LoadingSpinner from '@/components/shared/LoadingSpinner.vue'

const feedbackStore = useFeedbackStore()

const rating = ref(3)
const comment = ref('')
const isSubmitting = ref(false)
const submissionResult = ref<{ id: string } | null>(null)

async function submitFeedback() {
  if (!feedbackStore.lastQuery) return
  
  isSubmitting.value = true
  submissionResult.value = null
  
  try {
    const result = await feedbackStore.submitFeedback({
      query: feedbackStore.lastQuery,
      answer: feedbackStore.lastAnswer,
      rating: rating.value,
      comment: comment.value || undefined
    })
    
    submissionResult.value = result
    rating.value = 3
    comment.value = ''
  } catch (error) {
    console.error('Failed to submit feedback:', error)
  } finally {
    isSubmitting.value = false
  }
}
</script>
```

### System Info Tab Components

#### 9. System Health

**Streamlit (app.py:214-228)**
```python
health = fetch_api("/health")
if health:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_icon = "✅" if health["status"] == "healthy" else "⚠️"
        st.metric("System Status", f"{status_icon} {health['status'].title()}")
    
    with col2:
        pg_icon = "✅" if health["postgres"] else "❌"
        st.metric("PostgreSQL", pg_icon)
    
    with col3:
        openai_icon = "✅" if health["openai"] else "❌"
        st.metric("OpenAI API", openai_icon)
```

**Vue.js Equivalent: `SystemHealth.vue`**
```vue
<template>
  <div class="space-y-6">
    <h3 class="text-lg font-medium text-gray-900">System Health</h3>
    
    <div v-if="systemStore.isLoading" class="flex justify-center py-8">
      <LoadingSpinner />
    </div>
    
    <div v-else-if="systemStore.health" class="grid grid-cols-1 md:grid-cols-3 gap-4">
      <MetricCard
        title="System Status"
        :value="formatStatus(systemStore.health.status)"
        :color="systemStore.health.status === 'healthy' ? 'green' : 'red'"
      />
      <MetricCard
        title="PostgreSQL"
        :value="formatBooleanStatus(systemStore.health.postgres)"
        :color="systemStore.health.postgres ? 'green' : 'red'"
      />
      <MetricCard
        title="OpenAI API"
        :value="formatBooleanStatus(systemStore.health.openai)"
        :color="systemStore.health.openai ? 'green' : 'red'"
      />
    </div>
    
    <div v-else class="card">
      <p class="text-red-600 text-center py-4">
        Failed to load system health information
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useSystemStore } from '@/stores/system'
import MetricCard from '@/components/shared/MetricCard.vue'
import LoadingSpinner from '@/components/shared/LoadingSpinner.vue'

const systemStore = useSystemStore()

onMounted(() => {
  systemStore.loadHealth()
})

function formatStatus(status: string): string {
  const icon = status === 'healthy' ? '✅' : '⚠️'
  return `${icon} ${status.charAt(0).toUpperCase() + status.slice(1)}`
}

function formatBooleanStatus(value: boolean): string {
  return value ? '✅ Online' : '❌ Offline'
}
</script>
```

### Shared Components

#### 10. Metric Card

**Vue.js Implementation: `MetricCard.vue`**
```vue
<template>
  <div class="metric-card">
    <div class="flex items-center justify-between">
      <div>
        <p class="text-sm font-medium text-gray-600">{{ title }}</p>
        <p :class="[
          'text-2xl font-semibold',
          colorClasses
        ]">
          {{ value }}
        </p>
      </div>
      <div v-if="icon" class="flex-shrink-0">
        <component :is="icon" class="w-8 h-8 text-gray-400" />
      </div>
    </div>
    <div v-if="subtitle" class="mt-1">
      <p class="text-xs text-gray-500">{{ subtitle }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  title: string
  value: string | number
  subtitle?: string
  color?: 'default' | 'green' | 'red' | 'yellow' | 'blue'
  icon?: any
}

const props = withDefaults(defineProps<Props>(), {
  color: 'default'
})

const colorClasses = computed(() => {
  const colors = {
    default: 'text-gray-900',
    green: 'text-green-600',
    red: 'text-red-600',
    yellow: 'text-yellow-600',
    blue: 'text-blue-600'
  }
  return colors[props.color]
})
</script>
```

#### 11. Loading Spinner

**Vue.js Implementation: `LoadingSpinner.vue`**
```vue
<template>
  <div 
    :class="[
      'animate-spin rounded-full border-2 border-gray-300',
      'border-t-primary-600',
      sizeClasses
    ]"
    role="status"
    aria-label="Loading"
  >
    <span class="sr-only">Loading...</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  size?: 'sm' | 'md' | 'lg'
}

const props = withDefaults(defineProps<Props>(), {
  size: 'md'
})

const sizeClasses = computed(() => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8'
  }
  return sizes[props.size]
})
</script>
```

## Implementation Priority

### Phase 1: Core Layout (Days 1-2)
1. `AppLayout.vue` - Main application shell
2. `TabNavigation.vue` - Tab switching
3. `MetricCard.vue` - Reusable metric display
4. `LoadingSpinner.vue` - Loading states

### Phase 2: Query Functionality (Days 3-5)
1. `QueryForm.vue` - Main query interface
2. `ExampleQuestions.vue` - Example questions
3. `AdvancedOptions.vue` - Query settings
4. `StreamingAnswer.vue` - Real-time answer display
5. `QueryMetadata.vue` - Answer metadata
6. `SourceDocuments.vue` - Source documentation

### Phase 3: Feedback & System (Days 6-7)
1. `FeedbackForm.vue` - Rating and comments
2. `QueryContext.vue` - Previous query display
3. `SystemHealth.vue` - Health monitoring
4. `ProtocolsList.vue` - Available protocols
5. `DocumentSources.vue` - Source management

### Phase 4: Polish & Testing (Days 8-9)
1. Error handling components
2. Mobile responsiveness
3. Accessibility improvements
4. Performance optimization

This mapping provides a clear path from Streamlit components to Vue.js implementations while maintaining feature parity and improving user experience.