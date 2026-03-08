# Vue.js 3 Setup Guide for DeFrog Frontend

## Prerequisites

### System Requirements
- Bun 1.0+ (JavaScript runtime and package manager)
- Git

### Install Bun
```bash
# macOS/Linux
curl -fsSL https://bun.sh/install | bash

# Windows (PowerShell)
powershell -c "irm bun.sh/install.ps1 | iex"

# Verify installation
bun --version  # Should be 1.0+
```

## Project Initialization

### 1. Create Vue.js Project

```bash
# Navigate to project root
cd /Users/dakaii/Workspace/projects/defrog

# Create frontend directory
bunx create-vue@latest frontend

# During setup, select these options:
# ✅ TypeScript
# ✅ Router
# ✅ Pinia
# ❌ ESLint (we'll use Biome.js instead)
# ❌ Prettier (we'll use Biome.js instead)
# ❌ Vitest (we'll add later)
# ❌ End-to-End Testing (we'll add Playwright)
```

### 2. Navigate and Install Dependencies

```bash
cd frontend
bun install
```

### 3. Add Additional Dependencies

```bash
# Core dependencies
bun add axios @headlessui/vue @heroicons/vue

# Development dependencies
bun add -D tailwindcss@next postcss autoprefixer
bun add -D @tailwindcss/forms @tailwindcss/typography
bun add -D @types/node vitest @vue/test-utils
bun add -D playwright @playwright/test
bun add -D @biomejs/biome

# Optional: For better developer experience
bun add -D @vueuse/core @vueuse/head
```

## Configuration Files

### 1. Tailwind CSS Configuration

Create `tailwind.config.js`:
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        success: '#10b981',
        warning: '#f59e0b',
        error: '#ef4444',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}
```

### 2. PostCSS Configuration

Create `postcss.config.js`:
```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

### 3. Vite Configuration

Update `vite.config.ts`:
```typescript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['vue', 'vue-router', 'pinia'],
          'ui': ['@headlessui/vue', '@heroicons/vue'],
        },
      },
    },
  },
})
```

### 4. TypeScript Configuration

Update `tsconfig.json`:
```json
{
  "extends": "@vue/tsconfig/tsconfig.dom.json",
  "include": [
    "env.d.ts",
    "src/**/*",
    "src/**/*.vue"
  ],
  "exclude": [
    "src/**/__tests__/*"
  ],
  "compilerOptions": {
    "composite": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    },
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true
  }
}
```

### 5. Environment Variables

Create `.env.development`:
```env
VITE_API_URL=http://localhost:8000
VITE_API_KEY=your-dev-api-key
VITE_STREAMING_ENABLED=true
```

Create `.env.production`:
```env
VITE_API_URL=https://your-production-api.com
VITE_API_KEY=your-prod-api-key
VITE_STREAMING_ENABLED=true
```

### 6. Package.json Scripts

Update `package.json` scripts:
```json
{
  "scripts": {
    "dev": "bun --bun vite",
    "build": "vue-tsc && vite build",
    "preview": "vite preview",
    "test": "bun --bun vitest",
    "test:ui": "bun --bun vitest --ui",
    "test:coverage": "bun --bun vitest --coverage",
    "test:e2e": "bunx playwright test",
    "lint": "bunx @biomejs/biome lint --apply ./src",
    "format": "bunx @biomejs/biome format --write ./src",
    "check": "bunx @biomejs/biome check --apply ./src",
    "type-check": "vue-tsc --noEmit"
  }
}
```

## Project Structure Setup

### 1. Create Directory Structure

```bash
cd src
mkdir -p {components/{layout,query,feedback,system,shared},views,stores,services,composables,assets/{styles,images},types}
```

### 2. Update Main CSS File

Replace `src/assets/main.css`:
```css
@import 'tailwindcss/base';
@import 'tailwindcss/components';
@import 'tailwindcss/utilities';

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Custom component styles */
@layer components {
  .btn-primary {
    @apply bg-primary-600 text-white px-4 py-2 rounded-md hover:bg-primary-700 transition-colors;
  }
  
  .btn-secondary {
    @apply bg-gray-200 text-gray-900 px-4 py-2 rounded-md hover:bg-gray-300 transition-colors;
  }
  
  .card {
    @apply bg-white rounded-lg shadow-sm border border-gray-200 p-6;
  }
  
  .form-group {
    @apply mb-4;
  }
  
  .form-label {
    @apply block text-sm font-medium text-gray-700 mb-2;
  }
  
  .form-input {
    @apply w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500;
  }
  
  .metric-card {
    @apply bg-white rounded-lg p-4 border border-gray-200;
  }
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

/* Loading animation */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* Streaming text animation */
.streaming-text {
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
}
```

### 3. Create Basic Type Definitions

Create `src/types/api.ts`:
```typescript
export interface QueryRequest {
  query: string
  top_k: number
}

export interface QueryResponse {
  answer: string
  model_used: string
  latency_ms: number
  documents_retrieved: number
  cache_hit?: boolean
  sources: DocumentSource[]
}

export interface DocumentSource {
  protocol: string
  doc_type: string
  title: string
  similarity: number
  url?: string
  content?: string
}

export interface FeedbackRequest {
  query: string
  answer: string
  rating: number
  comment?: string
}

export interface FeedbackResponse {
  id: string
  status: 'accepted' | 'rejected'
}

export interface HealthResponse {
  status: 'healthy' | 'unhealthy'
  postgres: boolean
  openai: boolean
}

export interface ProtocolsResponse {
  protocols: string[]
}

export interface SourcesResponse {
  sources: DocumentSourceInfo[]
}

export interface DocumentSourceInfo {
  id: string
  protocol_name: string
  doc_type: string
  url: string
  enabled: boolean
}

export interface StreamEvent {
  type: 'chunk' | 'done' | 'error'
  content?: string
  message?: string
  model?: string
  documents_retrieved?: number
  cache_hit?: boolean
  sources?: DocumentSource[]
}
```

### 4. Create API Service

Create `src/services/api.ts`:
```typescript
import axios, { type AxiosInstance, type AxiosResponse } from 'axios'
import type {
  QueryRequest,
  QueryResponse,
  FeedbackRequest,
  FeedbackResponse,
  HealthResponse,
  ProtocolsResponse,
  SourcesResponse,
  StreamEvent
} from '@/types/api'

class ApiService {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: import.meta.env.VITE_API_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
        ...(import.meta.env.VITE_API_KEY && {
          'X-API-Key': import.meta.env.VITE_API_KEY
        })
      }
    })

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message)
        return Promise.reject(error)
      }
    )
  }

  async query(request: QueryRequest): Promise<QueryResponse> {
    const response: AxiosResponse<QueryResponse> = await this.client.post('/query', request)
    return response.data
  }

  async *streamQuery(request: QueryRequest): AsyncGenerator<StreamEvent> {
    const response = await fetch(`${import.meta.env.VITE_API_URL}/query/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
        ...(import.meta.env.VITE_API_KEY && {
          'X-API-Key': import.meta.env.VITE_API_KEY
        })
      },
      body: JSON.stringify(request)
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('Failed to get response reader')
    }

    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event: StreamEvent = JSON.parse(line.slice(6))
              yield event
            } catch (e) {
              console.warn('Failed to parse SSE event:', line)
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  }

  async submitFeedback(request: FeedbackRequest): Promise<FeedbackResponse> {
    const response: AxiosResponse<FeedbackResponse> = await this.client.post('/feedback', request)
    return response.data
  }

  async getHealth(): Promise<HealthResponse> {
    const response: AxiosResponse<HealthResponse> = await this.client.get('/health')
    return response.data
  }

  async getProtocols(): Promise<ProtocolsResponse> {
    const response: AxiosResponse<ProtocolsResponse> = await this.client.get('/protocols')
    return response.data
  }

  async getSources(): Promise<SourcesResponse> {
    const response: AxiosResponse<SourcesResponse> = await this.client.get('/sources')
    return response.data
  }
}

export const apiService = new ApiService()
```

### 5. Create Pinia Stores

Create `src/stores/query.ts`:
```typescript
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { apiService } from '@/services/api'
import type { QueryRequest, QueryResponse, DocumentSource } from '@/types/api'

export const useQueryStore = defineStore('query', () => {
  // State
  const currentQuery = ref('')
  const topK = ref(5)
  const useStreaming = ref(true)
  const isLoading = ref(false)
  const currentAnswer = ref('')
  const queryMetadata = ref<Partial<QueryResponse>>({})
  const sources = ref<DocumentSource[]>([])
  const error = ref<string | null>(null)
  
  // History
  const queryHistory = ref<{ query: string; answer: string; timestamp: number }[]>([])

  // Getters
  const hasCurrentQuery = computed(() => currentQuery.value.trim().length > 0)
  const hasAnswer = computed(() => currentAnswer.value.length > 0)

  // Actions
  async function executeQuery() {
    if (!hasCurrentQuery.value) return

    isLoading.value = true
    error.value = null
    currentAnswer.value = ''
    sources.value = []
    queryMetadata.value = {}

    const request: QueryRequest = {
      query: currentQuery.value,
      top_k: topK.value
    }

    try {
      if (useStreaming.value) {
        await executeStreamingQuery(request)
      } else {
        await executeBatchQuery(request)
      }
      
      // Add to history
      queryHistory.value.unshift({
        query: currentQuery.value,
        answer: currentAnswer.value,
        timestamp: Date.now()
      })
      
      // Keep only last 10 queries
      if (queryHistory.value.length > 10) {
        queryHistory.value = queryHistory.value.slice(0, 10)
      }
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'An error occurred'
    } finally {
      isLoading.value = false
    }
  }

  async function executeStreamingQuery(request: QueryRequest) {
    try {
      for await (const event of apiService.streamQuery(request)) {
        if (event.type === 'chunk' && event.content) {
          currentAnswer.value += event.content
        } else if (event.type === 'done') {
          queryMetadata.value = {
            model_used: event.model,
            documents_retrieved: event.documents_retrieved,
            cache_hit: event.cache_hit
          }
          sources.value = event.sources || []
        } else if (event.type === 'error') {
          throw new Error(event.message || 'Streaming error')
        }
      }
    } catch (err) {
      throw new Error(`Streaming failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function executeBatchQuery(request: QueryRequest) {
    const response = await apiService.query(request)
    currentAnswer.value = response.answer
    queryMetadata.value = response
    sources.value = response.sources
  }

  function clearQuery() {
    currentQuery.value = ''
    currentAnswer.value = ''
    sources.value = []
    queryMetadata.value = {}
    error.value = null
  }

  return {
    // State
    currentQuery,
    topK,
    useStreaming,
    isLoading,
    currentAnswer,
    queryMetadata,
    sources,
    error,
    queryHistory,
    
    // Getters
    hasCurrentQuery,
    hasAnswer,
    
    // Actions
    executeQuery,
    clearQuery
  }
})
```

### 7. Biome.js Configuration

Create `biome.json`:
```json
{
  "$schema": "https://biomejs.dev/schemas/1.4.1/schema.json",
  "organizeImports": {
    "enabled": true
  },
  "linter": {
    "enabled": true,
    "rules": {
      "recommended": true,
      "style": {
        "noNonNullAssertion": "off"
      },
      "suspicious": {
        "noExplicitAny": "warn"
      }
    }
  },
  "formatter": {
    "enabled": true,
    "indentStyle": "space",
    "indentWidth": 2,
    "lineWidth": 100
  },
  "javascript": {
    "formatter": {
      "quoteStyle": "single",
      "trailingComma": "es5",
      "semicolons": "asNeeded"
    }
  },
  "files": {
    "include": ["src/**/*", "*.vue", "*.ts", "*.js"],
    "ignore": ["node_modules", "dist", "build"]
  }
}
```

## Development Workflow

### 1. Start Development Server

```bash
bun run dev
```

Access the application at: `http://localhost:3000`

### 2. Available Commands

```bash
# Development
bun run dev              # Start dev server (with Bun runtime)
bun run build            # Build for production
bun run preview          # Preview production build

# Code Quality
bun run lint             # Biome lint check & fix
bun run format           # Biome format
bun run check            # Biome lint + format + organize imports
bun run type-check       # TypeScript check

# Testing
bun run test             # Unit tests (with Bun runtime)
bun run test:coverage    # Test coverage
bun run test:e2e         # E2E tests
```

### 3. Benefits of Bun + Biome.js

**Bun Runtime & Package Manager:**
- **25x faster** package installations than npm
- **Built-in bundler** - no need for separate build tools
- **Native TypeScript** - runs TS files directly
- **Hot reloading** - faster than Node.js for development
- **All-in-one** - runtime, package manager, bundler, test runner

**Biome.js Linter & Formatter:**
- **10-100x faster** than ESLint (written in Rust)
- **Zero configuration** - sensible defaults out of the box
- **Single tool** - replaces ESLint + Prettier + import sorting
- **Consistent** - same formatting across team and CI
- **Better error messages** - clearer, more actionable

### 4. Hot Module Replacement (HMR)

Vue 3 with Vite + Bun provides excellent HMR:
- Component updates preserve state
- CSS updates are instant  
- TypeScript compilation is faster with Bun
- Better developer experience overall

## Next Steps

1. **Run the setup commands** to initialize the project
2. **Test the development environment** with a simple component
3. **Begin component development** starting with the layout components
4. **Implement the query functionality** as the core feature
5. **Add feedback and system info tabs** progressively

## Troubleshooting

### Common Issues

1. **Bun installation issues**
   ```bash
   # Reinstall Bun
   curl -fsSL https://bun.sh/install | bash
   source ~/.bashrc  # or restart terminal
   ```

2. **Port already in use**
   ```bash
   # Change port in vite.config.ts
   server: { port: 3001 }
   ```

3. **TypeScript errors**
   ```bash
   bun run type-check  # Check for issues
   ```

4. **Tailwind styles not working**
   - Verify `main.css` imports
   - Check `tailwind.config.js` content paths

5. **Biome.js not formatting Vue files**
   - Ensure Vue files are included in `biome.json`
   - Check VS Code extension settings

### Performance Optimization

1. **Bundle analysis**
   ```bash
   bun run build -- --report
   ```

2. **Lazy loading components**
   ```typescript
   const QueryTab = defineAsyncComponent(() => import('@/views/QueryTab.vue'))
   ```

3. **Enable Vite PWA** (optional)
   ```bash
   bun add -D vite-plugin-pwa
   ```

### Editor Setup

**VS Code Extensions for Bun + Biome.js:**
```json
{
  "recommendations": [
    "biomejs.biome",
    "Vue.volar",
    "Vue.vscode-typescript-vue-plugin",
    "bradlc.vscode-tailwindcss"
  ]
}
```

**VS Code Settings:**
```json
{
  "editor.defaultFormatter": "biomejs.biome",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "quickfix.biome": true,
    "source.organizeImports.biome": true
  },
  "[vue]": {
    "editor.defaultFormatter": "biomejs.biome"
  }
}
```

This setup provides a solid foundation for the Vue.js 3 frontend with TypeScript, Pinia, and Tailwind CSS.