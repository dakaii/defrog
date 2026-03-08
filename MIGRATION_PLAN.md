# DeFrog Migration Plan: Streamlit to Vue.js 3 + Pinia + Tailwind CSS

## Overview

This document outlines the comprehensive migration plan from the current Streamlit dashboard to a modern Vue.js 3 application with Pinia state management and Tailwind CSS styling.

## Current State Analysis

### Streamlit App Structure (`dashboard/app.py`)
- **File size**: 294 lines
- **Dependencies**: streamlit, requests, python-dotenv, plotly, pandas, numpy
- **Core features**:
  - 3 main tabs: Query, Feedback, System Info
  - SSE streaming support for real-time query responses
  - API integration with FastAPI backend
  - Basic UI components (forms, buttons, expandables)

### Current Features Breakdown

#### Tab 1: Query Interface
- Text area for DeFi questions
- Example questions in expandable section
- Advanced options (top_k slider, streaming toggle)
- Real-time streaming or batch query modes
- Answer display with metadata (model, latency, cache status)
- Source documents with similarity scores
- Query/answer persistence for feedback

#### Tab 2: Feedback System
- Rating slider (1-5 stars)
- Optional comment text area
- Displays last query/answer for context
- Submits feedback to `/feedback` endpoint

#### Tab 3: System Information
- Health check status (system, PostgreSQL, OpenAI)
- Available protocols list
- Document sources with enable/disable status
- Usage instructions

#### Sidebar
- About section with tech stack info
- Refresh button

## Migration Strategy

### Phase 1: Project Setup & Foundation
**Timeline**: 1-2 days (faster with Bun)
**Effort**: Low-Medium

1. **Create Vue.js project structure**
   ```
   frontend/
   ├── src/
   │   ├── components/
   │   ├── views/
   │   ├── stores/
   │   ├── services/
   │   ├── composables/
   │   └── assets/
   ├── public/
   ├── package.json
   ├── vite.config.ts
   ├── tailwind.config.js
   └── biome.json
   ```

2. **Install and configure dependencies with Bun**
   - Vue.js 3 (Composition API)
   - Pinia for state management
   - Tailwind CSS v4.2
   - Vue Router for navigation
   - Axios for HTTP requests
   - TypeScript for type safety

3. **Setup development environment**
   - Bun runtime and package manager (25x faster installs)
   - Biome.js for linting and formatting (single tool, 100x faster)
   - Vite build tool with Bun integration
   - Environment variables handling

### Phase 2: Core Infrastructure
**Timeline**: 3-4 days
**Effort**: High

1. **API Service Layer**
   - Create `services/api.ts` for all backend communication
   - Implement SSE streaming with EventSource
   - Type definitions for all API responses
   - Error handling and retry logic

2. **State Management (Pinia)**
   - Query store for managing search state
   - Feedback store for rating system
   - System store for health/protocol data
   - UI store for loading states and notifications

3. **Routing Setup**
   - Main dashboard route
   - Nested routes for tabs if needed
   - Route guards for future authentication

### Phase 3: Component Development
**Timeline**: 5-7 days
**Effort**: High

#### Core Layout Components
1. **AppLayout.vue** - Main application shell
2. **AppHeader.vue** - Top navigation/branding
3. **AppSidebar.vue** - Side panel with about info
4. **TabNavigation.vue** - Tab switching interface

#### Query Tab Components
1. **QueryForm.vue** - Question input and options
2. **ExampleQuestions.vue** - Expandable examples list
3. **AdvancedOptions.vue** - Settings panel
4. **StreamingAnswer.vue** - Real-time response display
5. **QueryMetadata.vue** - Model, latency, cache info
6. **SourceDocuments.vue** - Document sources list

#### Feedback Tab Components
1. **FeedbackForm.vue** - Rating and comment input
2. **QueryContext.vue** - Previous Q&A display
3. **FeedbackSuccess.vue** - Submission confirmation

#### System Info Tab Components
1. **SystemHealth.vue** - Health status indicators
2. **ProtocolsList.vue** - Available protocols grid
3. **DocumentSources.vue** - Sources management
4. **UsageInstructions.vue** - Help documentation

#### Shared/Utility Components
1. **LoadingSpinner.vue** - Loading states
2. **ErrorMessage.vue** - Error display
3. **MetricCard.vue** - Stat display cards
4. **ExpandableSection.vue** - Collapsible content

### Phase 4: Advanced Features
**Timeline**: 3-4 days
**Effort**: Medium

1. **Enhanced Streaming**
   - WebSocket fallback for better reliability
   - Pause/resume streaming
   - Token-by-token highlighting

2. **Improved UX**
   - Auto-save draft queries
   - Query history with local storage
   - Keyboard shortcuts
   - Dark/light mode toggle

3. **Performance Optimizations**
   - Virtual scrolling for long responses
   - Image lazy loading
   - Component lazy loading
   - API response caching

### Phase 5: Testing & Deployment
**Timeline**: 2-3 days
**Effort**: Medium

1. **Testing Setup**
   - Unit tests with Vitest
   - Component tests with Vue Test Utils
   - E2E tests with Playwright
   - API integration tests

2. **Production Build**
   - Docker configuration
   - Environment-specific configs
   - Static asset optimization
   - PWA features (optional)

## Technology Stack Comparison

### Current (Streamlit)
- **Language**: Python
- **Framework**: Streamlit
- **State**: Server-side session state
- **Styling**: Limited built-in themes
- **Deployment**: Python server required

### Proposed (Vue.js)
- **Language**: TypeScript
- **Framework**: Vue.js 3 (Composition API)
- **Runtime**: Bun (faster than Node.js)
- **State**: Pinia (client-side)
- **Styling**: Tailwind CSS v4.2
- **Tooling**: Biome.js (linting + formatting)
- **Deployment**: Static files (CDN ready)

## Benefits of Migration

### User Experience
- **Faster interactions** - No server round trips for UI updates
- **Better mobile support** - Responsive design with Tailwind
- **Offline capabilities** - PWA features for basic functionality
- **Modern UI/UX** - Professional, polished interface

### Developer Experience
- **Better maintainability** - Component-based architecture
- **Type safety** - TypeScript prevents runtime errors
- **Faster tooling** - Bun runtime is 25x faster than npm, Biome.js is 100x faster than ESLint
- **Hot reloading** - Instant development feedback with Vite + Bun
- **Unified tooling** - Single Biome.js tool for linting, formatting, and import sorting
- **Extensive ecosystem** - Rich Vue.js plugin ecosystem

### Performance
- **Client-side rendering** - Faster initial loads after cache
- **Code splitting** - Load only necessary components
- **Better caching** - Static assets can be CDN cached
- **Real-time features** - WebSocket support for advanced streaming

## Implementation Priority

### High Priority (MVP)
1. Query interface with streaming
2. Basic feedback system
3. System health display
4. Source documents view

### Medium Priority (v1.1)
1. Advanced options panel
2. Query history
3. Enhanced error handling
4. Mobile responsive design

### Low Priority (Future)
1. Dark mode
2. PWA features
3. Advanced analytics
4. User authentication

## File Structure

```
frontend/
├── public/
│   ├── index.html
│   └── favicon.ico
├── src/
│   ├── components/
│   │   ├── layout/
│   │   │   ├── AppLayout.vue
│   │   │   ├── AppHeader.vue
│   │   │   └── AppSidebar.vue
│   │   ├── query/
│   │   │   ├── QueryForm.vue
│   │   │   ├── StreamingAnswer.vue
│   │   │   ├── QueryMetadata.vue
│   │   │   └── SourceDocuments.vue
│   │   ├── feedback/
│   │   │   ├── FeedbackForm.vue
│   │   │   └── QueryContext.vue
│   │   ├── system/
│   │   │   ├── SystemHealth.vue
│   │   │   ├── ProtocolsList.vue
│   │   │   └── DocumentSources.vue
│   │   └── shared/
│   │       ├── LoadingSpinner.vue
│   │       ├── ErrorMessage.vue
│   │       └── MetricCard.vue
│   ├── views/
│   │   ├── Dashboard.vue
│   │   ├── QueryTab.vue
│   │   ├── FeedbackTab.vue
│   │   └── SystemTab.vue
│   ├── stores/
│   │   ├── query.ts
│   │   ├── feedback.ts
│   │   ├── system.ts
│   │   └── ui.ts
│   ├── services/
│   │   ├── api.ts
│   │   ├── streaming.ts
│   │   └── types.ts
│   ├── composables/
│   │   ├── useStreaming.ts
│   │   ├── useLocalStorage.ts
│   │   └── useDebounce.ts
│   ├── assets/
│   │   ├── styles/
│   │   │   ├── main.css
│   │   │   └── components.css
│   │   └── images/
│   ├── App.vue
│   └── main.ts
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
└── README.md
```

## Risk Assessment

### High Risk
- **SSE streaming complexity** - Requires careful implementation
- **State synchronization** - Multiple tabs sharing data
- **Browser compatibility** - EventSource support

### Medium Risk
- **Performance with large responses** - Virtual scrolling needed
- **TypeScript learning curve** - Team familiarity required
- **Build process complexity** - Vite configuration

### Low Risk
- **Basic UI components** - Well-established patterns
- **API integration** - Straightforward HTTP calls
- **Tailwind CSS** - Proven utility framework

## Success Metrics

### Performance
- **Initial load time** < 2 seconds
- **Time to interactive** < 3 seconds
- **Streaming latency** < 100ms additional overhead

### User Experience
- **Mobile responsiveness** - 100% feature parity
- **Accessibility** - WCAG 2.1 AA compliance
- **Cross-browser** - Support for modern browsers

### Development
- **Bundle size** < 500KB gzipped
- **Test coverage** > 80%
- **Build time** < 30 seconds

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| 1. Setup | 1-2 days | Project structure, dependencies (faster with Bun) |
| 2. Infrastructure | 3-4 days | API layer, state management, routing |
| 3. Components | 5-7 days | All UI components and views |
| 4. Features | 3-4 days | Advanced features, UX improvements |
| 5. Testing | 2-3 days | Testing, deployment, documentation |

**Total Estimated Time: 14-20 days** (1-2 days faster with modern tooling)

## Next Steps

1. **Approve migration plan** - Get stakeholder buy-in
2. **Setup development environment** - Install Node.js, create project
3. **Create feature branch** - `feature/vue-migration`
4. **Begin Phase 1** - Project setup and configuration
5. **Regular progress reviews** - Daily standups during development

## Additional Considerations

### Backwards Compatibility
- Keep Streamlit app running during migration
- Gradual rollout with feature flags
- Easy rollback strategy

### Team Training
- Vue.js 3 Composition API workshop
- TypeScript best practices session
- Tailwind CSS utility classes training

### Documentation
- Component usage guidelines
- State management patterns
- API integration examples
- Deployment procedures

---

*This migration plan is a living document and should be updated as the project progresses.*