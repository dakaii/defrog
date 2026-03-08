# DeFrog Frontend

Vue 3 + TypeScript + Pinia + Tailwind CSS frontend for the DeFrog DeFi RAG system.

## Setup

```bash
bun install
# or: npm install
```

## Development

```bash
bun run dev
# or: npm run dev
```

Runs at http://localhost:3000. Ensure the FastAPI backend is running at http://localhost:8000.

## Environment

Copy `.env.example` to `.env` and configure:

- `VITE_API_URL` - Backend API URL (default: http://localhost:8000)
- `VITE_API_KEY` - Optional API key for protected endpoints

## Build

```bash
bun run build
# or: npm run build
```

Output in `dist/`.
