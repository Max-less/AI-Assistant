// Base URL of the RAG FastAPI service. Override with VITE_API_BASE if needed.
export const API_BASE: string =
  import.meta.env.VITE_API_BASE ?? "http://localhost:8000"
