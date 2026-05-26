import { API_BASE } from "./config"

export type Role = "user" | "assistant" | "system"

export interface HistoryItem {
  role: Role
  content: string
}

export interface AskRequest {
  question: string
  history?: HistoryItem[]
  top_k?: number | null
  use_expander?: boolean | null
  use_reranker?: boolean | null
}

export interface AskResponse {
  answer: string
  sources: string[]
  latency_ms: number
  latency_breakdown: Record<string, number>
}

export interface HealthResponse {
  status: "ok" | "loading"
  index_loaded: boolean
  chunk_count: number | null
}

/** POST /ask — send a question plus prior turns, get an answer with sources. */
export async function askQuestion(
  question: string,
  history: HistoryItem[],
): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history } satisfies AskRequest),
  })

  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      // non-JSON error body — keep the status line
    }
    throw new Error(detail)
  }

  return (await res.json()) as AskResponse
}

/** GET /health — service readiness and indexed chunk count. */
export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return (await res.json()) as HealthResponse
}
