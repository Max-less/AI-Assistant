import { API_BASE } from "./config"

export type Role = "user" | "assistant"
export type FeedbackValue = -1 | 0 | 1

export interface MessageOut {
  id: number
  role: Role
  content: string
  sources?: string[] | null
  latency_ms?: number | null
  latency_breakdown?: Record<string, number> | null
  created_at: string
  feedback?: -1 | 1 | null
}

export interface ChatRequest {
  question: string
  session_id: number | null
}

export interface ChatResponse {
  session_id: number
  message: MessageOut
}

export interface SessionSummary {
  id: number
  title: string
  created_at: string
  message_count: number
}

export interface SessionDetail {
  id: number
  title: string
  created_at: string
}

export interface HistoryResponse {
  session: SessionDetail
  messages: MessageOut[]
}

export interface HealthResponse {
  web: "ok"
  rag: "ok" | "loading" | "down"
  chunk_count: number | null
}

async function jsonRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
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
  return (await res.json()) as T
}

export function postChat(question: string, sessionId: number | null): Promise<ChatResponse> {
  return jsonRequest<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify({ question, session_id: sessionId } satisfies ChatRequest),
  })
}

export function getSessions(): Promise<SessionSummary[]> {
  return jsonRequest<SessionSummary[]>("/api/sessions")
}

export function getHistory(sessionId: number): Promise<HistoryResponse> {
  return jsonRequest<HistoryResponse>(`/api/history/${sessionId}`)
}

export function postFeedback(messageId: number, value: FeedbackValue): Promise<{ ok: boolean }> {
  return jsonRequest<{ ok: boolean }>("/api/feedback", {
    method: "POST",
    body: JSON.stringify({ message_id: messageId, value }),
  })
}

export function getHealth(): Promise<HealthResponse> {
  return jsonRequest<HealthResponse>("/api/health")
}
