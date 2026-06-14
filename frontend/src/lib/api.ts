import { API_BASE } from "./config"
import { clearAuth, getToken, type AuthUser } from "./auth"

export type Role = "user" | "assistant"
export type FeedbackValue = -1 | 0 | 1

export type { AuthUser }

export interface TokenResponse {
  access_token: string
  token_type: "bearer"
  user: AuthUser
}

export interface Source {
  filename: string
  // The retrieved passage the citation was drawn from. Empty for messages
  // persisted before snippets were stored.
  snippet: string
}

export interface MessageOut {
  id: number
  role: Role
  content: string
  sources?: Source[] | null
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
  const token = getToken()
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(init?.headers ?? {}),
    },
  })
  if (!res.ok) {
    // An expired/invalid token on a protected route — drop it and return to login.
    // Auth endpoints themselves use 401 for "wrong password", so skip them.
    if (res.status === 401 && token && !path.startsWith("/api/auth/")) {
      clearAuth()
      window.location.reload()
    }
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

export function register(email: string, name: string, password: string): Promise<TokenResponse> {
  return jsonRequest<TokenResponse>("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, name, password }),
  })
}

export function login(email: string, password: string): Promise<TokenResponse> {
  return jsonRequest<TokenResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  })
}

export function loginAsGuest(clientId: string): Promise<TokenResponse> {
  return jsonRequest<TokenResponse>("/api/auth/guest", {
    method: "POST",
    body: JSON.stringify({ client_id: clientId }),
  })
}

export function getMe(): Promise<AuthUser> {
  return jsonRequest<AuthUser>("/api/auth/me")
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

export async function deleteSession(sessionId: number): Promise<void> {
  const token = getToken()
  const res = await fetch(`${API_BASE}/api/history/${sessionId}`, {
    method: "DELETE",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
  if (!res.ok) {
    if (res.status === 401 && token) {
      clearAuth()
      window.location.reload()
    }
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      // 204 has no body, and error bodies may be non-JSON — keep the status line
    }
    throw new Error(detail)
  }
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

interface DocumentsResponse {
  documents: string[]
}

export function getDocuments(): Promise<string[]> {
  return jsonRequest<DocumentsResponse>("/api/documents").then((r) => r.documents)
}

// Absolute URL to a knowledge-base PDF, suitable for opening in a new tab.
// The endpoint is public, so no auth header is needed for plain navigation.
export function documentUrl(filename: string): string {
  return `${API_BASE}/api/documents/${encodeURIComponent(filename)}`
}
