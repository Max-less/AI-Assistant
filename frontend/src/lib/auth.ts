// Frontend-only auth stub. No real backend yet — we only persist a flag in
// localStorage so the chat is gated behind the login/register screen on first visit.
// Replace with real tokens (JWT) when the backend auth lands.

export type AuthMode = "user" | "guest"

export interface AuthState {
  mode: AuthMode
}

const AUTH_STORAGE_KEY = "svod.auth"

export function getAuth(): AuthState | null {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<AuthState>
    if (parsed.mode === "user" || parsed.mode === "guest") {
      return { mode: parsed.mode }
    }
    return null
  } catch {
    return null
  }
}

export function setAuth(mode: AuthMode): void {
  localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify({ mode }))
}

export function clearAuth(): void {
  localStorage.removeItem(AUTH_STORAGE_KEY)
}
