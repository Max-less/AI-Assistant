// Auth state persisted in localStorage: the JWT access token plus the current
// user profile. The token is attached as a Bearer header by lib/api.ts.

export interface AuthUser {
  id: number
  email: string | null
  name: string | null
  is_guest: boolean
  created_at: string
  guest_remaining: number | null
}

export interface Session {
  token: string
  user: AuthUser
}

const AUTH_STORAGE_KEY = "svod.auth"
// Kept separate from the token and deliberately NOT cleared on logout, so a guest
// can't reset their query quota just by signing out and back in.
const GUEST_ID_KEY = "svod.guest_id"

export function getOrCreateGuestId(): string {
  let id = localStorage.getItem(GUEST_ID_KEY)
  if (!id) {
    id =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `g-${Date.now()}-${Math.random().toString(36).slice(2)}`
    localStorage.setItem(GUEST_ID_KEY, id)
  }
  return id
}

export function getSession(): Session | null {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<Session>
    if (typeof parsed.token === "string" && parsed.user) {
      return { token: parsed.token, user: parsed.user as AuthUser }
    }
    return null
  } catch {
    return null
  }
}

export function getToken(): string | null {
  return getSession()?.token ?? null
}

export function setSession(token: string, user: AuthUser): void {
  localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify({ token, user }))
}

export function clearAuth(): void {
  localStorage.removeItem(AUTH_STORAGE_KEY)
}
