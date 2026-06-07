import { useState } from "react"
import App from "./App"
import { AuthPage } from "@/components/AuthPage"
import { clearAuth, getSession, type AuthUser } from "@/lib/auth"

// Top-level gate: show the login/register screen until the user authenticates
// (or continues as guest). No router — just a localStorage-backed token.
export default function Root() {
  const [user, setUser] = useState<AuthUser | null>(() => getSession()?.user ?? null)

  function handleLogout() {
    clearAuth()
    setUser(null)
  }

  if (!user) {
    return <AuthPage onAuthed={setUser} />
  }

  return <App user={user} onLogout={handleLogout} onUserChange={setUser} />
}
