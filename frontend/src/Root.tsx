import { useState } from "react"
import App from "./App"
import { AuthPage } from "@/components/AuthPage"
import { clearAuth, getAuth } from "@/lib/auth"

// Top-level gate: show the login/register screen until the user authenticates
// (or continues as guest). No router — just a localStorage-backed flag.
export default function Root() {
  const [authed, setAuthed] = useState(() => getAuth() !== null)

  function handleLogout() {
    clearAuth()
    setAuthed(false)
  }

  if (!authed) {
    return <AuthPage onAuthed={() => setAuthed(true)} />
  }

  return <App onLogout={handleLogout} />
}
