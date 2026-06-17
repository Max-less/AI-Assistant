import { useCallback, useEffect, useState } from "react"

export type Theme = "light" | "dark"

const STORAGE_KEY = "svod.theme"

function systemPrefersDark(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  )
}

function readStored(): Theme | null {
  const v = localStorage.getItem(STORAGE_KEY)
  return v === "dark" || v === "light" ? v : null
}

function apply(theme: Theme) {
  document.documentElement.classList.toggle("dark", theme === "dark")
  const meta = document.querySelector('meta[name="theme-color"]')
  if (meta) meta.setAttribute("content", theme === "dark" ? "#2e2e38" : "#264dd9")
}

// Light/dark theme with localStorage persistence. Defaults to the OS preference
// until the user makes an explicit choice; an inline script in index.html sets
// the initial class so there is no flash before this hook mounts.
export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() =>
    readStored() ?? (systemPrefersDark() ? "dark" : "light"),
  )

  useEffect(() => {
    apply(theme)
  }, [theme])

  // Follow OS changes only while the user hasn't pinned a preference.
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)")
    const onChange = (e: MediaQueryListEvent) => {
      if (!readStored()) setThemeState(e.matches ? "dark" : "light")
    }
    mq.addEventListener("change", onChange)
    return () => mq.removeEventListener("change", onChange)
  }, [])

  const setTheme = useCallback((next: Theme) => {
    localStorage.setItem(STORAGE_KEY, next)
    setThemeState(next)
  }, [])

  const toggle = useCallback(() => {
    setTheme(document.documentElement.classList.contains("dark") ? "light" : "dark")
  }, [setTheme])

  return { theme, setTheme, toggle }
}
