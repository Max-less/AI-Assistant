import { useEffect, useState } from "react"
import { getHealth, type HealthResponse } from "@/lib/api"

// Top bar — title from current session; status indicator wired to /api/health.
export function TopBar({ title }: { title: string }) {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    let cancelled = false
    getHealth()
      .then((h) => !cancelled && setHealth(h))
      .catch(() => !cancelled && setError(true))
    return () => {
      cancelled = true
    }
  }, [])

  let dotClass: string
  let statusText: string
  if (error || health?.rag === "down") {
    dotClass = "bg-error"
    statusText = "База знаний: нет соединения"
  } else if (health?.rag === "ok") {
    dotClass = "bg-success-fg"
    statusText = `База знаний: ${health.chunk_count ?? 0} фрагментов`
  } else {
    dotClass = "bg-warning-fg animate-pulse"
    statusText = "База знаний: загрузка…"
  }

  return (
    <header
      className="sticky top-0 z-10 hidden w-full items-center justify-between bg-surface/80 px-6 py-4 backdrop-blur-md md:flex"
      style={{ borderBottom: "1px solid oklch(0.94 0.005 250)" }}
    >
      <div className="flex items-center gap-4">
        <h1 className="font-markdown-h3 text-markdown-h3 text-ink">{title}</h1>
        <div className="flex items-center gap-1.5 rounded-md bg-surface-container px-2 py-1">
          <div className={`h-1.5 w-1.5 rounded-full ${dotClass}`} />
          <span className="font-mono-telemetry text-mono-telemetry text-ink-3">{statusText}</span>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <button className="rounded-md p-2 text-ink-3 transition-colors hover:bg-surface-container hover:text-primary">
          <span className="material-symbols-outlined">history</span>
        </button>
        <button className="rounded-md p-2 text-ink-3 transition-colors hover:bg-surface-container hover:text-primary">
          <span className="material-symbols-outlined">source</span>
        </button>
      </div>
    </header>
  )
}
