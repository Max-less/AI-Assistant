import { useEffect, useState } from "react"
import { getHealth, type HealthResponse } from "@/lib/api"
import { ThemeToggle } from "./ThemeToggle"

// Top bar — title from current session; status indicator wired to /api/health.
// Visible on every breakpoint: on mobile it also hosts the menu (hamburger)
// button that opens the sidebar drawer.
export function TopBar({
  title,
  onOpenKnowledgeBase,
  onOpenSidebar,
}: {
  title: string
  onOpenKnowledgeBase: () => void
  onOpenSidebar: () => void
}) {
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
    <header className="sticky top-0 z-10 flex w-full items-center justify-between gap-2 border-b border-hairline bg-surface/80 px-4 py-3 backdrop-blur-md md:px-6 md:py-4">
      <div className="flex min-w-0 items-center gap-2 md:gap-4">
        <button
          type="button"
          onClick={onOpenSidebar}
          aria-label="Меню"
          className="-ml-1 rounded-md p-2 text-ink-3 transition-colors hover:bg-surface-container hover:text-ink md:hidden"
        >
          <span className="material-symbols-outlined">menu</span>
        </button>
        <h1 className="min-w-0 truncate font-markdown-h3 text-markdown-h3 text-ink">{title}</h1>
        <div className="hidden items-center gap-1.5 rounded-md bg-surface-container px-2 py-1 md:flex">
          <div className={`h-1.5 w-1.5 rounded-full ${dotClass}`} />
          <span className="font-mono-telemetry text-mono-telemetry text-ink-3">{statusText}</span>
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-1">
        <button
          type="button"
          onClick={onOpenKnowledgeBase}
          title="База знаний"
          aria-label="База знаний"
          className="rounded-md p-2 text-ink-3 transition-colors hover:bg-surface-container hover:text-primary"
        >
          <span className="material-symbols-outlined">source</span>
        </button>
        <ThemeToggle />
      </div>
    </header>
  )
}
