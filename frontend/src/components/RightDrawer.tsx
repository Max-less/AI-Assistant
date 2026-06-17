import { useEffect, useRef, useState, type ReactNode } from "react"

interface RightDrawerProps {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
}

// Slide-over panel anchored to the right edge, with a click-away backdrop.
// Shared by the citation source view and the knowledge-base list. Stays mounted
// through the close transition so it slides out instead of snapping away.
export function RightDrawer({ open, onClose, title, children }: RightDrawerProps) {
  const [mounted, setMounted] = useState(open)
  // Drives the enter/leave transform: false = off-screen right, true = in view.
  const [shown, setShown] = useState(false)
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (open) {
      if (closeTimer.current) clearTimeout(closeTimer.current)
      setMounted(true)
      // Next frame so the browser applies the off-screen start state first.
      requestAnimationFrame(() => requestAnimationFrame(() => setShown(true)))
    } else if (mounted) {
      setShown(false)
      closeTimer.current = setTimeout(() => setMounted(false), 300)
    }
    return () => {
      if (closeTimer.current) clearTimeout(closeTimer.current)
    }
  }, [open, mounted])

  // Close on Escape while open.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose()
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [open, onClose])

  if (!mounted) return null

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div
        className={`absolute inset-0 bg-black/30 transition-opacity duration-300 ${
          shown ? "opacity-100" : "opacity-0"
        }`}
        onClick={onClose}
        aria-hidden="true"
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={`relative z-10 flex h-screen w-full max-w-[440px] flex-col border-l border-border-2 bg-surface shadow-xl transition-transform duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] ${
          shown ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <header className="flex items-center justify-between border-b border-hairline px-5 py-4">
          <h2 className="font-markdown-h3 text-markdown-h3 text-ink">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Закрыть"
            className="rounded-md p-1 text-ink-3 transition-colors hover:bg-surface-container hover:text-ink"
          >
            <span className="material-symbols-outlined">close</span>
          </button>
        </header>
        <div className="custom-scrollbar flex-1 overflow-y-auto p-5">{children}</div>
      </aside>
    </div>
  )
}
