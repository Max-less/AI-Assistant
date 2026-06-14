import type { ReactNode } from "react"

interface RightDrawerProps {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
}

// Slide-over panel anchored to the right edge, with a click-away backdrop.
// Shared by the citation source view and the knowledge-base list.
export function RightDrawer({ open, onClose, title, children }: RightDrawerProps) {
  if (!open) return null

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div
        className="absolute inset-0 bg-black/30"
        onClick={onClose}
        aria-hidden="true"
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="relative z-10 flex h-screen w-full max-w-[440px] flex-col border-l border-border-2 bg-surface shadow-xl"
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
