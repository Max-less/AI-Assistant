import { useEffect, useState } from "react"
import { documentUrl, getDocuments } from "@/lib/api"
import { RightDrawer } from "./RightDrawer"

interface KnowledgeBasePanelProps {
  open: boolean
  onClose: () => void
}

// Lists every PDF the assistant answers from; each opens in a new browser tab.
export function KnowledgeBasePanel({ open, onClose }: KnowledgeBasePanelProps) {
  const [docs, setDocs] = useState<string[] | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    if (!open) return
    let cancelled = false
    setDocs(null)
    setError(false)
    getDocuments()
      .then((list) => !cancelled && setDocs(list))
      .catch(() => !cancelled && setError(true))
    return () => {
      cancelled = true
    }
  }, [open])

  return (
    <RightDrawer open={open} onClose={onClose} title="База знаний">
      <p className="mb-4 font-body-secondary text-body-secondary text-ink-3">
        Документы, на основе которых ассистент строит ответы. Нажмите, чтобы открыть PDF.
      </p>

      {error && (
        <p className="font-body-secondary text-body-secondary text-error">
          Не удалось загрузить список документов.
        </p>
      )}

      {!error && docs === null && (
        <p className="font-body-secondary text-body-secondary text-ink-3">Загрузка…</p>
      )}

      {!error && docs?.length === 0 && (
        <p className="font-body-secondary text-body-secondary text-ink-3">
          В базе знаний пока нет документов.
        </p>
      )}

      <ul className="flex flex-col gap-2">
        {docs?.map((name) => (
          <li key={name}>
            <button
              type="button"
              onClick={() =>
                window.open(documentUrl(name), "_blank", "noopener,noreferrer")
              }
              className="flex w-full items-center gap-3 rounded-md border border-hairline bg-surface p-3 text-left transition-colors hover:border-primary/40 hover:bg-surface-container"
            >
              <span className="material-symbols-outlined text-[20px] text-error">
                picture_as_pdf
              </span>
              <span className="flex-1 break-all font-body-secondary text-body-secondary text-ink">
                {name}
              </span>
              <span className="material-symbols-outlined text-[16px] text-ink-4">
                open_in_new
              </span>
            </button>
          </li>
        ))}
      </ul>
    </RightDrawer>
  )
}
