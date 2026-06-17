import { documentUrl, type Source } from "@/lib/api"
import { normalizeSnippet } from "@/lib/utils"
import { RightDrawer } from "./RightDrawer"

interface SourceDrawerProps {
  source: Source | null
  onClose: () => void
}

// Shows the passage a citation was drawn from, plus a link to open the full PDF.
export function SourceDrawer({ source, onClose }: SourceDrawerProps) {
  return (
    <RightDrawer open={source !== null} onClose={onClose} title="Источник">
      {source && (
        <div className="flex flex-col gap-4">
          <div className="flex items-start gap-2">
            <span className="material-symbols-outlined text-[20px] text-error">
              picture_as_pdf
            </span>
            <span className="font-ui-label text-ui-label text-ink break-all">
              {source.filename}
            </span>
          </div>

          <button
            type="button"
            onClick={() =>
              window.open(documentUrl(source.filename), "_blank", "noopener,noreferrer")
            }
            className="flex items-center justify-center gap-2 rounded-md bg-primary px-4 py-2 font-ui-label text-ui-label text-on-primary shadow-sm transition-colors hover:bg-primary/90"
          >
            <span className="material-symbols-outlined text-[18px]">open_in_new</span>
            Открыть PDF
          </button>

          {source.snippet ? (
            <div className="flex flex-col gap-2">
              <span className="font-mono-label-xs text-mono-label-xs uppercase tracking-wider text-ink-4">
                Фрагмент из документа
              </span>
              <blockquote className="whitespace-pre-wrap border-l-2 border-accent bg-bg p-3 font-body-secondary text-body-secondary leading-relaxed text-ink-2">
                {normalizeSnippet(source.snippet)}
              </blockquote>
            </div>
          ) : (
            <p className="font-body-secondary text-body-secondary text-ink-3">
              Для этого ответа фрагмент не сохранён (старое сообщение). Откройте PDF,
              чтобы посмотреть документ целиком.
            </p>
          )}
        </div>
      )}
    </RightDrawer>
  )
}
