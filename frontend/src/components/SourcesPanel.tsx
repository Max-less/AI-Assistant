import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import type { Source } from "@/lib/api"

interface SourcesPanelProps {
  sources: Source[]
  onOpenSource: (source: Source) => void
}

// Lists the materials a message cited. Each row opens the source panel on the
// right with the exact passage and a link to the full PDF.
export function SourcesPanel({ sources, onOpenSource }: SourcesPanelProps) {
  if (!sources.length) return null

  return (
    <Collapsible className="mt-6 overflow-hidden rounded-lg border border-border-2 bg-bg">
      <CollapsibleTrigger className="group flex w-full items-center justify-between bg-surface-container-low p-3 font-mono-label-xs text-mono-label-xs uppercase tracking-wider text-ink-3 transition-colors hover:bg-surface-container">
        <span className="flex items-center gap-2">
          <span className="material-symbols-outlined text-[14px]">library_books</span>
          [ Использованные материалы: {sources.length} ]
        </span>
        <span className="material-symbols-outlined text-[16px] transition-transform duration-200 group-data-[state=open]:rotate-180">
          expand_more
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent className="flex flex-col gap-2 p-3">
        {sources.map((src, i) => (
          <button
            key={`${src.filename}-${i}`}
            type="button"
            onClick={() => onOpenSource(src)}
            className="flex items-center gap-3 rounded-md border border-hairline bg-surface p-3 text-left transition-colors hover:border-primary/40 hover:bg-surface-container"
          >
            <span className="font-ui-label text-ui-label font-bold text-accent">
              [{i + 1}]
            </span>
            <span className="flex-1 break-all font-body-secondary text-body-secondary text-ink">
              {src.filename}
            </span>
            <span className="material-symbols-outlined text-[16px] text-ink-4">
              chevron_right
            </span>
          </button>
        ))}
      </CollapsibleContent>
    </Collapsible>
  )
}
