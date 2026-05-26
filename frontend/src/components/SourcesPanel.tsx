import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"

// Show only the file name from the full path the API returns.
function basename(path: string): string {
  const parts = path.split(/[\\/]/)
  return parts[parts.length - 1] || path
}

// Honest rendering: the API returns only source file paths (no scores or
// snippets), so we list the file names and nothing we don't actually have.
export function SourcesPanel({ sources }: { sources: string[] }) {
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
      <CollapsibleContent className="flex flex-col gap-3 p-3">
        {sources.map((src, i) => (
          <div
            key={`${src}-${i}`}
            className="flex flex-col gap-2 rounded-md border border-hairline bg-surface p-3"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-ui-label text-ui-label font-bold text-ink">
                [{i + 1}] {basename(src)}
              </span>
            </div>
            <p className="border-l-2 border-hairline pl-3 font-mono-telemetry text-mono-telemetry text-ink-4 break-all">
              {src}
            </p>
          </div>
        ))}
      </CollapsibleContent>
    </Collapsible>
  )
}
