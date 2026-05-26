import ReactMarkdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"
import type { ChatMessage } from "@/types"
import { SourcesPanel } from "./SourcesPanel"

// Markdown element styling matched to the Stitch mockup's typography/tables.
const markdownComponents: Components = {
  p: ({ children }) => (
    <p className="font-body-primary text-body-primary leading-relaxed">{children}</p>
  ),
  h3: ({ children }) => (
    <h3 className="mb-2 mt-6 font-markdown-h3 text-markdown-h3 text-ink">{children}</h3>
  ),
  h2: ({ children }) => (
    <h2 className="mb-2 mt-6 font-markdown-h3 text-markdown-h3 text-ink">{children}</h2>
  ),
  ul: ({ children }) => <ul className="ml-1 list-disc space-y-2 pl-4">{children}</ul>,
  ol: ({ children }) => <ol className="ml-1 list-decimal space-y-2 pl-4">{children}</ol>,
  li: ({ children }) => (
    <li className="font-body-primary text-body-primary leading-relaxed">{children}</li>
  ),
  strong: ({ children }) => <strong className="font-semibold text-ink">{children}</strong>,
  a: ({ children, href }) => (
    <a href={href} className="text-primary underline" target="_blank" rel="noreferrer">
      {children}
    </a>
  ),
  code: ({ children }) => (
    <code className="rounded bg-surface-container px-1 py-0.5 font-mono-telemetry text-[12px] text-ink-2">
      {children}
    </code>
  ),
  table: ({ children }) => (
    <div className="my-4 overflow-x-auto rounded-lg border border-hairline">
      <table className="w-full border-collapse text-left">{children}</table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="bg-surface-container-low font-ui-label text-ui-label text-ink-2">
      {children}
    </thead>
  ),
  tbody: ({ children }) => (
    <tbody className="divide-y divide-hairline font-body-secondary text-body-secondary text-ink">
      {children}
    </tbody>
  ),
  tr: ({ children }) => <tr className="transition-colors hover:bg-bg">{children}</tr>,
  th: ({ children }) => <th className="border-b border-hairline p-3">{children}</th>,
  td: ({ children }) => <td className="p-3 text-ink-2 align-top">{children}</td>,
}

function Telemetry({ message }: { message: ChatMessage }) {
  if (message.latencyMs == null) return null
  const bd = message.latencyBreakdown ?? {}
  const parts = Object.entries(bd).map(([k, v]) => `${k} ${v}ms`)
  return (
    <div className="mt-4 flex flex-wrap items-center gap-2 font-mono-telemetry text-mono-telemetry text-ink-4">
      <span className="material-symbols-outlined text-[12px]">speed</span>
      <span>всего {message.latencyMs}ms</span>
      {parts.length > 0 && <span className="text-ink-4">· {parts.join(" · ")}</span>}
    </div>
  )
}

export function AssistantMessage({ message }: { message: ChatMessage }) {
  return (
    <div className="flex w-full max-w-[95%] flex-col gap-2 self-start md:max-w-[85%]">
      <div className="flex items-center gap-2 px-1">
        <div className="flex h-6 w-6 items-center justify-center rounded-md border border-primary/20 bg-primary/10">
          <span className="material-symbols-outlined text-[14px] text-primary">auto_awesome</span>
        </div>
        <span className="font-ui-label text-ui-label text-ink">Свод</span>
        {!message.error && (
          <div className="flex items-center gap-1 rounded border border-success-fg/20 bg-success-bg px-1.5 py-0.5 text-success-fg">
            <span className="material-symbols-outlined text-[12px]">verified</span>
            <span className="font-mono-label-xs text-mono-label-xs">RAG</span>
          </div>
        )}
      </div>

      <div
        className={
          message.error
            ? "space-y-4 rounded-xl rounded-tl-sm border border-error/30 bg-error-container px-5 py-5 text-on-error-container shadow-sm"
            : "space-y-4 rounded-xl rounded-tl-sm border border-border-2 bg-surface px-5 py-5 text-ink shadow-sm"
        }
      >
        {message.error ? (
          <p className="flex items-start gap-2 font-body-primary text-body-primary leading-relaxed">
            <span className="material-symbols-outlined mt-0.5 text-[18px]">error</span>
            <span>{message.content}</span>
          </p>
        ) : (
          <div className="space-y-4">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {!message.error && <SourcesPanel sources={message.sources ?? []} />}
        {!message.error && <Telemetry message={message} />}
      </div>
    </div>
  )
}
