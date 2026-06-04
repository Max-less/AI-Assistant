import { useEffect, useRef } from "react"
import type { FeedbackValue } from "@/lib/api"
import type { ChatMessage } from "@/types"
import { AssistantMessage } from "./AssistantMessage"

function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex max-w-[85%] flex-col gap-2 self-end md:max-w-[70%]">
      <div className="flex items-center justify-end gap-2 px-1">
        <span className="font-ui-label text-ui-label text-ink-3">Вы</span>
      </div>
      <div className="whitespace-pre-wrap rounded-xl rounded-tr-sm border border-hairline bg-surface-container px-5 py-4 font-body-primary text-body-primary text-ink shadow-sm">
        {content}
      </div>
    </div>
  )
}

function Thinking() {
  return (
    <div className="flex flex-col gap-2 self-start">
      <div className="flex items-center gap-2 px-1">
        <div className="flex h-6 w-6 items-center justify-center rounded-md border border-primary/20 bg-primary/10">
          <span className="material-symbols-outlined text-[14px] text-primary">auto_awesome</span>
        </div>
        <span className="font-ui-label text-ui-label text-ink">Свод</span>
      </div>
      <div className="flex items-center gap-2 rounded-xl rounded-tl-sm border border-border-2 bg-surface px-5 py-4 text-ink-3 shadow-sm">
        <span className="h-2 w-2 animate-bounce rounded-full bg-ink-4 [animation-delay:-0.3s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-ink-4 [animation-delay:-0.15s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-ink-4" />
        <span className="ml-1 font-mono-telemetry text-mono-telemetry">ищу в базе знаний…</span>
      </div>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-3 py-20 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-primary/20 bg-primary/10">
        <span className="material-symbols-outlined text-[24px] text-primary">auto_awesome</span>
      </div>
      <h2 className="font-hero-heading text-hero-heading text-ink">Чем помочь?</h2>
      <p className="max-w-md font-body-primary text-body-primary text-ink-3">
        Задайте вопрос по методичкам кафедры — Scrum, ГОСТ, оформление ТЗ,
        жизненный цикл ПО. Ответы строятся на базе знаний с ссылками на источники.
      </p>
    </div>
  )
}

interface MessageListProps {
  messages: ChatMessage[]
  loading: boolean
  onFeedback: (messageId: number, value: FeedbackValue) => void
}

export function MessageList({ messages, loading, onFeedback }: MessageListProps) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading])

  return (
    <div className="custom-scrollbar flex flex-1 flex-col items-center overflow-y-auto p-4 pb-40 md:p-8">
      <div className="flex w-full max-w-[820px] flex-col gap-8">
        {messages.length === 0 && !loading && <EmptyState />}
        {messages.map((m) =>
          m.role === "user" ? (
            <UserMessage key={m.id} content={m.content} />
          ) : (
            <AssistantMessage key={m.id} message={m} onFeedback={onFeedback} />
          ),
        )}
        {loading && <Thinking />}
        <div ref={endRef} />
      </div>
    </div>
  )
}
