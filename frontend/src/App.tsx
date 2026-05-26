import { useState } from "react"
import { Composer } from "@/components/Composer"
import { MessageList } from "@/components/MessageList"
import { Sidebar } from "@/components/Sidebar"
import { TopBar } from "@/components/TopBar"
import { askQuestion, type HistoryItem } from "@/lib/api"
import type { ChatMessage } from "@/types"

const CHAT_TITLE = "Структура ТЗ для курсового проекта"

function newId() {
  return Math.random().toString(36).slice(2)
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)

  async function handleSend(text: string) {
    const userMsg: ChatMessage = { id: newId(), role: "user", content: text }

    // History = the conversation so far (before this turn), mapped to the API shape.
    const history: HistoryItem[] = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }))

    setMessages((prev) => [...prev, userMsg])
    setLoading(true)

    try {
      const res = await askQuestion(text, history)
      setMessages((prev) => [
        ...prev,
        {
          id: newId(),
          role: "assistant",
          content: res.answer,
          sources: res.sources,
          latencyMs: res.latency_ms,
          latencyBreakdown: res.latency_breakdown,
        },
      ])
    } catch (e) {
      const detail = e instanceof Error ? e.message : "Неизвестная ошибка"
      setMessages((prev) => [
        ...prev,
        {
          id: newId(),
          role: "assistant",
          content: `Не удалось получить ответ: ${detail}. Проверьте, что RAG-сервис запущен на localhost:8000.`,
          error: true,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg">
      <Sidebar />
      <main className="relative flex h-screen flex-1 flex-col bg-bg-tint md:ml-[280px]">
        <TopBar title={CHAT_TITLE} />
        <MessageList messages={messages} loading={loading} />
        <Composer onSend={handleSend} disabled={loading} />
      </main>
    </div>
  )
}
