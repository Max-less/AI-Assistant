import { useCallback, useEffect, useState } from "react"
import { Composer } from "@/components/Composer"
import { KnowledgeBasePanel } from "@/components/KnowledgeBasePanel"
import { MessageList } from "@/components/MessageList"
import { Sidebar } from "@/components/Sidebar"
import { SourceDrawer } from "@/components/SourceDrawer"
import { TopBar } from "@/components/TopBar"
import {
  type FeedbackValue,
  type MessageOut,
  type SessionSummary,
  type Source,
  deleteSession,
  getHistory,
  getMe,
  getSessions,
  postChat,
  postFeedback,
} from "@/lib/api"
import { getSession, setSession, type AuthUser } from "@/lib/auth"
import type { ChatMessage } from "@/types"

const SESSION_STORAGE_KEY = "svod.lastSessionId"

function newId() {
  return Math.random().toString(36).slice(2)
}

function loadStoredSessionId(): number | null {
  const raw = localStorage.getItem(SESSION_STORAGE_KEY)
  if (!raw) return null
  const n = Number(raw)
  return Number.isFinite(n) && n > 0 ? n : null
}

function fromServerMessage(m: MessageOut): ChatMessage {
  return {
    id: newId(),
    messageId: m.id,
    role: m.role,
    content: m.content,
    sources: m.sources ?? undefined,
    latencyMs: m.latency_ms ?? undefined,
    latencyBreakdown: m.latency_breakdown ?? undefined,
    feedback: m.feedback ?? null,
  }
}

export default function App({
  user,
  onLogout,
  onUserChange,
}: {
  user: AuthUser
  onLogout?: () => void
  onUserChange?: (user: AuthUser) => void
}) {
  const [sessionId, setSessionId] = useState<number | null>(null)
  const [sessionTitle, setSessionTitle] = useState<string | null>(null)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [composerHeight, setComposerHeight] = useState(160)
  // Right-side panels: a single cited source, and the full knowledge base.
  const [activeSource, setActiveSource] = useState<Source | null>(null)
  const [kbOpen, setKbOpen] = useState(false)
  // Mobile sidebar drawer (ignored at md+ where the rail is always visible).
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const refreshSessions = useCallback(async () => {
    try {
      const list = await getSessions()
      setSessions(list)
      return list
    } catch {
      return [] as SessionSummary[]
    }
  }, [])

  // Initial load: pull sessions and try restoring the last open one.
  useEffect(() => {
    let cancelled = false
    void (async () => {
      const list = await refreshSessions()
      if (cancelled) return
      const stored = loadStoredSessionId()
      if (stored && list.some((s) => s.id === stored)) {
        try {
          const h = await getHistory(stored)
          if (cancelled) return
          setSessionId(h.session.id)
          setSessionTitle(h.session.title)
          setMessages(h.messages.map(fromServerMessage))
        } catch {
          // session vanished server-side — start fresh
          localStorage.removeItem(SESSION_STORAGE_KEY)
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [refreshSessions])

  const handleSelectSession = useCallback(async (id: number) => {
    try {
      const h = await getHistory(id)
      setSessionId(h.session.id)
      setSessionTitle(h.session.title)
      setMessages(h.messages.map(fromServerMessage))
      localStorage.setItem(SESSION_STORAGE_KEY, String(h.session.id))
    } catch (e) {
      console.error(e)
    }
  }, [])

  const handleNewChat = useCallback(() => {
    setSessionId(null)
    setSessionTitle(null)
    setMessages([])
    localStorage.removeItem(SESSION_STORAGE_KEY)
  }, [])

  const handleDeleteSession = useCallback(
    async (id: number) => {
      await deleteSession(id)
      // If the deleted chat is the one on screen, fall back to a blank state.
      setSessionId((current) => {
        if (current === id) {
          setSessionTitle(null)
          setMessages([])
          localStorage.removeItem(SESSION_STORAGE_KEY)
          return null
        }
        return current
      })
      await refreshSessions()
    },
    [refreshSessions],
  )

  const handleSend = useCallback(
    async (text: string) => {
      const wasNewSession = sessionId === null
      const userMsg: ChatMessage = { id: newId(), role: "user", content: text }
      setMessages((prev) => [...prev, userMsg])
      setLoading(true)

      try {
        const res = await postChat(text, sessionId)
        setSessionId(res.session_id)
        localStorage.setItem(SESSION_STORAGE_KEY, String(res.session_id))
        setMessages((prev) => [...prev, { ...fromServerMessage(res.message), reveal: true }])
        if (wasNewSession) {
          const list = await refreshSessions()
          const created = list.find((s) => s.id === res.session_id)
          if (created) setSessionTitle(created.title)
        } else {
          // keep sessions list fresh (message_count changes)
          void refreshSessions()
        }
      } catch (e) {
        const detail = e instanceof Error ? e.message : "Неизвестная ошибка"
        setMessages((prev) => [
          ...prev,
          {
            id: newId(),
            role: "assistant",
            content: `Не удалось получить ответ: ${detail}.`,
            error: true,
          },
        ])
      } finally {
        setLoading(false)
        // Keep the guest quota indicator fresh (it changes per query).
        if (user.is_guest) {
          try {
            const me = await getMe()
            onUserChange?.(me)
            const s = getSession()
            if (s) setSession(s.token, me)
          } catch {
            /* ignore */
          }
        }
      }
    },
    [refreshSessions, sessionId, user.is_guest, onUserChange],
  )

  const handleFeedback = useCallback(
    async (messageId: number, value: FeedbackValue) => {
      // optimistic
      setMessages((prev) =>
        prev.map((m) =>
          m.messageId === messageId
            ? { ...m, feedback: value === 0 ? null : (value as -1 | 1) }
            : m,
        ),
      )
      try {
        await postFeedback(messageId, value)
      } catch (e) {
        console.error("feedback failed", e)
        // best-effort: refetch the current session to resync feedback state
        if (sessionId !== null) {
          try {
            const h = await getHistory(sessionId)
            setMessages(h.messages.map(fromServerMessage))
          } catch {
            /* ignore */
          }
        }
      }
    },
    [sessionId],
  )

  return (
    <div className="flex h-dvh w-screen overflow-hidden bg-bg">
      <Sidebar
        sessions={sessions}
        activeId={sessionId}
        onSelect={handleSelectSession}
        onNewChat={handleNewChat}
        onDelete={handleDeleteSession}
        onLogout={onLogout}
        user={user}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <main className="relative flex h-dvh min-w-0 flex-1 flex-col bg-bg-tint md:ml-[280px]">
        <TopBar
          title={sessionTitle ?? "Новая беседа"}
          onOpenKnowledgeBase={() => setKbOpen(true)}
          onOpenSidebar={() => setSidebarOpen(true)}
        />
        <MessageList
          messages={messages}
          loading={loading}
          onFeedback={handleFeedback}
          onOpenSource={setActiveSource}
          bottomInset={composerHeight}
        />
        <Composer onSend={handleSend} disabled={loading} onHeightChange={setComposerHeight} />
      </main>

      <SourceDrawer source={activeSource} onClose={() => setActiveSource(null)} />
      <KnowledgeBasePanel open={kbOpen} onClose={() => setKbOpen(false)} />
    </div>
  )
}
