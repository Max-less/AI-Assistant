import { useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import type { SessionSummary } from "@/lib/api"
import type { AuthUser } from "@/lib/auth"

interface SidebarProps {
  sessions: SessionSummary[]
  activeId: number | null
  onSelect: (id: number) => void
  onNewChat: () => void
  onDelete: (id: number) => void | Promise<void>
  onLogout?: () => void
  user: AuthUser
  // Mobile drawer state. On md+ the sidebar is always-on and these are ignored.
  open: boolean
  onClose: () => void
}

interface SessionGroup {
  label: string
  items: SessionSummary[]
}

// Bucket sessions by relative date (today / yesterday / this week / earlier),
// matching the look of the original Stitch mockup.
function groupSessions(sessions: SessionSummary[]): SessionGroup[] {
  const now = new Date()
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const startOfYesterday = startOfToday - 24 * 60 * 60 * 1000
  const startOfWeek = startOfToday - 7 * 24 * 60 * 60 * 1000

  const buckets: Record<string, SessionSummary[]> = {
    today: [],
    yesterday: [],
    week: [],
    earlier: [],
  }
  for (const s of sessions) {
    const t = new Date(s.created_at).getTime()
    if (t >= startOfToday) buckets.today.push(s)
    else if (t >= startOfYesterday) buckets.yesterday.push(s)
    else if (t >= startOfWeek) buckets.week.push(s)
    else buckets.earlier.push(s)
  }
  return [
    { label: "Сегодня", items: buckets.today },
    { label: "Вчера", items: buckets.yesterday },
    { label: "На этой неделе", items: buckets.week },
    { label: "Ранее", items: buckets.earlier },
  ].filter((g) => g.items.length > 0)
}

export function Sidebar({
  sessions,
  activeId,
  onSelect,
  onNewChat,
  onDelete,
  onLogout,
  user,
  open,
  onClose,
}: SidebarProps) {
  const groups = useMemo(() => groupSessions(sessions), [sessions])
  // On mobile, picking a chat or starting a new one should dismiss the drawer.
  const selectAndClose = (id: number) => {
    onSelect(id)
    onClose()
  }
  const newChatAndClose = () => {
    onNewChat()
    onClose()
  }
  // Session pending deletion, awaiting confirmation in the modal.
  const [pendingDelete, setPendingDelete] = useState<SessionSummary | null>(null)
  const [deleting, setDeleting] = useState(false)

  async function confirmDelete() {
    if (!pendingDelete) return
    setDeleting(true)
    try {
      await onDelete(pendingDelete.id)
      setPendingDelete(null)
    } finally {
      setDeleting(false)
    }
  }

  const primaryName = user.is_guest ? "Гость" : user.name || user.email || "Аккаунт"
  const secondaryLine = user.is_guest
    ? `Осталось ${user.guest_remaining ?? 0} запросов`
    : user.email || "Зарегистрирован"

  return (
    <>
    {/* Backdrop — mobile only, shown while the drawer is open. */}
    <div
      onClick={onClose}
      aria-hidden="true"
      className={`fixed inset-0 z-30 bg-black/40 transition-opacity duration-300 md:hidden ${
        open ? "opacity-100" : "pointer-events-none opacity-0"
      }`}
    />
    <nav
      className={`fixed left-0 top-0 z-40 flex h-dvh w-[280px] flex-col border-r border-border-2 bg-surface-container-low p-4 transition-transform duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] md:z-20 md:translate-x-0 ${
        open ? "translate-x-0 shadow-2xl md:shadow-none" : "-translate-x-full"
      }`}
    >
      <div className="mb-6 flex items-center gap-2 px-2">
        <span className="font-hero-heading text-hero-heading text-primary">Свод</span>
        <span className="rounded-sm bg-surface-variant px-1.5 py-0.5 font-mono-label-xs text-mono-label-xs text-ink-3">
          rag·assistant
        </span>
        <button
          type="button"
          onClick={onClose}
          aria-label="Закрыть меню"
          className="ml-auto rounded-md p-1 text-ink-3 transition-colors hover:bg-surface-container-high hover:text-ink md:hidden"
        >
          <span className="material-symbols-outlined text-[20px]">close</span>
        </button>
      </div>

      <Button variant="outline" className="mb-6 w-full justify-start" onClick={newChatAndClose}>
        <span className="material-symbols-outlined text-[18px]">add</span>
        <span>Новая беседа</span>
      </Button>

      <div className="custom-scrollbar flex-1 overflow-y-auto pr-2">
        {groups.length === 0 ? (
          <p className="px-2 font-body-secondary text-body-secondary text-ink-4">
            Пока нет бесед. Задай первый вопрос — он сохранится здесь.
          </p>
        ) : (
          groups.map((group) => (
            <div key={group.label} className="mb-6">
              <h3 className="mb-2 px-2 font-ui-label text-ui-label uppercase tracking-wider text-ink-3">
                {group.label}
              </h3>
              <ul className="space-y-1">
                {group.items.map((s) => {
                  const active = s.id === activeId
                  return (
                    <li key={s.id} className="group/item relative">
                      <button
                        onClick={() => selectAndClose(s.id)}
                        title={s.title}
                        className={
                          active
                            ? "w-full truncate rounded-r-md border-l-2 border-accent bg-accent-bg py-2 pl-3 pr-9 text-left font-ui-label text-ui-label text-accent"
                            : "w-full truncate rounded-md border-l-2 border-transparent py-2 pl-3 pr-9 text-left font-ui-label text-ui-label text-ink-3 transition-colors duration-200 hover:bg-surface-container-high hover:text-ink-2"
                        }
                      >
                        {s.title}
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          setPendingDelete(s)
                        }}
                        title="Удалить беседу"
                        aria-label="Удалить беседу"
                        className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded-md p-1 text-ink-4 opacity-0 transition-colors duration-200 hover:bg-surface-container-high hover:text-error focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent group-hover/item:opacity-100"
                      >
                        <span className="material-symbols-outlined text-[18px]">delete</span>
                      </button>
                    </li>
                  )
                })}
              </ul>
            </div>
          ))
        )}
      </div>

      <div className="mt-auto flex items-center justify-between border-t border-border-2 pt-4">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center overflow-hidden rounded-full border border-hairline bg-surface-container-high">
            <span className="material-symbols-outlined text-[18px] text-ink-3">
              {user.is_guest ? "person_outline" : "person"}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="font-ui-label text-ui-label text-ink">{primaryName}</span>
            <span className="font-mono-label-xs text-mono-label-xs text-ink-4">{secondaryLine}</span>
          </div>
        </div>
        <button
          onClick={onLogout}
          title="Выйти"
          className="rounded-md p-1.5 text-ink-3 transition-colors hover:bg-surface-container-high hover:text-ink"
        >
          <span className="material-symbols-outlined text-[18px]">logout</span>
        </button>
      </div>
    </nav>

    {pendingDelete && (
      <div
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
        onClick={() => !deleting && setPendingDelete(null)}
      >
        <div
          role="alertdialog"
          aria-modal="true"
          aria-labelledby="delete-dialog-title"
          className="w-full max-w-sm rounded-xl border border-hairline bg-surface p-6 shadow-lg"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="mb-4 flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-error-container">
              <span className="material-symbols-outlined text-[20px] text-error">delete</span>
            </div>
            <div className="flex flex-col gap-1">
              <h2 id="delete-dialog-title" className="font-ui-label text-ui-label text-ink">
                Удалить беседу?
              </h2>
              <p className="font-body-secondary text-body-secondary text-ink-3">
                «{pendingDelete.title}» и все её сообщения будут удалены без возможности
                восстановления.
              </p>
            </div>
          </div>
          <div className="flex justify-end gap-2">
            <Button
              variant="outline"
              onClick={() => setPendingDelete(null)}
              disabled={deleting}
            >
              Отмена
            </Button>
            <Button
              onClick={confirmDelete}
              disabled={deleting}
              className="bg-error text-on-error shadow-sm hover:bg-error/90"
            >
              {deleting ? "Удаление…" : "Удалить"}
            </Button>
          </div>
        </div>
      </div>
    )}
    </>
  )
}
