import { useMemo } from "react"
import { Button } from "@/components/ui/button"
import type { SessionSummary } from "@/lib/api"
import type { AuthUser } from "@/lib/auth"

interface SidebarProps {
  sessions: SessionSummary[]
  activeId: number | null
  onSelect: (id: number) => void
  onNewChat: () => void
  onLogout?: () => void
  user: AuthUser
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

export function Sidebar({ sessions, activeId, onSelect, onNewChat, onLogout, user }: SidebarProps) {
  const groups = useMemo(() => groupSessions(sessions), [sessions])

  const primaryName = user.is_guest ? "Гость" : user.name || user.email || "Аккаунт"
  const secondaryLine = user.is_guest
    ? `Осталось ${user.guest_remaining ?? 0} запросов`
    : user.email || "Зарегистрирован"

  return (
    <nav className="z-20 hidden h-screen w-[280px] flex-col border-r border-border-2 bg-surface-container-low p-4 md:fixed md:left-0 md:top-0 md:flex">
      <div className="mb-6 flex items-center gap-2 px-2">
        <span className="font-hero-heading text-hero-heading text-primary">Свод</span>
        <span className="rounded-sm bg-surface-variant px-1.5 py-0.5 font-mono-label-xs text-mono-label-xs text-ink-3">
          rag·assistant
        </span>
      </div>

      <Button variant="outline" className="mb-6 w-full justify-start" onClick={onNewChat}>
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
                    <li key={s.id}>
                      <button
                        onClick={() => onSelect(s.id)}
                        title={s.title}
                        className={
                          active
                            ? "w-full truncate rounded-r-md border-l-2 border-accent bg-accent-bg px-3 py-2 text-left font-ui-label text-ui-label text-accent"
                            : "w-full truncate rounded-md border-l-2 border-transparent px-3 py-2 text-left font-ui-label text-ui-label text-ink-3 transition-colors duration-200 hover:bg-surface-container-high hover:text-ink-2"
                        }
                      >
                        {s.title}
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
  )
}
