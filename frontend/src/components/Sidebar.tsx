import { Button } from "@/components/ui/button"

// Static sidebar — preserves the Stitch design. Conversation list, search and
// profile are visual only in this first version (chat state is not persisted).
const groups: { label: string; items: string[]; activeIndex?: number }[] = [
  {
    label: "Сегодня",
    items: ["Структура ТЗ для курсового проекта", "Рецензирование кода (GOST)"],
    activeIndex: 0,
  },
  {
    label: "Вчера",
    items: ["Жизненный цикл ПО", "Требования к оформлению схем"],
  },
  {
    label: "На этой неделе",
    items: ["Velocity и burndown", "Анализ предметной области"],
  },
]

export function Sidebar() {
  return (
    <nav className="z-20 hidden h-screen w-[280px] flex-col border-r border-border-2 bg-surface-container-low p-4 md:fixed md:left-0 md:top-0 md:flex">
      <div className="mb-6 flex items-center gap-2 px-2">
        <span className="font-hero-heading text-hero-heading text-primary">Свод</span>
        <span className="rounded-sm bg-surface-variant px-1.5 py-0.5 font-mono-label-xs text-mono-label-xs text-ink-3">
          rag·assistant
        </span>
      </div>

      <Button variant="outline" className="mb-6 w-full justify-start">
        <span className="material-symbols-outlined text-[18px]">add</span>
        <span>Новая беседа</span>
      </Button>

      <div className="relative mb-6">
        <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-[18px] text-ink-3">
          search
        </span>
        <input
          type="text"
          placeholder="Поиск по беседам..."
          className="w-full rounded-lg border-none bg-surface-container-highest py-2 pl-9 pr-3 font-body-secondary text-body-secondary text-ink placeholder:text-ink-4 focus:outline-none focus:ring-1 focus:ring-accent"
        />
      </div>

      <div className="custom-scrollbar flex-1 overflow-y-auto pr-2">
        {groups.map((group) => (
          <div key={group.label} className="mb-6">
            <h3 className="mb-2 px-2 font-ui-label text-ui-label uppercase tracking-wider text-ink-3">
              {group.label}
            </h3>
            <ul className="space-y-1">
              {group.items.map((item, i) => {
                const active = group.activeIndex === i
                return (
                  <li key={item}>
                    <button
                      className={
                        active
                          ? "w-full truncate rounded-r-md border-l-2 border-accent bg-accent-bg px-3 py-2 text-left font-ui-label text-ui-label text-accent"
                          : "w-full truncate rounded-md border-l-2 border-transparent px-3 py-2 text-left font-ui-label text-ui-label text-ink-3 transition-colors duration-200 hover:bg-surface-container-high hover:text-ink-2"
                      }
                    >
                      {item}
                    </button>
                  </li>
                )
              })}
            </ul>
          </div>
        ))}
      </div>

      <div className="mt-auto flex items-center justify-between border-t border-border-2 pt-4">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center overflow-hidden rounded-full border border-hairline bg-surface-container-high">
            <span className="material-symbols-outlined text-[18px] text-ink-3">person</span>
          </div>
          <div className="flex flex-col">
            <span className="font-ui-label text-ui-label text-ink">Иван Студент</span>
            <span className="font-mono-label-xs text-mono-label-xs text-ink-4">РТФ, 4 курс</span>
          </div>
        </div>
        <button className="rounded-md p-1.5 text-ink-3 transition-colors hover:bg-surface-container-high hover:text-ink">
          <span className="material-symbols-outlined text-[18px]">settings</span>
        </button>
      </div>
    </nav>
  )
}
