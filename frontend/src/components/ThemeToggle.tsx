import { useTheme } from "@/lib/theme"

// Sun/moon toggle. The icon cross-fades and rotates on switch for a bit of
// polish; the underlying `useTheme` hook persists the choice.
export function ThemeToggle({ className = "" }: { className?: string }) {
  const { theme, toggle } = useTheme()
  const dark = theme === "dark"

  return (
    <button
      type="button"
      onClick={toggle}
      title={dark ? "Светлая тема" : "Тёмная тема"}
      aria-label={dark ? "Включить светлую тему" : "Включить тёмную тему"}
      aria-pressed={dark}
      className={`relative flex h-9 w-9 items-center justify-center overflow-hidden rounded-md text-ink-3 transition-colors hover:bg-surface-container hover:text-primary ${className}`}
    >
      <span
        className={`material-symbols-outlined absolute text-[20px] transition-all duration-300 ${
          dark ? "rotate-0 opacity-100" : "-rotate-90 opacity-0"
        }`}
      >
        dark_mode
      </span>
      <span
        className={`material-symbols-outlined absolute text-[20px] transition-all duration-300 ${
          dark ? "rotate-90 opacity-0" : "rotate-0 opacity-100"
        }`}
      >
        light_mode
      </span>
    </button>
  )
}
