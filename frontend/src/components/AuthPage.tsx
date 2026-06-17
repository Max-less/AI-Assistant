import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ThemeToggle } from "@/components/ThemeToggle"
import { cn } from "@/lib/utils"
import { getOrCreateGuestId, setSession, type AuthUser } from "@/lib/auth"
import { login, loginAsGuest, register } from "@/lib/api"

interface AuthPageProps {
  onAuthed: (user: AuthUser) => void
}

type Tab = "login" | "register"

const FEATURES = [
  { icon: "forum", title: "Контекст диалога", desc: "Помнит ход беседы и уточняющие вопросы" },
  { icon: "library_books", title: "База знаний", desc: "Ответы по методичкам кафедры с источниками" },
  { icon: "thumb_up", title: "Обратная связь", desc: "Оценивайте ответы — система становится точнее" },
]

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

// Single bordered field with a Material Symbols icon prefix. Mirrors the Composer
// input shell (border-2 + focus-within:border-primary).
function Field({
  icon,
  invalid,
  children,
}: {
  icon: string
  invalid?: boolean
  children: React.ReactNode
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-lg border-2 bg-surface px-3 py-2.5 transition-colors focus-within:border-primary/50",
        invalid ? "border-error" : "border-border-2",
      )}
    >
      <span className="material-symbols-outlined text-[20px] text-ink-4">{icon}</span>
      {children}
    </div>
  )
}

export function AuthPage({ onAuthed }: AuthPageProps) {
  const [tab, setTab] = useState<Tab>("login")
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirm, setConfirm] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const isRegister = tab === "register"

  function switchTab(next: Tab) {
    setTab(next)
    setError(null)
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (loading) return
    // Client-side validation before hitting the server.
    if (isRegister && !name.trim()) {
      setError("Введите имя")
      return
    }
    if (!EMAIL_RE.test(email.trim())) {
      setError("Введите корректный email")
      return
    }
    if (password.length < 6) {
      setError("Пароль должен быть не короче 6 символов")
      return
    }
    if (isRegister && password !== confirm) {
      setError("Пароли не совпадают")
      return
    }

    setError(null)
    setLoading(true)
    try {
      const res = isRegister
        ? await register(email.trim(), name.trim(), password)
        : await login(email.trim(), password)
      setSession(res.access_token, res.user)
      onAuthed(res.user)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось выполнить вход")
    } finally {
      setLoading(false)
    }
  }

  async function handleGuest() {
    if (loading) return
    setError(null)
    setLoading(true)
    try {
      const res = await loginAsGuest(getOrCreateGuestId())
      setSession(res.access_token, res.user)
      onAuthed(res.user)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось войти как гость")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="relative flex h-dvh w-screen overflow-hidden bg-bg">
      <div className="absolute right-3 top-3 z-20">
        <ThemeToggle />
      </div>
      {/* Left — brand showcase */}
      <aside className="relative hidden w-[46%] flex-col justify-between overflow-hidden bg-primary p-12 text-on-primary md:flex">
        {/* Decorative gradient blobs */}
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-primary via-secondary to-accent opacity-90" />
        <div className="pointer-events-none absolute -left-24 -top-24 h-72 w-72 rounded-full bg-primary-fixed-dim/30 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-32 -right-16 h-80 w-80 rounded-full bg-accent/40 blur-3xl" />

        <div className="relative flex items-center gap-2">
          <span className="font-hero-heading text-hero-heading">Свод</span>
          <span className="rounded-sm bg-white/15 px-1.5 py-0.5 font-mono-label-xs text-mono-label-xs">
            rag·assistant
          </span>
        </div>

        <div className="relative max-w-md">
          <h1 className="font-display-cover text-display-cover leading-[1.05]">
            Интеллектуальный ассистент по проектному управлению
          </h1>
          <p className="mt-4 font-body-primary text-body-primary text-on-primary/80">
            Agile, Scrum, DevOps и техническое задание — ответы по выверенной базе знаний кафедры,
            со ссылками на источники.
          </p>
        </div>

        <ul className="relative space-y-4">
          {FEATURES.map((f) => (
            <li key={f.icon} className="flex items-start gap-3">
              <span className="flex h-9 w-9 flex-none items-center justify-center rounded-lg bg-white/15">
                <span className="material-symbols-outlined text-[20px]">{f.icon}</span>
              </span>
              <div>
                <p className="font-ui-label text-ui-label">{f.title}</p>
                <p className="font-body-secondary text-body-secondary text-on-primary/75">{f.desc}</p>
              </div>
            </li>
          ))}
        </ul>
      </aside>

      {/* Right — auth card */}
      <main className="flex flex-1 items-center justify-center px-6 py-10">
        <div className="w-full max-w-[400px]">
          {/* Compact brand for mobile (left panel is hidden) */}
          <div className="mb-8 flex items-center justify-center gap-2 md:hidden">
            <span className="font-hero-heading text-hero-heading text-primary">Свод</span>
            <span className="rounded-sm bg-surface-variant px-1.5 py-0.5 font-mono-label-xs text-mono-label-xs text-ink-3">
              rag·assistant
            </span>
          </div>

          <div className="rounded-[14px] border-2 border-border-2 bg-surface p-6 shadow-[0_8px_24px_oklch(0.4_0.02_250/0.06)]">
            <h2 className="font-hero-heading text-hero-heading text-ink">
              {isRegister ? "Создать аккаунт" : "С возвращением"}
            </h2>
            <p className="mt-1 font-body-secondary text-body-secondary text-ink-3">
              {isRegister
                ? "Зарегистрируйтесь, чтобы сохранять историю бесед"
                : "Войдите, чтобы продолжить работу со «Сводом»"}
            </p>

            {/* Segmented tab switcher */}
            <div className="mt-6 grid grid-cols-2 gap-1 rounded-lg bg-surface-container-high p-1">
              {(["login", "register"] as const).map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => switchTab(t)}
                  className={cn(
                    "rounded-md py-1.5 font-ui-label text-ui-label transition-colors",
                    tab === t
                      ? "bg-surface text-ink shadow-sm"
                      : "text-ink-3 hover:text-ink-2",
                  )}
                >
                  {t === "login" ? "Вход" : "Регистрация"}
                </button>
              ))}
            </div>

            <form className="mt-5 space-y-3" onSubmit={handleSubmit} noValidate>
              {isRegister && (
                <Field icon="person">
                  <Input
                    type="text"
                    autoComplete="name"
                    placeholder="Имя"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                  />
                </Field>
              )}
              <Field icon="mail">
                <Input
                  type="email"
                  autoComplete="email"
                  placeholder="Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </Field>
              <Field icon="lock">
                <Input
                  type="password"
                  autoComplete={isRegister ? "new-password" : "current-password"}
                  placeholder="Пароль"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </Field>
              {isRegister && (
                <Field icon="lock_reset">
                  <Input
                    type="password"
                    autoComplete="new-password"
                    placeholder="Повторите пароль"
                    value={confirm}
                    onChange={(e) => setConfirm(e.target.value)}
                  />
                </Field>
              )}

              {error && (
                <p className="flex items-center gap-1.5 font-body-secondary text-body-secondary text-error">
                  <span className="material-symbols-outlined text-[16px]">error</span>
                  {error}
                </p>
              )}

              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? "Подождите…" : isRegister ? "Создать аккаунт" : "Войти"}
              </Button>
            </form>

            {/* Divider + guest */}
            <div className="my-5 flex items-center gap-3">
              <div className="h-px flex-1 bg-border" />
              <span className="font-mono-label-xs text-mono-label-xs uppercase tracking-wider text-ink-4">
                или
              </span>
              <div className="h-px flex-1 bg-border" />
            </div>

            <Button variant="ghost" className="w-full" onClick={handleGuest} disabled={loading}>
              <span className="material-symbols-outlined text-[18px]">person_outline</span>
              Продолжить как гость
            </Button>
          </div>

          <p className="mt-4 text-center font-body-secondary text-body-secondary text-ink-4">
            Продолжая, вы соглашаетесь с условиями использования и политикой конфиденциальности.
          </p>
        </div>
      </main>
    </div>
  )
}
