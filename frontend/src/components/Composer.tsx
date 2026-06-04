import { useEffect, useRef, useState } from "react"
import { Textarea } from "@/components/ui/textarea"

export function Composer({
  onSend,
  disabled,
  onHeightChange,
}: {
  onSend: (text: string) => void
  disabled: boolean
  onHeightChange?: (height: number) => void
}) {
  const [value, setValue] = useState("")
  const taRef = useRef<HTMLTextAreaElement>(null)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!onHeightChange || !rootRef.current) return
    const el = rootRef.current
    const ro = new ResizeObserver((entries) => {
      const h = entries[0]?.contentRect.height
      if (typeof h === "number") onHeightChange(h)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [onHeightChange])

  function autoGrow() {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = ""
    ta.style.height = `${ta.scrollHeight}px`
  }

  function submit() {
    const text = value.trim()
    if (!text || disabled) return
    onSend(text)
    setValue("")
    if (taRef.current) taRef.current.style.height = ""
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div
      ref={rootRef}
      className="pointer-events-none absolute bottom-0 left-0 z-10 flex w-full justify-center bg-gradient-to-t from-bg-tint via-bg-tint to-transparent px-4 pb-6 pt-10 md:px-8"
    >
      <div className="pointer-events-auto flex w-full max-w-[820px] flex-col rounded-[14px] border-2 border-border-2 bg-surface p-2 shadow-[0_8px_24px_oklch(0.4_0.02_250/0.04)] transition-all focus-within:border-primary/50">
        <div className="flex items-center gap-2 px-2 pb-1 pt-1">
          <span className="flex items-center gap-1 rounded bg-surface-container-high px-2 py-0.5 font-mono-telemetry text-mono-telemetry text-ink-3">
            <span className="material-symbols-outlined text-[12px]">model_training</span>
            GigaChat
          </span>
          <div className="h-3 w-px bg-hairline" />
          <span className="font-mono-telemetry text-mono-telemetry text-ink-4">
            База: Проектная деятельность
          </span>
        </div>
        <div className="relative flex items-end gap-2">
          <Textarea
            ref={taRef}
            rows={1}
            value={value}
            disabled={disabled}
            placeholder="Задайте вопрос по методичкам кафедры..."
            className="max-h-[150px] custom-scrollbar"
            onChange={(e) => {
              setValue(e.target.value)
              autoGrow()
            }}
            onKeyDown={onKeyDown}
          />
          <div className="flex gap-1 pb-1 pr-1">
            <button
              type="button"
              className="rounded-lg p-2 text-ink-4 transition-colors hover:bg-surface-container hover:text-ink-2"
              title="Прикрепить файл (недоступно)"
            >
              <span className="material-symbols-outlined text-[20px]">attach_file</span>
            </button>
            <button
              type="button"
              onClick={submit}
              disabled={disabled || !value.trim()}
              className="flex items-center justify-center rounded-lg bg-primary p-2 text-on-primary shadow-sm transition-colors hover:bg-surface-tint disabled:cursor-not-allowed disabled:opacity-50"
            >
              <span className="material-symbols-outlined text-[20px]">
                {disabled ? "hourglass_empty" : "send"}
              </span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
