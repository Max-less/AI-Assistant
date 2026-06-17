import { useEffect, useRef, useState } from "react"

// Progressively reveals `text` like a typewriter when `enabled` is true, giving
// a non-streaming answer the feel of streaming. Speed scales with length so
// long answers don't drag and short ones don't crawl. `onTick` fires as the
// revealed text grows (used to keep the view scrolled to the bottom).
export function useReveal(
  text: string,
  enabled: boolean,
  onTick?: () => void,
): { shown: string; done: boolean } {
  const [count, setCount] = useState(enabled ? 0 : text.length)
  const onTickRef = useRef(onTick)
  onTickRef.current = onTick

  useEffect(() => {
    if (!enabled) {
      setCount(text.length)
      return
    }
    const total = text.length
    // Finish in ~2.5s for long answers; never slower than 80 chars/sec.
    const cps = Math.max(80, total / 2.5)
    let raf = 0
    let start: number | null = null
    const tick = (ts: number) => {
      if (start === null) start = ts
      const n = Math.min(total, Math.floor(((ts - start) / 1000) * cps))
      setCount(n)
      onTickRef.current?.()
      if (n < total) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
    // Re-run only if the answer text itself changes.
  }, [text, enabled])

  const safeCount = Math.min(count, text.length)
  return { shown: text.slice(0, safeCount), done: safeCount >= text.length }
}
