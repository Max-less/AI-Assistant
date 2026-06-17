import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// PDF text extraction (pypdf) often emits one word per line, so a raw passage
// renders as a broken vertical stack. Rejoin soft line breaks into running
// prose while preserving real paragraph breaks (blank lines). Hyphenated words
// split across a line break ("раз-\nработка") are stitched back together.
export function normalizeSnippet(raw: string): string {
  return raw
    .replace(/\r\n?/g, "\n")
    .replace(/(\w)-\n(\w)/g, "$1$2") // de-hyphenate words broken across lines
    .replace(/[ \t]*\n{2,}[ \t]*/g, "\n\n") // collapse blank-line runs to one break
    .replace(/[ \t]*\n[ \t]*/g, " ") // soft line break -> space
    .replace(/[ \t]{2,}/g, " ")
    .trim()
}
