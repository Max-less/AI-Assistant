export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  // assistant-only metadata
  sources?: string[]
  latencyMs?: number
  latencyBreakdown?: Record<string, number>
  error?: boolean
}
