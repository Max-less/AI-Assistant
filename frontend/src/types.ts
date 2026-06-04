export interface ChatMessage {
  id: string
  // Server-assigned id (present for messages persisted by the web backend;
  // missing for the optimistic user echo and for error placeholders).
  messageId?: number
  role: "user" | "assistant"
  content: string
  // assistant-only metadata
  sources?: string[]
  latencyMs?: number
  latencyBreakdown?: Record<string, number>
  feedback?: -1 | 1 | null
  error?: boolean
}
