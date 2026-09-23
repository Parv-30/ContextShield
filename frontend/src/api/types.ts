export interface MessageLabel {
  category: string
  probability: number
  toxicity_score: number
}

export interface Message {
  id: number
  seq_index: number
  text: string
  labels: MessageLabel[]
}

export type EscalationStatus = 'stable' | 'elevated' | 'escalating'

export interface EscalationSnapshot {
  seq_index: number
  windowed_mean: number
  slope: number
  escalation_score: number
  status: EscalationStatus
}

export interface Conversation {
  id: number
  source: string
  language: string
  messages: Message[]
  escalation_snapshots: EscalationSnapshot[]
}

export interface TrendSnapshot {
  time_bucket: string
  category: string
  language: string
  topic_id: number | null
  topic_terms: string | null
  message_count: number
}
