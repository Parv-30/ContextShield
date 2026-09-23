import type { Conversation, TopicSnapshot, TrendSnapshot } from './types'

const BASE_URL = import.meta.env.VITE_API_BASE_URL as string

async function handleResponse<T>(response: Response): Promise<T> {
  const text = await response.text()
  let body: any = undefined
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = undefined
    }
  }

  if (!response.ok) {
    const detail = body?.detail
    if (typeof detail === 'string') {
      throw new Error(detail)
    }
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => (item && typeof item === 'object' && 'msg' in item ? String(item.msg) : String(item)))
        .join(', ')
      throw new Error(messages || `Request failed (${response.status})`)
    }
    throw new Error(`Request failed (${response.status})`)
  }

  return body as T
}

export async function createConversation(payload: {
  messages: { text: string; author?: string }[]
  source: 'upload' | 'seed'
  language: 'en' | 'hi' | 'ta'
}): Promise<Conversation> {
  const response = await fetch(`${BASE_URL}/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return handleResponse<Conversation>(response)
}

export async function getConversation(id: number): Promise<Conversation> {
  const response = await fetch(`${BASE_URL}/conversations/${id}`)
  return handleResponse<Conversation>(response)
}

export async function listConversations(limit = 20, offset = 0): Promise<Conversation[]> {
  const response = await fetch(`${BASE_URL}/conversations?limit=${limit}&offset=${offset}`)
  return handleResponse<Conversation[]>(response)
}

export async function uploadConversation(file: File, language: string): Promise<Conversation> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('language', language)
  const response = await fetch(`${BASE_URL}/conversations/upload`, {
    method: 'POST',
    body: formData,
  })
  return handleResponse<Conversation>(response)
}

export async function getTrends(filters?: {
  start?: string
  end?: string
  category?: string
  language?: string
}): Promise<TrendSnapshot[]> {
  const params = new URLSearchParams()
  if (filters?.start) params.set('start', filters.start)
  if (filters?.end) params.set('end', filters.end)
  if (filters?.category) params.set('category', filters.category)
  if (filters?.language) params.set('language', filters.language)
  const query = params.toString()
  const response = await fetch(`${BASE_URL}/trends${query ? `?${query}` : ''}`)
  return handleResponse<TrendSnapshot[]>(response)
}

export async function getTopics(): Promise<TopicSnapshot[]> {
  const response = await fetch(`${BASE_URL}/trends/topics`)
  return handleResponse<TopicSnapshot[]>(response)
}
