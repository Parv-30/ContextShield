import { describe, it, expect, vi, beforeEach } from 'vitest'
import { createConversation, getConversation, getTrends } from './client'

describe('api client', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('createConversation posts to /conversations and returns parsed JSON', async () => {
    const mockResponse = { id: 1, source: 'upload', language: 'en', messages: [], escalation_snapshots: [] }
    ;(fetch as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    })

    const result = await createConversation({
      messages: [{ text: 'hello' }],
      source: 'upload',
      language: 'en',
    })

    expect(result).toEqual(mockResponse)
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/conversations'),
      expect.objectContaining({ method: 'POST' })
    )
  })

  it('getConversation throws on non-ok response with detail message', async () => {
    ;(fetch as any).mockResolvedValue({
      ok: false,
      json: async () => ({ detail: 'Conversation not found' }),
    })

    await expect(getConversation(999)).rejects.toThrow('Conversation not found')
  })

  it('getTrends builds query params from filters', async () => {
    ;(fetch as any).mockResolvedValue({ ok: true, json: async () => [] })

    await getTrends({ category: 'offensive', language: 'en' })

    const calledUrl = (fetch as any).mock.calls[0][0] as string
    expect(calledUrl).toContain('category=offensive')
    expect(calledUrl).toContain('language=en')
  })
})
