import { describe, it, expect, vi, beforeEach } from 'vitest'
import { createConversation, getConversation, getTopics, getTrends } from './client'

describe('api client', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('createConversation posts to /conversations and returns parsed JSON', async () => {
    const mockResponse = { id: 1, source: 'upload', language: 'en', messages: [], escalation_snapshots: [] }
    ;(fetch as any).mockResolvedValue({
      ok: true,
      text: async () => JSON.stringify(mockResponse),
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
      status: 404,
      text: async () => JSON.stringify({ detail: 'Conversation not found' }),
    })

    await expect(getConversation(999)).rejects.toThrow('Conversation not found')
  })

  it('getTrends builds query params from filters', async () => {
    ;(fetch as any).mockResolvedValue({ ok: true, text: async () => JSON.stringify([]) })

    await getTrends({ category: 'offensive', language: 'en' })

    const calledUrl = (fetch as any).mock.calls[0][0] as string
    expect(calledUrl).toContain('category=offensive')
    expect(calledUrl).toContain('language=en')
  })

  it('throws a readable message when detail is a FastAPI validation error array', async () => {
    ;(fetch as any).mockResolvedValue({
      ok: false,
      status: 422,
      text: async () =>
        JSON.stringify({
          detail: [
            { loc: ['body', 'messages', 0, 'text'], msg: 'ensure this value has at most 2000 characters', type: 'value_error' },
            { loc: ['body', 'language'], msg: 'field required', type: 'value_error.missing' },
          ],
        }),
    })

    await expect(getConversation(1)).rejects.toThrow(
      'ensure this value has at most 2000 characters, field required'
    )
  })

  it('handles a non-JSON error body without crashing and includes the status code', async () => {
    ;(fetch as any).mockResolvedValue({
      ok: false,
      status: 502,
      text: async () => 'Bad Gateway',
    })

    await expect(getConversation(1)).rejects.toThrow('502')
  })

  it('getTopics fetches /trends/topics and returns parsed JSON', async () => {
    const mockTopics = [
      { id: 1, topic_id: 0, topic_terms: 'a,b,c', topic_examples: 'example one\nexample two', created_at: '2026-01-01T00:00:00Z' },
    ]
    ;(fetch as any).mockResolvedValue({
      ok: true,
      text: async () => JSON.stringify(mockTopics),
    })

    const result = await getTopics()

    expect(result).toEqual(mockTopics)
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/trends/topics'))
  })
})
