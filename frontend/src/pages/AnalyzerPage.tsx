// frontend/src/pages/AnalyzerPage.tsx
import { useState } from 'react'
import { Upload, Loader2 } from 'lucide-react'
import { createConversation, uploadConversation } from '../api/client'
import type { Conversation } from '../api/types'
import MessageList from '../components/MessageList'
import EscalationChart from '../components/EscalationChart'

export default function AnalyzerPage() {
  const [text, setText] = useState('')
  const [language, setLanguage] = useState<'en' | 'hi' | 'ta'>('en')
  const [file, setFile] = useState<File | null>(null)
  const [conversation, setConversation] = useState<Conversation | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      let result: Conversation
      if (file) {
        result = await uploadConversation(file, language)
      } else {
        const lines = text.split('\n').map((l) => l.trim()).filter(Boolean)
        result = await createConversation({
          messages: lines.map((line) => ({ text: line })),
          source: 'upload',
          language,
        })
      }
      setConversation(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-lg font-semibold text-foreground">Analyze a Conversation</h2>
        <p className="text-sm text-muted-foreground">
          Paste a conversation or upload a text file to detect harmful content and escalation.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4 rounded-lg border border-border bg-card p-6 shadow-sm">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste one message per line"
          rows={8}
          disabled={!!file}
          className="w-full rounded border border-border px-4 py-3 text-sm text-foreground transition-colors duration-200 focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20 disabled:bg-muted disabled:text-muted-foreground"
        />

        <div className="flex flex-wrap items-center gap-4">
          <label className="flex cursor-pointer items-center gap-2 rounded border border-border px-3 py-2 text-sm text-muted-foreground transition-colors duration-200 hover:border-primary hover:text-primary">
            <Upload className="h-4 w-4" aria-hidden="true" />
            {file ? file.name : 'Upload .txt file'}
            <input
              type="file"
              accept=".txt"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>

          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value as 'en' | 'hi' | 'ta')}
            className="cursor-pointer rounded border border-border px-3 py-2 text-sm text-foreground transition-colors duration-200 focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
          >
            <option value="en">English</option>
            <option value="hi">Hindi</option>
            <option value="ta">Tamil</option>
          </select>

          <button
            type="submit"
            disabled={loading}
            className="ml-auto flex items-center gap-2 rounded bg-accent px-6 py-2 text-sm font-semibold text-on-accent transition-all duration-200 hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50 cursor-pointer"
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}
            {loading ? 'Analyzing…' : 'Analyze'}
          </button>
        </div>
      </form>

      {error && (
        <p role="alert" className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </p>
      )}

      {conversation && (
        <div className="flex flex-col gap-4">
          <h3 className="text-base font-semibold text-foreground">Results</h3>
          <MessageList messages={conversation.messages} />
          <EscalationChart snapshots={conversation.escalation_snapshots} />
        </div>
      )}
    </div>
  )
}
