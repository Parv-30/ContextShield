// frontend/src/pages/BrowsePage.tsx
import { useEffect, useState } from 'react'
import { Globe2 } from 'lucide-react'
import { listConversations, getConversation } from '../api/client'
import type { Conversation } from '../api/types'
import MessageList from '../components/MessageList'
import EscalationChart from '../components/EscalationChart'
import StatusBadge from '../components/StatusBadge'

const LANGUAGE_LABELS: Record<string, string> = { en: 'English', hi: 'Hindi', ta: 'Tamil' }

export default function BrowsePage() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [selected, setSelected] = useState<Conversation | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listConversations().then(setConversations).catch((err) => setError(err.message))
  }, [])

  async function handleSelect(id: number) {
    try {
      const convo = await getConversation(id)
      setSelected(convo)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load conversation')
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-lg font-semibold text-foreground">Browse Conversations</h2>
        <p className="text-sm text-muted-foreground">Seeded and previously analyzed conversations.</p>
      </div>

      {error && (
        <p role="alert" className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </p>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <ul className="flex flex-col gap-2">
          {conversations.map((convo) => {
            const latestStatus = convo.escalation_snapshots.at(-1)?.status ?? 'n/a'
            const isSelected = selected?.id === convo.id
            return (
              <li key={convo.id}>
                <button
                  onClick={() => handleSelect(convo.id)}
                  className={`flex w-full cursor-pointer items-center justify-between gap-3 rounded-lg border px-4 py-3 text-left text-sm shadow-sm transition-colors duration-200 ${
                    isSelected ? 'border-primary bg-primary/5' : 'border-border bg-card hover:border-primary/50'
                  }`}
                >
                  <span className="flex items-center gap-2">
                    <Globe2 className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                    <span className="font-medium text-foreground">#{convo.id}</span>
                    <span className="text-muted-foreground">
                      {LANGUAGE_LABELS[convo.language] ?? convo.language} · {convo.source} · {convo.messages.length} msgs
                    </span>
                  </span>
                  <StatusBadge status={latestStatus} />
                </button>
              </li>
            )
          })}
        </ul>

        {selected && (
          <div className="flex flex-col gap-4">
            <h3 className="text-base font-semibold text-foreground">Conversation #{selected.id}</h3>
            <MessageList messages={selected.messages} />
            <EscalationChart snapshots={selected.escalation_snapshots} />
          </div>
        )}
      </div>
    </div>
  )
}
