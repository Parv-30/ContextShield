// frontend/src/components/TopicCards.tsx
import { Hash } from 'lucide-react'
import type { TrendSnapshot } from '../api/types'

export default function TopicCards({ snapshots }: { snapshots: TrendSnapshot[] }) {
  const seen = new Map<number, string>()
  for (const snap of snapshots) {
    if (snap.topic_id !== null && snap.topic_terms && !seen.has(snap.topic_id)) {
      seen.set(snap.topic_id, snap.topic_terms)
    }
  }

  if (seen.size === 0) {
    return (
      <p className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground shadow-sm">
        No topic clusters available yet (need more harmful messages).
      </p>
    )
  }

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from(seen.entries()).map(([topicId, terms]) => (
        <div key={topicId} className="rounded-lg border border-border bg-card p-4 shadow-sm transition-shadow duration-200 hover:shadow-md">
          <div className="mb-1 flex items-center gap-1.5 text-sm font-semibold text-foreground">
            <Hash className="h-4 w-4 text-primary" aria-hidden="true" />
            Topic {topicId}
          </div>
          <p className="text-sm text-muted-foreground">{terms}</p>
        </div>
      ))}
    </div>
  )
}
