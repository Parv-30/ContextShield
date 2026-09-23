// frontend/src/components/TopicCards.tsx
import { Hash } from 'lucide-react'
import type { TopicSnapshot } from '../api/types'

export default function TopicCards({ topics }: { topics: TopicSnapshot[] }) {
  if (topics.length === 0) {
    return (
      <p className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground shadow-sm">
        No topic clusters available yet (need more harmful messages).
      </p>
    )
  }

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {topics.map((topic) => {
        const examples = topic.topic_examples ? topic.topic_examples.split('\n').filter(Boolean) : []
        return (
          <div key={topic.id} className="rounded-lg border border-border bg-card p-4 shadow-sm transition-shadow duration-200 hover:shadow-md">
            <div className="mb-1 flex items-center gap-1.5 text-sm font-semibold text-foreground">
              <Hash className="h-4 w-4 text-primary" aria-hidden="true" />
              Topic {topic.topic_id}
            </div>
            <p className="text-sm text-muted-foreground">{topic.topic_terms}</p>
            {examples.length > 0 && (
              <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-muted-foreground">
                {examples.map((example, idx) => (
                  <li key={idx}>{example}</li>
                ))}
              </ul>
            )}
          </div>
        )
      })}
    </div>
  )
}
