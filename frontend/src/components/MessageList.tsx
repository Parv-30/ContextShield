// frontend/src/components/MessageList.tsx
import type { Message } from '../api/types'
import CategoryBadge from './CategoryBadge'

function topCategory(message: Message): { category: string; probability: number } {
  const nonNone = message.labels.filter((l) => l.category !== 'none')
  return nonNone.reduce(
    (best, current) => (current.probability > best.probability ? current : best),
    { category: 'none', probability: 0 }
  )
}

export default function MessageList({ messages }: { messages: Message[] }) {
  return (
    <ul className="divide-y divide-border rounded-lg border border-border bg-card shadow-sm">
      {messages.map((message) => {
        const top = topCategory(message)
        return (
          <li key={message.id} className="flex items-center justify-between gap-4 px-4 py-3">
            <span className="text-sm text-foreground">{message.text}</span>
            {top.probability > 0.5 && (
              <CategoryBadge category={top.category} probability={top.probability} />
            )}
          </li>
        )
      })}
    </ul>
  )
}
