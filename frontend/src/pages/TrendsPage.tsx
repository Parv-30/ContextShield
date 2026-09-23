// frontend/src/pages/TrendsPage.tsx
import { useEffect, useMemo, useState } from 'react'
import { BarChart3, MessagesSquare, Layers } from 'lucide-react'
import { getTrends } from '../api/client'
import type { TrendSnapshot } from '../api/types'
import TrendChart from '../components/TrendChart'
import TopicCards from '../components/TopicCards'
import StatTile from '../components/StatTile'

export default function TrendsPage() {
  const [snapshots, setSnapshots] = useState<TrendSnapshot[]>([])
  const [language, setLanguage] = useState('')
  const [category, setCategory] = useState('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getTrends({ language: language || undefined, category: category || undefined })
      .then(setSnapshots)
      .catch((err) => setError(err.message))
  }, [language, category])

  const totalMessages = useMemo(() => snapshots.reduce((sum, s) => sum + s.message_count, 0), [snapshots])
  const topicCount = useMemo(() => new Set(snapshots.map((s) => s.topic_id).filter((id) => id !== null)).size, [snapshots])
  const categoryCount = useMemo(() => new Set(snapshots.map((s) => s.category)).size, [snapshots])

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-lg font-semibold text-foreground">Trends Dashboard</h2>
        <p className="text-sm text-muted-foreground">Harmful content volume and emerging topics over time.</p>
      </div>

      {error && (
        <p role="alert" className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </p>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatTile label="Flagged messages" value={totalMessages} icon={MessagesSquare} />
        <StatTile label="Categories represented" value={categoryCount} icon={BarChart3} />
        <StatTile label="Topic clusters" value={topicCount} icon={Layers} />
      </div>

      <div className="flex flex-wrap gap-3">
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="cursor-pointer rounded border border-border px-3 py-2 text-sm text-foreground transition-colors duration-200 focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
        >
          <option value="">All languages</option>
          <option value="en">English</option>
          <option value="hi">Hindi</option>
          <option value="ta">Tamil</option>
        </select>
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="cursor-pointer rounded border border-border px-3 py-2 text-sm text-foreground transition-colors duration-200 focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
        >
          <option value="">All categories</option>
          <option value="hate_speech">Hate speech</option>
          <option value="offensive">Offensive</option>
          <option value="threat">Threat</option>
          <option value="harassment">Harassment</option>
        </select>
      </div>

      <TrendChart snapshots={snapshots} />

      <div>
        <h3 className="mb-3 text-base font-semibold text-foreground">Emerging Topics</h3>
        <TopicCards snapshots={snapshots} />
      </div>
    </div>
  )
}
