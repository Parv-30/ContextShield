// frontend/src/components/TrendChart.tsx
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer } from 'recharts'
import type { TrendSnapshot } from '../api/types'

const COLORS: Record<string, string> = {
  hate_speech: '#DC2626',
  threat: '#DC2626',
  offensive: '#D97706',
  harassment: '#D97706',
}

export default function TrendChart({ snapshots }: { snapshots: TrendSnapshot[] }) {
  const byBucket = new Map<string, Record<string, number>>()
  for (const snap of snapshots) {
    const entry = byBucket.get(snap.time_bucket) ?? {}
    entry[snap.category] = (entry[snap.category] ?? 0) + snap.message_count
    byBucket.set(snap.time_bucket, entry)
  }
  const data = Array.from(byBucket.entries())
    .map(([time_bucket, counts]) => ({ time_bucket, ...counts }))
    .sort((a, b) => a.time_bucket.localeCompare(b.time_bucket))

  const categories = Array.from(new Set(snapshots.map((s) => s.category)))

  if (data.length === 0) {
    return (
      <p className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground shadow-sm">
        No trend data available yet.
      </p>
    )
  }

  return (
    <div className="rounded-lg border border-border bg-card p-4 shadow-sm">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#DBEAFE" />
          <XAxis dataKey="time_bucket" stroke="#475569" fontSize={12} />
          <YAxis stroke="#475569" fontSize={12} />
          <Tooltip />
          <Legend />
          {categories.map((category) => (
            <Bar key={category} dataKey={category} stackId="a" fill={COLORS[category] ?? '#3B82F6'} radius={[2, 2, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
