// frontend/src/components/EscalationChart.tsx
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts'
import type { EscalationSnapshot } from '../api/types'

const STATUS_COLOR: Record<string, string> = {
  stable: '#3B82F6',
  elevated: '#D97706',
  escalating: '#DC2626',
}

export default function EscalationChart({ snapshots }: { snapshots: EscalationSnapshot[] }) {
  if (snapshots.length === 0) {
    return (
      <p className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground shadow-sm">
        Not enough messages for an escalation timeline (need at least 3).
      </p>
    )
  }

  const data = snapshots.map((s) => ({
    seq_index: s.seq_index,
    escalation_score: s.escalation_score,
    status: s.status,
  }))
  const latestStatus = data[data.length - 1].status
  const lineColor = STATUS_COLOR[latestStatus] ?? '#3B82F6'

  return (
    <div className="rounded-lg border border-border bg-card p-4 shadow-sm">
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#DBEAFE" />
          <XAxis dataKey="seq_index" label={{ value: 'Message index', position: 'insideBottom', offset: -5 }} stroke="#475569" fontSize={12} />
          <YAxis domain={[0, 1]} stroke="#475569" fontSize={12} />
          <Tooltip
            formatter={(value, _name, props) => [
              `${Number(value).toFixed(2)} (${props.payload.status})`,
              'Escalation score',
            ]}
          />
          <Line type="monotone" dataKey="escalation_score" stroke={lineColor} strokeWidth={2} dot />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
