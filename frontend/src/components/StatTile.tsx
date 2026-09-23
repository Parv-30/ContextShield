// frontend/src/components/StatTile.tsx
import type { LucideIcon } from 'lucide-react'

export default function StatTile({
  label,
  value,
  icon: Icon,
}: {
  label: string
  value: number | string
  icon: LucideIcon
}) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-border bg-card p-4 shadow-sm">
      <div className="rounded bg-primary/10 p-2">
        <Icon className="h-5 w-5 text-primary" aria-hidden="true" />
      </div>
      <div>
        <p className="font-mono text-2xl font-semibold text-foreground">{value}</p>
        <p className="text-xs text-muted-foreground">{label}</p>
      </div>
    </div>
  )
}
