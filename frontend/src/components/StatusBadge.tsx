// frontend/src/components/StatusBadge.tsx
const STATUS_STYLES: Record<string, string> = {
  stable: 'bg-secondary/10 text-secondary border-secondary/30',
  elevated: 'bg-accent/10 text-accent border-accent/30',
  escalating: 'bg-destructive/10 text-destructive border-destructive/30',
  'n/a': 'bg-muted text-muted-foreground border-border',
}

export default function StatusBadge({ status }: { status: string }) {
  const style = STATUS_STYLES[status] ?? STATUS_STYLES['n/a']
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium capitalize ${style}`}>
      {status}
    </span>
  )
}
