// frontend/src/components/CategoryBadge.tsx
const CATEGORY_STYLES: Record<string, string> = {
  hate_speech: 'bg-destructive/10 text-destructive border-destructive/30',
  threat: 'bg-destructive/10 text-destructive border-destructive/30',
  offensive: 'bg-accent/10 text-accent border-accent/30',
  harassment: 'bg-accent/10 text-accent border-accent/30',
  none: 'bg-muted text-muted-foreground border-border',
}

const CATEGORY_LABELS: Record<string, string> = {
  hate_speech: 'Hate speech',
  threat: 'Threat',
  offensive: 'Offensive',
  harassment: 'Harassment',
  none: 'None',
}

export default function CategoryBadge({ category, probability }: { category: string; probability: number }) {
  const style = CATEGORY_STYLES[category] ?? CATEGORY_STYLES.none
  const label = CATEGORY_LABELS[category] ?? category
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${style}`}
    >
      {label}
      <span className="font-mono">{probability.toFixed(2)}</span>
    </span>
  )
}
