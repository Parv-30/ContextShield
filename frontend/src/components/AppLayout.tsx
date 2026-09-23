// frontend/src/components/AppLayout.tsx
import { NavLink } from 'react-router-dom'
import { ShieldAlert, MessageSquareText, FolderOpen, TrendingUp } from 'lucide-react'

const navLinkClasses = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-2 rounded px-3 py-2 text-sm font-medium transition-colors duration-200 cursor-pointer ${
    isActive
      ? 'bg-primary text-on-primary'
      : 'text-muted-foreground hover:bg-muted hover:text-foreground'
  }`

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card px-6 py-4">
        <div className="mx-auto flex max-w-6xl items-center gap-3">
          <ShieldAlert className="h-6 w-6 text-primary" aria-hidden="true" />
          <div>
            <h1 className="text-xl font-semibold text-foreground">ContextShield</h1>
            <p className="text-sm text-muted-foreground">
              Multilingual harmful content detection and trend analysis
            </p>
          </div>
        </div>
      </header>
      <nav className="border-b border-border bg-card px-6 py-2">
        <div className="mx-auto flex max-w-6xl gap-2">
          <NavLink to="/" end className={navLinkClasses}>
            <MessageSquareText className="h-4 w-4" aria-hidden="true" />
            Analyzer
          </NavLink>
          <NavLink to="/browse" className={navLinkClasses}>
            <FolderOpen className="h-4 w-4" aria-hidden="true" />
            Browse
          </NavLink>
          <NavLink to="/trends" className={navLinkClasses}>
            <TrendingUp className="h-4 w-4" aria-hidden="true" />
            Trends
          </NavLink>
        </div>
      </nav>
      <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
    </div>
  )
}
