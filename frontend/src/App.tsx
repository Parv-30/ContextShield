// frontend/src/App.tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import AppLayout from './components/AppLayout'
import AnalyzerPage from './pages/AnalyzerPage'
import BrowsePage from './pages/BrowsePage'
import TrendsPage from './pages/TrendsPage'

function App() {
  return (
    <BrowserRouter>
      <AppLayout>
        <Routes>
          <Route path="/" element={<AnalyzerPage />} />
          <Route path="/browse" element={<BrowsePage />} />
          <Route path="/trends" element={<TrendsPage />} />
        </Routes>
      </AppLayout>
    </BrowserRouter>
  )
}

export default App
