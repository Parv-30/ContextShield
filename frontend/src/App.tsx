// frontend/src/App.tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import AppLayout from './components/AppLayout'
import AnalyzerPage from './pages/AnalyzerPage'

function App() {
  return (
    <BrowserRouter>
      <AppLayout>
        <Routes>
          <Route path="/" element={<AnalyzerPage />} />
        </Routes>
      </AppLayout>
    </BrowserRouter>
  )
}

export default App
