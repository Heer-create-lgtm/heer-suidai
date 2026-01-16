import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { RiskPredictor } from './components/RiskPredictor';
import './index.css';

function App() {
  return (
    <BrowserRouter>
      <div className="app-container">
        <Sidebar />
        <Routes>
          <Route path="/" element={<RiskPredictor />} />
          <Route path="/dashboard" element={<PlaceholderPage title="Dashboard" />} />
          <Route path="/hotspots" element={<PlaceholderPage title="Geographic Hotspots" />} />
          <Route path="/forecast" element={<PlaceholderPage title="Enrollment Forecast" />} />
          <Route path="/anomalies" element={<PlaceholderPage title="Anomaly Detection" />} />
          <Route path="/gender" element={<PlaceholderPage title="Gender Tracker" />} />
          <Route path="/reports" element={<PlaceholderPage title="Reports" />} />
          <Route path="/settings" element={<PlaceholderPage title="Settings" />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

function PlaceholderPage({ title }: { title: string }) {
  return (
    <main className="main-content">
      <div className="page-header">
        <div>
          <h1 className="page-title">{title}</h1>
          <p className="page-subtitle">This feature is coming soon</p>
        </div>
      </div>
      <div className="empty-state">
        <div className="empty-icon">🚧</div>
        <h2 className="empty-title">{title}</h2>
        <p className="empty-text">This feature is under development. Check back soon!</p>
      </div>
    </main>
  );
}

export default App;
