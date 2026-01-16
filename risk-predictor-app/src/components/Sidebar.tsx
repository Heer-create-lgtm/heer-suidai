import { NavLink } from 'react-router-dom';

interface NavItem {
    icon: string;
    label: string;
    path: string;
}

const mainNavItems: NavItem[] = [
    { icon: '🔐', label: 'Risk Predictor', path: '/' },
    { icon: '📊', label: 'Dashboard', path: '/dashboard' },
    { icon: '📍', label: 'Geographic Hotspots', path: '/hotspots' },
    { icon: '📈', label: 'Enrollment Forecast', path: '/forecast' },
    { icon: '⚠️', label: 'Anomaly Detection', path: '/anomalies' },
    { icon: '👥', label: 'Gender Tracker', path: '/gender' },
];

const systemNavItems: NavItem[] = [
    { icon: '📄', label: 'Reports', path: '/reports' },
    { icon: '⚙️', label: 'Settings', path: '/settings' },
];

export function Sidebar() {
    return (
        <aside className="sidebar">
            <div className="logo">
                <div className="logo-icon">U</div>
                <div>
                    <div className="logo-text">UIDAI Analytics</div>
                    <div className="logo-subtitle">Policy Intelligence</div>
                </div>
            </div>

            <div className="nav-section">
                <div className="nav-label">Analytics</div>
                {mainNavItems.map((item) => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
                    >
                        <span className="nav-icon">{item.icon}</span>
                        <span>{item.label}</span>
                    </NavLink>
                ))}
            </div>

            <div className="nav-section">
                <div className="nav-label">System</div>
                {systemNavItems.map((item) => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
                    >
                        <span className="nav-icon">{item.icon}</span>
                        <span>{item.label}</span>
                    </NavLink>
                ))}
            </div>
        </aside>
    );
}
