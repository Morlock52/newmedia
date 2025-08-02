// Media Server Dashboard - React Component Structure
// This file demonstrates the component architecture for the media server dashboard

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Chart as ChartJS } from 'chart.js/auto';

// ============= TYPES & INTERFACES =============

interface Service {
  id: string;
  name: string;
  icon: string;
  description: string;
  status: 'running' | 'stopped' | 'warning';
  health: number;
  cpu: number;
  memory: string;
  disk: string;
  network: {
    download: string;
    upload: string;
  };
  container: string;
  uptime: string;
  connections: number;
}

interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  progress: number;
  maxProgress: number;
  unlocked: boolean;
  unlockedAt?: Date;
}

interface User {
  name: string;
  avatar: string;
  level: number;
  xp: number;
  nextLevelXp: number;
  streak: number;
  achievements: Achievement[];
}

interface EnvVariable {
  key: string;
  value: string;
  isSecret: boolean;
  category: 'security' | 'paths' | 'network' | 'general';
  description?: string;
}

type ViewMode = 'simple' | 'advanced';
type Theme = 'dark' | 'light' | 'auto';

// ============= HOOKS =============

// Custom hook for real-time service monitoring
const useServiceMonitoring = (serviceId: string) => {
  const [metrics, setMetrics] = useState({
    cpu: 0,
    memory: 0,
    network: { in: 0, out: 0 }
  });

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8080/services/${serviceId}/metrics`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMetrics(data);
    };

    return () => ws.close();
  }, [serviceId]);

  return metrics;
};

// Custom hook for gesture handling
const useGestures = (element: HTMLElement | null) => {
  const [gesture, setGesture] = useState<string | null>(null);
  
  useEffect(() => {
    if (!element) return;

    let hammer: any; // Hammer.js instance
    
    // Initialize touch gestures
    if (typeof window !== 'undefined' && 'Hammer' in window) {
      hammer = new (window as any).Hammer(element);
      
      hammer.on('swipeleft', () => setGesture('swipeleft'));
      hammer.on('swiperight', () => setGesture('swiperight'));
      hammer.on('press', () => setGesture('longpress'));
      hammer.on('pinch', () => setGesture('pinch'));
    }

    return () => {
      if (hammer) hammer.destroy();
    };
  }, [element]);

  return gesture;
};

// Custom hook for gamification
const useGamification = (userId: string) => {
  const [xpGained, setXpGained] = useState(0);
  const [newAchievements, setNewAchievements] = useState<Achievement[]>([]);

  const awardXP = (amount: number, reason: string) => {
    setXpGained(prev => prev + amount);
    
    // Check for achievements
    checkAchievements(reason);
    
    // Persist to backend
    fetch('/api/users/xp', {
      method: 'POST',
      body: JSON.stringify({ userId, amount, reason })
    });
  };

  const checkAchievements = (action: string) => {
    // Achievement logic here
  };

  return { xpGained, newAchievements, awardXP };
};

// ============= CONTEXT PROVIDERS =============

const DashboardContext = React.createContext({
  viewMode: 'simple' as ViewMode,
  setViewMode: (mode: ViewMode) => {},
  theme: 'dark' as Theme,
  setTheme: (theme: Theme) => {},
  user: null as User | null,
  services: [] as Service[]
});

// ============= MAIN DASHBOARD COMPONENT =============

export const MediaDashboard: React.FC = () => {
  const [viewMode, setViewMode] = useState<ViewMode>('simple');
  const [services, setServices] = useState<Service[]>([]);
  const [showEnvEditor, setShowEnvEditor] = useState(false);

  return (
    <DashboardContext.Provider value={{ viewMode, setViewMode, theme: 'dark', setTheme: () => {}, user: null, services }}>
      <div className="dashboard">
        <Header />
        <AnimatePresence mode="wait">
          {viewMode === 'simple' ? (
            <motion.div
              key="simple"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <SimpleDashboard />
            </motion.div>
          ) : (
            <motion.div
              key="advanced"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <AdvancedDashboard />
            </motion.div>
          )}
        </AnimatePresence>
        
        {showEnvEditor && <EnvEditor onClose={() => setShowEnvEditor(false)} />}
      </div>
    </DashboardContext.Provider>
  );
};

// ============= HEADER COMPONENT =============

const Header: React.FC = () => {
  const { viewMode, setViewMode, user } = React.useContext(DashboardContext);
  
  return (
    <header className="header">
      <div className="logo">
        {viewMode === 'simple' ? '🏠' : '🖥️'} Media Server {viewMode === 'simple' ? 'Hub' : 'Control Center'}
      </div>
      
      <div className="mode-switcher">
        <button 
          className={`mode-btn ${viewMode === 'simple' ? 'active' : ''}`}
          onClick={() => setViewMode('simple')}
        >
          Simple
        </button>
        <button 
          className={`mode-btn ${viewMode === 'advanced' ? 'active' : ''}`}
          onClick={() => setViewMode('advanced')}
        >
          Advanced
        </button>
        
        <UserAvatar user={user} />
      </div>
    </header>
  );
};

// ============= SIMPLE MODE COMPONENTS =============

const SimpleDashboard: React.FC = () => {
  const { user, services } = React.useContext(DashboardContext);
  const { awardXP } = useGamification(user?.name || '');

  return (
    <div className="simple-dashboard">
      <WelcomeSection user={user} />
      <GamificationBar user={user} />
      <QuickActions onAction={(action) => awardXP(5, `Used ${action} quick action`)} />
      <ServiceList services={services} simple={true} />
      <AchievementSection achievements={user?.achievements || []} />
    </div>
  );
};

const GamificationBar: React.FC<{ user: User | null }> = ({ user }) => {
  if (!user) return null;

  const xpPercentage = (user.xp / user.nextLevelXp) * 100;

  return (
    <motion.div 
      className="gamification-bar"
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <div className="streak-info">
        <motion.span 
          className="fire-emoji"
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ repeat: Infinity, duration: 2 }}
        >
          🔥
        </motion.span>
        <span>{user.streak} days</span>
      </div>
      
      <div className="level-info">
        <div className="level-title">Level {user.level} Media Master</div>
        <div className="xp-bar">
          <motion.div 
            className="xp-fill"
            initial={{ width: 0 }}
            animate={{ width: `${xpPercentage}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
          />
        </div>
        <div className="xp-text">{user.xp}/{user.nextLevelXp} XP</div>
      </div>
    </motion.div>
  );
};

const QuickActions: React.FC<{ onAction: (action: string) => void }> = ({ onAction }) => {
  const actions = [
    { id: 'watch', icon: '📺', label: 'Watch' },
    { id: 'download', icon: '🎬', label: 'Download' },
    { id: 'organize', icon: '📚', label: 'Organize' },
    { id: 'play', icon: '🎮', label: 'Play' }
  ];

  return (
    <div className="quick-actions">
      <h2 className="section-title">Quick Actions</h2>
      <div className="action-grid">
        {actions.map((action, index) => (
          <motion.button
            key={action.id}
            className="action-card"
            onClick={() => onAction(action.id)}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span className="action-icon">{action.icon}</span>
            <span className="action-label">{action.label}</span>
          </motion.button>
        ))}
      </div>
    </div>
  );
};

// ============= SERVICE COMPONENTS =============

const ServiceCard: React.FC<{ 
  service: Service; 
  simple?: boolean;
  onToggle: (serviceId: string) => void;
}> = ({ service, simple = false, onToggle }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);
  const gesture = useGestures(cardRef.current);
  const metrics = useServiceMonitoring(service.id);

  useEffect(() => {
    if (gesture === 'swipeleft') {
      setIsExpanded(true);
    } else if (gesture === 'swiperight') {
      setIsExpanded(false);
    }
  }, [gesture]);

  return (
    <motion.div
      ref={cardRef}
      className={`service-card ${service.status}`}
      layout
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      whileHover={{ scale: 1.02 }}
    >
      <div className="service-main">
        <div className="service-info">
          <div className="service-icon">{service.icon}</div>
          <div className="service-details">
            <h3>{service.name}</h3>
            <p className="service-description">{service.description}</p>
            {!simple && (
              <div className="service-stats">
                CPU: {metrics.cpu}% • MEM: {service.memory} • 
                ↓{metrics.network.in}MB/s ↑{metrics.network.out}MB/s
              </div>
            )}
          </div>
        </div>
        
        <div className="service-controls">
          <ServiceStatus status={service.status} health={service.health} />
          <ToggleSwitch 
            active={service.status === 'running'}
            onChange={() => onToggle(service.id)}
          />
        </div>
      </div>
      
      <AnimatePresence>
        {isExpanded && !simple && (
          <motion.div
            className="service-expanded"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
          >
            <ServiceDetails service={service} />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

const ToggleSwitch: React.FC<{ 
  active: boolean; 
  onChange: () => void;
  size?: 'small' | 'medium' | 'large';
}> = ({ active, onChange, size = 'medium' }) => {
  return (
    <motion.button
      className={`toggle-switch ${size} ${active ? 'active' : ''}`}
      onClick={onChange}
      whileTap={{ scale: 0.95 }}
    >
      <motion.div 
        className="toggle-thumb"
        layout
        transition={{ type: "spring", stiffness: 500, damping: 30 }}
      />
    </motion.button>
  );
};

// ============= ADVANCED MODE COMPONENTS =============

const AdvancedDashboard: React.FC = () => {
  const { services } = React.useContext(DashboardContext);
  
  return (
    <div className="advanced-dashboard">
      <div className="dashboard-grid">
        <SystemOverview />
        <AchievementsPanel />
      </div>
      
      <ServiceMatrix services={services} />
      
      <div className="dashboard-grid">
        <ContainerOrchestra services={services} />
        <PerformanceGraph />
      </div>
      
      <EnvConfiguration />
    </div>
  );
};

const SystemOverview: React.FC = () => {
  const [metrics, setMetrics] = useState({
    cpu: 52,
    ram: 84,
    disk: 61,
    network: { down: '2.4GB/s', up: '458MB/s' }
  });

  return (
    <motion.section 
      className="system-overview"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h2>System Overview</h2>
      
      <div className="resource-meters">
        {Object.entries(metrics).filter(([key]) => key !== 'network').map(([resource, value]) => (
          <ResourceMeter 
            key={resource}
            label={resource.toUpperCase()}
            value={value as number}
            warning={value > 80}
          />
        ))}
      </div>
      
      <div className="network-stats">
        <span>↓ {metrics.network.down}</span>
        <span>↑ {metrics.network.up}</span>
      </div>
    </motion.section>
  );
};

const ResourceMeter: React.FC<{ 
  label: string; 
  value: number; 
  warning?: boolean;
}> = ({ label, value, warning = false }) => {
  return (
    <div className="meter-item">
      <span className="meter-label">{label}</span>
      <div className="meter-bar">
        <motion.div 
          className={`meter-fill ${warning ? 'high' : ''}`}
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
        >
          {value}%
        </motion.div>
      </div>
    </div>
  );
};

const ContainerOrchestra: React.FC<{ services: Service[] }> = ({ services }) => {
  const [connections, setConnections] = useState<Array<[string, string]>>([]);

  useEffect(() => {
    // Simulate container connections
    const mockConnections: Array<[string, string]> = [
      ['plex', 'sonarr'],
      ['plex', 'radarr'],
      ['sonarr', 'prowlarr'],
      ['radarr', 'prowlarr']
    ];
    setConnections(mockConnections);
  }, []);

  return (
    <section className="container-orchestra">
      <h2>Container Orchestra</h2>
      
      <svg className="orchestra-diagram" viewBox="0 0 400 300">
        {/* Draw connections */}
        {connections.map(([from, to], index) => (
          <motion.line
            key={`${from}-${to}`}
            className="connection-line"
            x1={0} y1={0} x2={100} y2={100} // Calculate positions
            stroke="#2D3561"
            strokeWidth="2"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1, delay: index * 0.1 }}
          />
        ))}
        
        {/* Draw nodes */}
        {services.slice(0, 8).map((service, index) => (
          <ContainerNode 
            key={service.id}
            service={service}
            x={100 + (index % 4) * 80}
            y={50 + Math.floor(index / 4) * 100}
          />
        ))}
      </svg>
    </section>
  );
};

const ContainerNode: React.FC<{ 
  service: Service; 
  x: number; 
  y: number;
}> = ({ service, x, y }) => {
  return (
    <motion.g
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      whileHover={{ scale: 1.1 }}
      style={{ cursor: 'pointer' }}
    >
      <rect
        x={x - 30}
        y={y - 30}
        width={60}
        height={60}
        rx={12}
        fill="#242B47"
        stroke={service.status === 'running' ? '#10B981' : '#EF4444'}
        strokeWidth={2}
      />
      <text
        x={x}
        y={y + 5}
        textAnchor="middle"
        fill="#F8F9FA"
        fontSize={18}
        fontWeight={600}
      >
        {service.name.charAt(0)}
      </text>
    </motion.g>
  );
};

// ============= ENV CONFIGURATION COMPONENT =============

const EnvConfiguration: React.FC = () => {
  const [envVars, setEnvVars] = useState<EnvVariable[]>([]);
  const [filter, setFilter] = useState<'all' | 'modified' | 'security'>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [showAI, setShowAI] = useState(false);

  const filteredVars = envVars.filter(env => {
    if (filter === 'security' && env.category !== 'security') return false;
    if (searchTerm && !env.key.toLowerCase().includes(searchTerm.toLowerCase())) return false;
    return true;
  });

  return (
    <section className="env-config">
      <div className="section-header">
        <h2>Environment Configuration (.env)</h2>
        <div className="env-modes">
          <button onClick={() => setShowAI(false)}>Visual</button>
          <button onClick={() => setShowAI(true)}>AI Assistant</button>
        </div>
      </div>
      
      {showAI ? (
        <AIEnvAssistant />
      ) : (
        <>
          <div className="env-controls">
            <input 
              type="text"
              className="search-box"
              placeholder="Search..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <button 
              className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
              onClick={() => setFilter('all')}
            >
              All
            </button>
            <button 
              className={`filter-btn ${filter === 'modified' ? 'active' : ''}`}
              onClick={() => setFilter('modified')}
            >
              Modified
            </button>
            <button 
              className={`filter-btn ${filter === 'security' ? 'active' : ''}`}
              onClick={() => setFilter('security')}
            >
              Security
            </button>
          </div>
          
          <div className="env-list">
            {filteredVars.map(env => (
              <EnvVariableRow key={env.key} variable={env} />
            ))}
          </div>
          
          <div className="env-actions">
            <button className="btn-primary">💾 Save</button>
            <button className="btn-secondary">🔄 Reset</button>
            <button className="btn-secondary">📋 Export</button>
            <button className="btn-primary">🚀 Apply & Restart</button>
          </div>
        </>
      )}
    </section>
  );
};

const EnvVariableRow: React.FC<{ variable: EnvVariable }> = ({ variable }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [showValue, setShowValue] = useState(!variable.isSecret);
  const [value, setValue] = useState(variable.value);

  return (
    <motion.div 
      className="env-item"
      layout
      whileHover={{ scale: 1.02 }}
    >
      <span className="env-key">{variable.key}</span>
      
      <div className="env-value">
        {isEditing ? (
          <input 
            type={variable.isSecret && !showValue ? "password" : "text"}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onBlur={() => setIsEditing(false)}
            autoFocus
          />
        ) : (
          <span className={`env-value-text ${!showValue ? 'hidden' : ''}`}>
            {showValue ? value : '••••••••••••••••••'}
          </span>
        )}
        
        {variable.isSecret && (
          <button 
            className="env-action"
            onClick={() => setShowValue(!showValue)}
          >
            {showValue ? 'Hide' : 'Show'}
          </button>
        )}
        
        <button 
          className="env-action"
          onClick={() => setIsEditing(true)}
        >
          Edit
        </button>
      </div>
    </motion.div>
  );
};

// ============= AI ASSISTANT COMPONENT =============

const AIEnvAssistant: React.FC = () => {
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'assistant'; content: string }>>([]);
  const [input, setInput] = useState('');

  const suggestions = [
    "Set up automated TV show downloads",
    "Configure remote access",
    "Optimize for 4K streaming",
    "Enable hardware transcoding"
  ];

  const sendMessage = async () => {
    if (!input.trim()) return;

    setMessages(prev => [...prev, { role: 'user', content: input }]);
    setInput('');

    // Simulate AI response
    setTimeout(() => {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `I'll help you ${input.toLowerCase()}. Here's what I'll configure...`
      }]);
    }, 1000);
  };

  return (
    <div className="ai-assistant">
      <div className="ai-messages">
        {messages.length === 0 && (
          <div className="ai-welcome">
            <p>I'll help you configure your media server. What would you like to set up?</p>
            <div className="suggestions">
              {suggestions.map(suggestion => (
                <button 
                  key={suggestion}
                  className="suggestion-chip"
                  onClick={() => setInput(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}
        
        {messages.map((message, index) => (
          <motion.div
            key={index}
            className={`message ${message.role}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            {message.content}
          </motion.div>
        ))}
      </div>
      
      <div className="ai-input">
        <input
          type="text"
          placeholder="Type your request..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
};

// ============= PERFORMANCE MONITORING =============

const PerformanceGraph: React.FC = () => {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d'>('24h');

  useEffect(() => {
    if (!chartRef.current) return;

    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;

    const chart = new ChartJS(ctx, {
      type: 'line',
      data: {
        labels: Array.from({ length: 24 }, (_, i) => `${i}:00`),
        datasets: [{
          label: 'CPU Usage',
          data: Array.from({ length: 24 }, () => Math.random() * 100),
          borderColor: '#3B82F6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            ticks: { callback: (value) => `${value}%` }
          }
        }
      }
    });

    return () => chart.destroy();
  }, [timeRange]);

  return (
    <section className="performance-graph">
      <div className="section-header">
        <h2>Performance Graph</h2>
        <div className="time-range-selector">
          <button 
            className={timeRange === '1h' ? 'active' : ''}
            onClick={() => setTimeRange('1h')}
          >
            1h
          </button>
          <button 
            className={timeRange === '24h' ? 'active' : ''}
            onClick={() => setTimeRange('24h')}
          >
            24h
          </button>
          <button 
            className={timeRange === '7d' ? 'active' : ''}
            onClick={() => setTimeRange('7d')}
          >
            7d
          </button>
        </div>
      </div>
      
      <div className="graph-container">
        <canvas ref={chartRef} />
      </div>
    </section>
  );
};

// ============= UTILITY COMPONENTS =============

const UserAvatar: React.FC<{ user: User | null }> = ({ user }) => {
  if (!user) return <div className="user-avatar">?</div>;
  
  return (
    <div className="user-avatar" title={`${user.name} - Level ${user.level}`}>
      {user.avatar || user.name.charAt(0).toUpperCase()}
    </div>
  );
};

const WelcomeSection: React.FC<{ user: User | null }> = ({ user }) => {
  const [serverHealth, setServerHealth] = useState(92);
  
  return (
    <section className="welcome-section">
      <motion.h1 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        Welcome back, {user?.name || 'Guest'}! 🎉
      </motion.h1>
      <p className="server-health">
        Your server is <span className="health-percentage">{serverHealth}% healthy</span>
      </p>
    </section>
  );
};

const ServiceStatus: React.FC<{ 
  status: Service['status']; 
  health: number;
}> = ({ status, health }) => {
  const statusConfig = {
    running: { color: 'var(--accent-green)', text: 'Running' },
    stopped: { color: 'var(--accent-red)', text: 'Stopped' },
    warning: { color: 'var(--accent-yellow)', text: 'Warning' }
  };

  const config = statusConfig[status];

  return (
    <div className="service-status">
      <span 
        className="status-dot"
        style={{ backgroundColor: config.color }}
      />
      <span style={{ color: config.color }}>{config.text}</span>
    </div>
  );
};

const ServiceDetails: React.FC<{ service: Service }> = ({ service }) => {
  return (
    <div className="service-details-expanded">
      <div className="detail-grid">
        <div className="detail-item">
          <span className="detail-label">Container</span>
          <span className="detail-value">{service.container}</span>
        </div>
        <div className="detail-item">
          <span className="detail-label">Uptime</span>
          <span className="detail-value">{service.uptime}</span>
        </div>
        <div className="detail-item">
          <span className="detail-label">Connections</span>
          <span className="detail-value">{service.connections}</span>
        </div>
      </div>
      
      <div className="service-actions">
        <button className="action-btn">Restart</button>
        <button className="action-btn">Logs</button>
        <button className="action-btn">Shell</button>
        <button className="action-btn">Update</button>
      </div>
    </div>
  );
};

const ServiceList: React.FC<{ 
  services: Service[]; 
  simple?: boolean;
}> = ({ services, simple = false }) => {
  const handleToggle = (serviceId: string) => {
    // Toggle service logic
    console.log(`Toggling service: ${serviceId}`);
  };

  return (
    <section className="services-section">
      <h2 className="section-title">Essential Services</h2>
      <div className="service-list">
        {services.map((service, index) => (
          <motion.div
            key={service.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <ServiceCard 
              service={service} 
              simple={simple}
              onToggle={handleToggle}
            />
          </motion.div>
        ))}
      </div>
    </section>
  );
};

const ServiceMatrix: React.FC<{ services: Service[] }> = ({ services }) => {
  return (
    <section className="service-matrix">
      <h2>Service Matrix</h2>
      <table className="matrix-table">
        <thead>
          <tr>
            <th>Service</th>
            <th>Status</th>
            <th>CPU</th>
            <th>RAM</th>
            <th>Disk</th>
            <th>Net</th>
            <th>Health</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {services.map(service => (
            <ServiceMatrixRow key={service.id} service={service} />
          ))}
        </tbody>
      </table>
    </section>
  );
};

const ServiceMatrixRow: React.FC<{ service: Service }> = ({ service }) => {
  const [isOn, setIsOn] = useState(service.status === 'running');
  
  return (
    <motion.tr
      whileHover={{ backgroundColor: 'rgba(59, 130, 246, 0.1)' }}
    >
      <td className="service-name">{service.name}</td>
      <td><ServiceStatus status={service.status} health={service.health} /></td>
      <td>{service.cpu}%</td>
      <td>{service.memory}</td>
      <td>{service.disk}</td>
      <td>↓{service.network.download}</td>
      <td>
        <span className={`health-indicator ${service.health > 85 ? 'good' : 'warning'}`}>
          {service.health}%
        </span>
      </td>
      <td>
        <ToggleSwitch 
          active={isOn}
          onChange={() => setIsOn(!isOn)}
          size="small"
        />
      </td>
    </motion.tr>
  );
};

const AchievementSection: React.FC<{ achievements: Achievement[] }> = ({ achievements }) => {
  const recentAchievements = achievements
    .filter(a => a.unlocked)
    .sort((a, b) => (b.unlockedAt?.getTime() || 0) - (a.unlockedAt?.getTime() || 0))
    .slice(0, 5);

  const nextAchievement = achievements
    .filter(a => !a.unlocked)
    .sort((a, b) => (b.progress / b.maxProgress) - (a.progress / a.maxProgress))[0];

  return (
    <section className="achievements-section">
      <h2 className="section-title">Recent Achievements 🏆</h2>
      <div className="achievement-list">
        {recentAchievements.map((achievement, index) => (
          <motion.div
            key={achievement.id}
            className="achievement-badge"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: index * 0.1, type: "spring" }}
            whileHover={{ scale: 1.1, rotate: 5 }}
          >
            {achievement.icon}
          </motion.div>
        ))}
        
        {nextAchievement && (
          <div className="next-achievement">
            <span>→ Next: {nextAchievement.description}</span>
          </div>
        )}
      </div>
    </section>
  );
};

const AchievementsPanel: React.FC = () => {
  const achievements = [
    {
      name: 'Server Uptime Master',
      current: 95,
      max: 100,
      unit: 'hrs'
    },
    {
      name: 'Automation Wizard',
      current: 42,
      max: 50,
      unit: 'flows'
    }
  ];

  return (
    <section className="achievements-panel">
      <h2>Achievements</h2>
      <div className="achievement-progress">
        {achievements.map(achievement => (
          <div key={achievement.name} className="achievement-item">
            <div className="achievement-header">
              <span>{achievement.name}</span>
              <span>{achievement.current}/{achievement.max} {achievement.unit}</span>
            </div>
            <div className="achievement-bar">
              <motion.div 
                className="achievement-fill"
                initial={{ width: 0 }}
                animate={{ width: `${(achievement.current / achievement.max) * 100}%` }}
                transition={{ duration: 1, ease: "easeOut" }}
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};

// Export all components
export {
  Header,
  SimpleDashboard,
  AdvancedDashboard,
  ServiceCard,
  ServiceList,
  ServiceMatrix,
  GamificationBar,
  QuickActions,
  SystemOverview,
  ContainerOrchestra,
  PerformanceGraph,
  EnvConfiguration,
  AIEnvAssistant,
  AchievementSection,
  AchievementsPanel,
  ToggleSwitch,
  UserAvatar,
  WelcomeSection,
  ServiceStatus,
  ServiceDetails
};