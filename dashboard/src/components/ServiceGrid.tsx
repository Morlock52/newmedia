import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Service {
  id: string;
  name: string;
  icon: string;
  status: 'online' | 'offline' | 'warning' | 'maintenance';
  url: string;
  port: number;
  category: 'media' | 'download' | 'automation' | 'monitoring' | 'utility';
  metrics: {
    cpu: number;
    memory: number;
    uptime: number;
    response: number;
  };
  version: string;
  lastCheck: Date;
}

interface GridSettings {
  layout: 'grid' | 'list' | 'compact';
  sortBy: 'name' | 'status' | 'category' | 'uptime';
  filterCategory: string;
  autoRefresh: boolean;
  showMetrics: boolean;
  gridSize: 'small' | 'medium' | 'large';
}

const ServiceGrid: React.FC = () => {
  const [services, setServices] = useState<Service[]>([]);
  const [settings, setSettings] = useState<GridSettings>({
    layout: 'grid',
    sortBy: 'status',
    filterCategory: 'all',
    autoRefresh: true,
    showMetrics: true,
    gridSize: 'medium'
  });
  const [selectedService, setSelectedService] = useState<Service | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [connectionTest, setConnectionTest] = useState<Record<string, boolean>>({});
  const [notifications, setNotifications] = useState<Array<{id: string, message: string, type: string}>>([]);
  
  const gridRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    initializeServices();
    setupWebSocket();
    if (settings.autoRefresh) {
      const interval = setInterval(refreshServices, 30000);
      return () => clearInterval(interval);
    }
  }, [settings.autoRefresh]);

  const initializeServices = () => {
    const serviceList: Service[] = [
      {
        id: 'jellyfin',
        name: 'Jellyfin Media Server',
        icon: '🎥',
        status: 'online',
        url: 'http://localhost:8096',
        port: 8096,
        category: 'media',
        metrics: { cpu: 35, memory: 45, uptime: 99.8, response: 120 },
        version: '10.8.13',
        lastCheck: new Date()
      },
      {
        id: 'plex',
        name: 'Plex Media Server',
        icon: '📺',
        status: 'online',
        url: 'http://localhost:32400',
        port: 32400,
        category: 'media',
        metrics: { cpu: 42, memory: 55, uptime: 99.9, response: 85 },
        version: '1.32.8',
        lastCheck: new Date()
      },
      {
        id: 'sonarr',
        name: 'Sonarr TV Automation',
        icon: '📡',
        status: 'online',
        url: 'http://localhost:8989',
        port: 8989,
        category: 'automation',
        metrics: { cpu: 15, memory: 25, uptime: 99.9, response: 200 },
        version: '4.0.2',
        lastCheck: new Date()
      },
      {
        id: 'radarr',
        name: 'Radarr Movie Automation',
        icon: '🎦',
        status: 'online',
        url: 'http://localhost:7878',
        port: 7878,
        category: 'automation',
        metrics: { cpu: 12, memory: 22, uptime: 99.7, response: 180 },
        version: '5.2.6',
        lastCheck: new Date()
      },
      {
        id: 'prowlarr',
        name: 'Prowlarr Indexer Manager',
        icon: '🔍',
        status: 'warning',
        url: 'http://localhost:9696',
        port: 9696,
        category: 'automation',
        metrics: { cpu: 8, memory: 18, uptime: 98.5, response: 350 },
        version: '1.11.4',
        lastCheck: new Date()
      },
      {
        id: 'qbittorrent',
        name: 'qBittorrent',
        icon: '⬇️',
        status: 'online',
        url: 'http://localhost:8080',
        port: 8080,
        category: 'download',
        metrics: { cpu: 25, memory: 30, uptime: 99.5, response: 95 },
        version: '4.6.2',
        lastCheck: new Date()
      },
      {
        id: 'sabnzbd',
        name: 'SABnzbd',
        icon: '📦',
        status: 'online',
        url: 'http://localhost:8085',
        port: 8085,
        category: 'download',
        metrics: { cpu: 18, memory: 28, uptime: 99.2, response: 110 },
        version: '4.0.3',
        lastCheck: new Date()
      },
      {
        id: 'grafana',
        name: 'Grafana Dashboard',
        icon: '📈',
        status: 'online',
        url: 'http://localhost:3000',
        port: 3000,
        category: 'monitoring',
        metrics: { cpu: 10, memory: 15, uptime: 99.9, response: 75 },
        version: '10.2.3',
        lastCheck: new Date()
      },
      {
        id: 'portainer',
        name: 'Portainer CE',
        icon: '🐳',
        status: 'online',
        url: 'http://localhost:9000',
        port: 9000,
        category: 'utility',
        metrics: { cpu: 5, memory: 12, uptime: 99.8, response: 60 },
        version: '2.19.4',
        lastCheck: new Date()
      },
      {
        id: 'uptime-kuma',
        name: 'Uptime Kuma',
        icon: '👨‍⚕️',
        status: 'offline',
        url: 'http://localhost:3001',
        port: 3001,
        category: 'monitoring',
        metrics: { cpu: 0, memory: 0, uptime: 0, response: 0 },
        version: '1.23.11',
        lastCheck: new Date()
      }
    ];
    
    setServices(serviceList);
  };

  const setupWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8080/services/status');
      
      wsRef.current.onmessage = (event) => {
        const update = JSON.parse(event.data);
        handleServiceUpdate(update);
      };
      
      wsRef.current.onerror = () => {
        console.warn('WebSocket connection failed - using polling fallback');
      };
    } catch (error) {
      console.warn('WebSocket not available');
    }
  };

  const handleServiceUpdate = (update: any) => {
    setServices(prev => prev.map(service => 
      service.id === update.id ? { ...service, ...update } : service
    ));
  };

  const refreshServices = async () => {
    setIsLoading(true);
    
    // Simulate service health checks
    const updatedServices = await Promise.all(
      services.map(async (service) => {
        const isOnline = Math.random() > 0.1; // 90% uptime simulation
        const newMetrics = {
          cpu: Math.max(0, Math.min(100, service.metrics.cpu + (Math.random() - 0.5) * 10)),
          memory: Math.max(0, Math.min(100, service.metrics.memory + (Math.random() - 0.5) * 8)),
          uptime: isOnline ? Math.min(100, service.metrics.uptime + 0.1) : 0,
          response: Math.max(50, service.metrics.response + (Math.random() - 0.5) * 50)
        };
        
        return {
          ...service,
          status: isOnline ? 'online' : 'offline' as const,
          metrics: newMetrics,
          lastCheck: new Date()
        };
      })
    );
    
    setServices(updatedServices);
    setIsLoading(false);
  };

  const testConnection = async (service: Service) => {
    setConnectionTest(prev => ({ ...prev, [service.id]: true }));
    
    try {
      // Simulate connection test
      await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
      const success = Math.random() > 0.2;
      
      addNotification({
        id: Date.now().toString(),
        message: `${service.name}: ${success ? 'Connection successful' : 'Connection failed'}`,
        type: success ? 'success' : 'error'
      });
      
      if (success) {
        setServices(prev => prev.map(s => 
          s.id === service.id ? { ...s, status: 'online', lastCheck: new Date() } : s
        ));
      }
    } catch (error) {
      addNotification({
        id: Date.now().toString(),
        message: `${service.name}: Connection test failed`,
        type: 'error'
      });
    } finally {
      setConnectionTest(prev => ({ ...prev, [service.id]: false }));
    }
  };

  const addNotification = (notification: {id: string, message: string, type: string}) => {
    setNotifications(prev => [...prev, notification]);
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== notification.id));
    }, 5000);
  };

  const filteredServices = services
    .filter(service => 
      (settings.filterCategory === 'all' || service.category === settings.filterCategory) &&
      (searchQuery === '' || 
        service.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        service.category.toLowerCase().includes(searchQuery.toLowerCase())
      )
    )
    .sort((a, b) => {
      switch (settings.sortBy) {
        case 'status':
          const statusOrder = { online: 4, warning: 3, offline: 2, maintenance: 1 };
          return statusOrder[b.status] - statusOrder[a.status];
        case 'category':
          return a.category.localeCompare(b.category);
        case 'uptime':
          return b.metrics.uptime - a.metrics.uptime;
        default:
          return a.name.localeCompare(b.name);
      }
    });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return '#00FF00';
      case 'offline': return '#FF0040';
      case 'warning': return '#FFFF00';
      case 'maintenance': return '#FF00FF';
      default: return '#666666';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'media': return '#00FFFF';
      case 'download': return '#FF00FF';
      case 'automation': return '#FFFF00';
      case 'monitoring': return '#00FF00';
      case 'utility': return '#FF8800';
      default: return '#FFFFFF';
    }
  };

  const gridSizeClasses = {
    small: 'min-w-[200px]',
    medium: 'min-w-[280px]',
    large: 'min-w-[350px]'
  };

  return (
    <div style={{
      background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%)',
      color: '#00FFFF',
      fontFamily: 'Orbitron, monospace',
      minHeight: '100vh',
      padding: '20px'
    }}>
      {/* Header */}
      <motion.header
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '30px',
          flexWrap: 'wrap',
          gap: '20px'
        }}
      >
        <div>
          <h1 style={{
            fontSize: '3rem',
            margin: 0,
            background: 'linear-gradient(45deg, #00FFFF, #FF00FF, #FFFF00)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            textShadow: '0 0 20px #00FFFF',
            animation: 'textGlow 3s ease-in-out infinite alternate'
          }}>
            SERVICE GRID
          </h1>
          <div style={{ display: 'flex', gap: '20px', marginTop: '10px', fontSize: '0.9rem' }}>
            <span style={{ color: '#00FF00' }}>Online: {services.filter(s => s.status === 'online').length}</span>
            <span style={{ color: '#FFFF00' }}>Warning: {services.filter(s => s.status === 'warning').length}</span>
            <span style={{ color: '#FF0040' }}>Offline: {services.filter(s => s.status === 'offline').length}</span>
            <span style={{ color: '#FF00FF' }}>Total: {services.length}</span>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <button
            onClick={refreshServices}
            disabled={isLoading}
            style={{
              padding: '10px 20px',
              background: 'linear-gradient(45deg, #00FFFF, #FF00FF)',
              border: 'none',
              borderRadius: '8px',
              color: '#000',
              fontWeight: 'bold',
              cursor: 'pointer'
            }}
          >
            {isLoading ? 'Refreshing...' : '🔄 Refresh'}
          </button>
        </div>
      </motion.header>

      {/* Controls */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '15px',
          marginBottom: '30px',
          padding: '20px',
          background: 'rgba(0,0,0,0.7)',
          border: '2px solid #00FFFF',
          borderRadius: '12px'
        }}
      >
        <input
          type="text"
          placeholder="Search services..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{
            flex: '1 1 300px',
            padding: '12px',
            background: 'rgba(0,0,0,0.8)',
            border: '2px solid #FF00FF',
            borderRadius: '8px',
            color: '#00FFFF',
            fontSize: '1rem'
          }}
        />
        
        <select
          value={settings.filterCategory}
          onChange={(e) => setSettings(prev => ({ ...prev, filterCategory: e.target.value }))}
          style={{
            padding: '12px',
            background: 'rgba(0,0,0,0.8)',
            border: '2px solid #FFFF00',
            borderRadius: '8px',
            color: '#00FFFF',
            fontSize: '1rem'
          }}
        >
          <option value="all">All Categories</option>
          <option value="media">Media</option>
          <option value="download">Download</option>
          <option value="automation">Automation</option>
          <option value="monitoring">Monitoring</option>
          <option value="utility">Utility</option>
        </select>
        
        <select
          value={settings.sortBy}
          onChange={(e) => setSettings(prev => ({ ...prev, sortBy: e.target.value as any }))}
          style={{
            padding: '12px',
            background: 'rgba(0,0,0,0.8)',
            border: '2px solid #FF00FF',
            borderRadius: '8px',
            color: '#00FFFF',
            fontSize: '1rem'
          }}
        >
          <option value="name">Sort by Name</option>
          <option value="status">Sort by Status</option>
          <option value="category">Sort by Category</option>
          <option value="uptime">Sort by Uptime</option>
        </select>
        
        <div style={{ display: 'flex', gap: '10px' }}>
          {['small', 'medium', 'large'].map(size => (
            <button
              key={size}
              onClick={() => setSettings(prev => ({ ...prev, gridSize: size as any }))}
              style={{
                padding: '8px 15px',
                background: settings.gridSize === size ? 'linear-gradient(45deg, #FFFF00, #FF00FF)' : 'rgba(255,255,0,0.2)',
                border: '1px solid #FFFF00',
                borderRadius: '6px',
                color: settings.gridSize === size ? '#000' : '#FFFF00',
                cursor: 'pointer',
                fontSize: '0.9rem',
                textTransform: 'capitalize'
              }}
            >
              {size}
            </button>
          ))}
        </div>
      </motion.section>

      {/* Service Grid */}
      <motion.div
        ref={gridRef}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(auto-fit, minmax(${settings.gridSize === 'small' ? '250px' : settings.gridSize === 'medium' ? '320px' : '400px'}, 1fr))`,
          gap: '20px',
          marginBottom: '30px'
        }}
      >
        <AnimatePresence>
          {filteredServices.map((service, index) => (
            <motion.div
              key={service.id}
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: -20 }}
              transition={{ delay: index * 0.05 }}
              whileHover={{ scale: 1.05, y: -5 }}
              style={{
                background: 'rgba(0,0,0,0.8)',
                border: `3px solid ${getStatusColor(service.status)}`,
                borderRadius: '15px',
                padding: '25px',
                cursor: 'pointer',
                position: 'relative',
                overflow: 'hidden',
                boxShadow: `0 10px 30px rgba(${service.status === 'online' ? '0,255,0' : service.status === 'warning' ? '255,255,0' : '255,0,64'},0.3)`
              }}
              onClick={() => setSelectedService(service)}
            >
              {/* Status Indicator */}
              <div style={{
                position: 'absolute',
                top: '15px',
                right: '15px',
                width: '15px',
                height: '15px',
                background: getStatusColor(service.status),
                borderRadius: '50%',
                boxShadow: `0 0 15px ${getStatusColor(service.status)}`,
                animation: service.status === 'online' ? 'pulse 2s infinite' : 'none'
              }} />
              
              {/* Service Header */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '20px' }}>
                <div style={{
                  fontSize: '3rem',
                  filter: 'drop-shadow(0 0 10px currentColor)'
                }}>
                  {service.icon}
                </div>
                <div style={{ flex: 1 }}>
                  <h3 style={{
                    margin: 0,
                    color: '#00FFFF',
                    fontSize: '1.3rem',
                    fontWeight: 'bold'
                  }}>
                    {service.name}
                  </h3>
                  <div style={{ display: 'flex', gap: '10px', marginTop: '5px' }}>
                    <span style={{
                      padding: '3px 8px',
                      background: `rgba(${getCategoryColor(service.category) === '#00FFFF' ? '0,255,255' : getCategoryColor(service.category) === '#FF00FF' ? '255,0,255' : getCategoryColor(service.category) === '#FFFF00' ? '255,255,0' : '0,255,0'},0.2)`,
                      border: `1px solid ${getCategoryColor(service.category)}`,
                      borderRadius: '10px',
                      fontSize: '0.8rem',
                      color: getCategoryColor(service.category),
                      textTransform: 'capitalize'
                    }}>
                      {service.category}
                    </span>
                    <span style={{
                      fontSize: '0.9rem',
                      opacity: 0.7,
                      color: '#FFFF00'
                    }}>
                      v{service.version}
                    </span>
                  </div>
                </div>
              </div>
              
              {/* Metrics */}
              {settings.showMetrics && (
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '12px',
                  marginBottom: '20px'
                }}>
                  {[
                    { label: 'CPU', value: service.metrics.cpu, unit: '%', color: '#FF00FF' },
                    { label: 'Memory', value: service.metrics.memory, unit: '%', color: '#FFFF00' },
                    { label: 'Uptime', value: service.metrics.uptime, unit: '%', color: '#00FF00' },
                    { label: 'Response', value: service.metrics.response, unit: 'ms', color: '#00FFFF' }
                  ].map(metric => (
                    <div key={metric.label} style={{ textAlign: 'center' }}>
                      <div style={{
                        fontSize: '1.2rem',
                        fontWeight: 'bold',
                        color: metric.color
                      }}>
                        {typeof metric.value === 'number' ? metric.value.toFixed(1) : metric.value}{metric.unit}
                      </div>
                      <div style={{
                        fontSize: '0.8rem',
                        opacity: 0.7
                      }}>
                        {metric.label}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              
              {/* Action Buttons */}
              <div style={{
                display: 'flex',
                gap: '10px',
                justifyContent: 'space-between'
              }}>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    window.open(service.url, '_blank');
                  }}
                  style={{
                    flex: 1,
                    padding: '8px 12px',
                    background: 'linear-gradient(45deg, #00FFFF, #FF00FF)',
                    border: 'none',
                    borderRadius: '6px',
                    color: '#000',
                    fontWeight: 'bold',
                    cursor: 'pointer',
                    fontSize: '0.9rem'
                  }}
                >
                  Open
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    testConnection(service);
                  }}
                  disabled={connectionTest[service.id]}
                  style={{
                    flex: 1,
                    padding: '8px 12px',
                    background: connectionTest[service.id] ? 'rgba(255,255,0,0.3)' : 'rgba(255,255,0,0.1)',
                    border: '1px solid #FFFF00',
                    borderRadius: '6px',
                    color: '#FFFF00',
                    cursor: 'pointer',
                    fontSize: '0.9rem'
                  }}
                >
                  {connectionTest[service.id] ? 'Testing...' : 'Test'}
                </button>
              </div>
              
              {/* Last Check */}
              <div style={{
                marginTop: '15px',
                fontSize: '0.8rem',
                opacity: 0.6,
                textAlign: 'center'
              }}>
                Last check: {service.lastCheck.toLocaleTimeString()}
              </div>
              
              {/* Port Badge */}
              <div style={{
                position: 'absolute',
                bottom: '10px',
                left: '15px',
                fontSize: '0.8rem',
                color: '#666',
                background: 'rgba(0,0,0,0.8)',
                padding: '2px 6px',
                borderRadius: '4px'
              }}>
                :{service.port}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </motion.div>

      {/* Notifications */}
      <div style={{
        position: 'fixed',
        top: '20px',
        right: '20px',
        zIndex: 1000,
        display: 'flex',
        flexDirection: 'column',
        gap: '10px'
      }}>
        <AnimatePresence>
          {notifications.map(notification => (
            <motion.div
              key={notification.id}
              initial={{ opacity: 0, x: 300 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 300 }}
              style={{
                padding: '15px 20px',
                background: notification.type === 'success' ? 
                  'linear-gradient(45deg, #00FF00, #00FFFF)' : 
                  'linear-gradient(45deg, #FF0040, #FF00FF)',
                color: '#000',
                borderRadius: '8px',
                fontWeight: 'bold',
                maxWidth: '300px',
                boxShadow: '0 5px 20px rgba(0,0,0,0.5)'
              }}
            >
              {notification.message}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      <style jsx>{`
        @keyframes textGlow {
          0% { text-shadow: 0 0 20px #00FFFF, 0 0 40px #00FFFF; }
          100% { text-shadow: 0 0 30px #FF00FF, 0 0 60px #FF00FF; }
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.6; }
        }
      `}</style>
    </div>
  );
};

export default ServiceGrid;