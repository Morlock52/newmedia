import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './ServiceGridDashboard.css';

interface ServiceCard {
  id: string;
  name: string;
  icon: string;
  status: 'online' | 'offline' | 'degraded' | 'maintenance';
  type: 'media' | 'download' | 'management' | 'monitoring' | 'storage' | 'utility';
  url: string;
  metrics: {
    cpu: number;
    memory: number;
    disk: number;
    uptime: number;
    requests: number;
    errors: number;
  };
  features: string[];
  lastUpdate: Date;
  version?: string;
  port?: number;
}

interface QuickAction {
  id: string;
  label: string;
  icon: string;
  action: () => void;
  color: string;
}

const ServiceGridDashboard: React.FC = () => {
  const [services, setServices] = useState<ServiceCard[]>([]);
  const [selectedService, setSelectedService] = useState<ServiceCard | null>(null);
  const [filterType, setFilterType] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list' | 'compact'>('grid');
  const [sortBy, setSortBy] = useState<'name' | 'status' | 'cpu' | 'memory'>('name');
  const [showMetrics, setShowMetrics] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000);
  
  const gridRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    initializeServices();
    setupWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        updateServiceMetrics();
      }, refreshInterval);
      
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  const initializeServices = () => {
    const serviceList: ServiceCard[] = [
      // Media Servers
      {
        id: 'jellyfin',
        name: 'Jellyfin',
        icon: '🎬',
        status: 'online',
        type: 'media',
        url: 'http://localhost:8096',
        port: 8096,
        version: '10.8.13',
        metrics: { cpu: 45, memory: 60, disk: 35, uptime: 99.9, requests: 15420, errors: 3 },
        features: ['Streaming', 'Transcoding', 'Libraries', 'Users'],
        lastUpdate: new Date()
      },
      {
        id: 'plex',
        name: 'Plex',
        icon: '📺',
        status: 'online',
        type: 'media',
        url: 'http://localhost:32400',
        port: 32400,
        version: '1.32.8',
        metrics: { cpu: 55, memory: 70, disk: 40, uptime: 99.8, requests: 18900, errors: 0 },
        features: ['Streaming', 'Live TV', 'DVR', 'Mobile Sync'],
        lastUpdate: new Date()
      },
      {
        id: 'emby',
        name: 'Emby',
        icon: '🎭',
        status: 'degraded',
        type: 'media',
        url: 'http://localhost:8096',
        port: 8096,
        version: '4.7.14',
        metrics: { cpu: 60, memory: 65, disk: 38, uptime: 95.2, requests: 8200, errors: 12 },
        features: ['Streaming', 'Live TV', 'Cinema Mode'],
        lastUpdate: new Date()
      },
      
      // Content Management
      {
        id: 'sonarr',
        name: 'Sonarr',
        icon: '📡',
        status: 'online',
        type: 'management',
        url: 'http://localhost:8989',
        port: 8989,
        version: '4.0.2',
        metrics: { cpu: 30, memory: 40, disk: 15, uptime: 99.9, requests: 5200, errors: 0 },
        features: ['TV Shows', 'Calendar', 'Automation', 'Indexers'],
        lastUpdate: new Date()
      },
      {
        id: 'radarr',
        name: 'Radarr',
        icon: '🎥',
        status: 'online',
        type: 'management',
        url: 'http://localhost:7878',
        port: 7878,
        version: '5.2.6',
        metrics: { cpu: 25, memory: 35, disk: 12, uptime: 99.9, requests: 4800, errors: 0 },
        features: ['Movies', 'Lists', 'Automation', 'Quality'],
        lastUpdate: new Date()
      },
      {
        id: 'lidarr',
        name: 'Lidarr',
        icon: '🎵',
        status: 'online',
        type: 'management',
        url: 'http://localhost:8686',
        port: 8686,
        version: '2.0.7',
        metrics: { cpu: 20, memory: 30, disk: 10, uptime: 99.8, requests: 2100, errors: 1 },
        features: ['Music', 'Artists', 'Automation', 'Metadata'],
        lastUpdate: new Date()
      },
      {
        id: 'readarr',
        name: 'Readarr',
        icon: '📚',
        status: 'offline',
        type: 'management',
        url: 'http://localhost:8787',
        port: 8787,
        version: '0.3.18',
        metrics: { cpu: 0, memory: 0, disk: 0, uptime: 0, requests: 0, errors: 50 },
        features: ['Books', 'Authors', 'Series', 'Automation'],
        lastUpdate: new Date()
      },
      {
        id: 'bazarr',
        name: 'Bazarr',
        icon: '💬',
        status: 'online',
        type: 'management',
        url: 'http://localhost:6767',
        port: 6767,
        version: '1.4.0',
        metrics: { cpu: 15, memory: 25, disk: 5, uptime: 99.7, requests: 3200, errors: 2 },
        features: ['Subtitles', 'Languages', 'Providers', 'Automation'],
        lastUpdate: new Date()
      },
      {
        id: 'prowlarr',
        name: 'Prowlarr',
        icon: '🔍',
        status: 'online',
        type: 'management',
        url: 'http://localhost:9696',
        port: 9696,
        version: '1.11.4',
        metrics: { cpu: 35, memory: 45, disk: 8, uptime: 99.9, requests: 10500, errors: 0 },
        features: ['Indexers', 'Search', 'Sync', 'Statistics'],
        lastUpdate: new Date()
      },
      
      // Download Clients
      {
        id: 'qbittorrent',
        name: 'qBittorrent',
        icon: '⬇️',
        status: 'online',
        type: 'download',
        url: 'http://localhost:8080',
        port: 8080,
        version: '4.6.2',
        metrics: { cpu: 70, memory: 80, disk: 60, uptime: 99.5, requests: 8900, errors: 0 },
        features: ['Torrents', 'RSS', 'Search', 'Categories'],
        lastUpdate: new Date()
      },
      {
        id: 'sabnzbd',
        name: 'SABnzbd',
        icon: '📦',
        status: 'online',
        type: 'download',
        url: 'http://localhost:8085',
        port: 8085,
        version: '4.0.3',
        metrics: { cpu: 50, memory: 55, disk: 45, uptime: 99.3, requests: 4200, errors: 3 },
        features: ['Usenet', 'NZB', 'Categories', 'Scripts'],
        lastUpdate: new Date()
      },
      {
        id: 'transmission',
        name: 'Transmission',
        icon: '🔄',
        status: 'maintenance',
        type: 'download',
        url: 'http://localhost:9091',
        port: 9091,
        version: '4.0.5',
        metrics: { cpu: 0, memory: 0, disk: 0, uptime: 0, requests: 0, errors: 0 },
        features: ['Torrents', 'Web UI', 'Remote', 'Encryption'],
        lastUpdate: new Date()
      },
      
      // Request Management
      {
        id: 'overseerr',
        name: 'Overseerr',
        icon: '🎯',
        status: 'online',
        type: 'utility',
        url: 'http://localhost:5055',
        port: 5055,
        version: '1.33.2',
        metrics: { cpu: 20, memory: 30, disk: 5, uptime: 99.8, requests: 6200, errors: 0 },
        features: ['Requests', 'Discovery', 'Users', 'Notifications'],
        lastUpdate: new Date()
      },
      {
        id: 'jellyseerr',
        name: 'Jellyseerr',
        icon: '🌟',
        status: 'online',
        type: 'utility',
        url: 'http://localhost:5055',
        port: 5055,
        version: '1.7.0',
        metrics: { cpu: 18, memory: 28, disk: 4, uptime: 99.7, requests: 5100, errors: 1 },
        features: ['Requests', 'Jellyfin', 'Users', 'Discovery'],
        lastUpdate: new Date()
      },
      
      // Monitoring
      {
        id: 'tautulli',
        name: 'Tautulli',
        icon: '📊',
        status: 'online',
        type: 'monitoring',
        url: 'http://localhost:8181',
        port: 8181,
        version: '2.13.4',
        metrics: { cpu: 20, memory: 30, disk: 10, uptime: 99.9, requests: 7200, errors: 0 },
        features: ['Statistics', 'History', 'Notifications', 'Newsletters'],
        lastUpdate: new Date()
      },
      {
        id: 'grafana',
        name: 'Grafana',
        icon: '📈',
        status: 'online',
        type: 'monitoring',
        url: 'http://localhost:3000',
        port: 3000,
        version: '10.2.3',
        metrics: { cpu: 25, memory: 35, disk: 8, uptime: 99.9, requests: 12000, errors: 0 },
        features: ['Dashboards', 'Alerts', 'Datasources', 'Plugins'],
        lastUpdate: new Date()
      },
      {
        id: 'prometheus',
        name: 'Prometheus',
        icon: '🔥',
        status: 'online',
        type: 'monitoring',
        url: 'http://localhost:9090',
        port: 9090,
        version: '2.48.1',
        metrics: { cpu: 40, memory: 50, disk: 20, uptime: 99.9, requests: 50000, errors: 0 },
        features: ['Metrics', 'Queries', 'Alerts', 'Targets'],
        lastUpdate: new Date()
      },
      {
        id: 'uptimekuma',
        name: 'Uptime Kuma',
        icon: '🏥',
        status: 'online',
        type: 'monitoring',
        url: 'http://localhost:3001',
        port: 3001,
        version: '1.23.11',
        metrics: { cpu: 15, memory: 20, disk: 3, uptime: 99.9, requests: 3600, errors: 0 },
        features: ['Monitors', 'Status Pages', 'Notifications', 'Certificates'],
        lastUpdate: new Date()
      },
      
      // Organization
      {
        id: 'organizr',
        name: 'Organizr',
        icon: '🏠',
        status: 'online',
        type: 'utility',
        url: 'http://localhost:80',
        port: 80,
        version: '2.1.2',
        metrics: { cpu: 10, memory: 15, disk: 2, uptime: 99.8, requests: 4500, errors: 0 },
        features: ['Dashboard', 'Tabs', 'Auth', 'Themes'],
        lastUpdate: new Date()
      },
      {
        id: 'heimdall',
        name: 'Heimdall',
        icon: '🛡️',
        status: 'online',
        type: 'utility',
        url: 'http://localhost:8080',
        port: 8080,
        version: '2.5.8',
        metrics: { cpu: 8, memory: 12, disk: 1, uptime: 99.7, requests: 2100, errors: 0 },
        features: ['Dashboard', 'Apps', 'Search', 'Tags'],
        lastUpdate: new Date()
      },
      
      // Container Management
      {
        id: 'portainer',
        name: 'Portainer',
        icon: '🐳',
        status: 'online',
        type: 'utility',
        url: 'http://localhost:9000',
        port: 9000,
        version: '2.19.4',
        metrics: { cpu: 30, memory: 40, disk: 5, uptime: 99.9, requests: 8900, errors: 0 },
        features: ['Containers', 'Stacks', 'Networks', 'Volumes'],
        lastUpdate: new Date()
      },
      
      // Storage
      {
        id: 'nextcloud',
        name: 'Nextcloud',
        icon: '☁️',
        status: 'online',
        type: 'storage',
        url: 'http://localhost:8080',
        port: 8080,
        version: '28.0.1',
        metrics: { cpu: 35, memory: 45, disk: 70, uptime: 99.5, requests: 7800, errors: 5 },
        features: ['Files', 'Calendar', 'Contacts', 'Apps'],
        lastUpdate: new Date()
      },
      {
        id: 'syncthing',
        name: 'Syncthing',
        icon: '🔄',
        status: 'online',
        type: 'storage',
        url: 'http://localhost:8384',
        port: 8384,
        version: '1.27.2',
        metrics: { cpu: 30, memory: 40, disk: 50, uptime: 99.6, requests: 4300, errors: 2 },
        features: ['Sync', 'Folders', 'Devices', 'Versioning'],
        lastUpdate: new Date()
      },
      {
        id: 'duplicati',
        name: 'Duplicati',
        icon: '💾',
        status: 'online',
        type: 'storage',
        url: 'http://localhost:8200',
        port: 8200,
        version: '2.0.8',
        metrics: { cpu: 25, memory: 30, disk: 40, uptime: 99.4, requests: 1200, errors: 0 },
        features: ['Backup', 'Restore', 'Schedule', 'Encryption'],
        lastUpdate: new Date()
      },
      
      // Other Services
      {
        id: 'nginxproxymanager',
        name: 'Nginx Proxy Manager',
        icon: '🔐',
        status: 'online',
        type: 'utility',
        url: 'http://localhost:81',
        port: 81,
        version: '2.11.1',
        metrics: { cpu: 15, memory: 20, disk: 2, uptime: 99.9, requests: 25000, errors: 0 },
        features: ['Proxy', 'SSL', 'Access Lists', 'Streams'],
        lastUpdate: new Date()
      },
      {
        id: 'watchtower',
        name: 'Watchtower',
        icon: '🔔',
        status: 'online',
        type: 'utility',
        url: 'http://localhost:8080',
        port: 8080,
        version: '1.5.3',
        metrics: { cpu: 5, memory: 10, disk: 1, uptime: 99.9, requests: 500, errors: 0 },
        features: ['Updates', 'Notifications', 'Schedule', 'Cleanup'],
        lastUpdate: new Date()
      },
      {
        id: 'freshrss',
        name: 'FreshRSS',
        icon: '📰',
        status: 'online',
        type: 'utility',
        url: 'http://localhost:80',
        port: 80,
        version: '1.23.1',
        metrics: { cpu: 12, memory: 18, disk: 3, uptime: 99.6, requests: 3400, errors: 1 },
        features: ['RSS', 'Feeds', 'Categories', 'Sharing'],
        lastUpdate: new Date()
      },
      {
        id: 'calibreweb',
        name: 'Calibre-Web',
        icon: '📖',
        status: 'online',
        type: 'utility',
        url: 'http://localhost:8083',
        port: 8083,
        version: '0.6.21',
        metrics: { cpu: 18, memory: 25, disk: 15, uptime: 99.5, requests: 2800, errors: 2 },
        features: ['eBooks', 'Library', 'Reader', 'Conversion'],
        lastUpdate: new Date()
      },
      {
        id: 'photoprism',
        name: 'PhotoPrism',
        icon: '📷',
        status: 'online',
        type: 'storage',
        url: 'http://localhost:2342',
        port: 2342,
        version: '231128',
        metrics: { cpu: 45, memory: 55, disk: 80, uptime: 99.3, requests: 5600, errors: 4 },
        features: ['Photos', 'AI', 'Albums', 'Sharing'],
        lastUpdate: new Date()
      }
    ];
    
    setServices(serviceList);
  };

  const setupWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8080/services');
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServiceUpdate(data);
      };
    } catch (error) {
      console.error('WebSocket connection failed:', error);
    }
  };

  const handleServiceUpdate = (update: any) => {
    setServices(prev => prev.map(service => 
      service.id === update.id 
        ? { ...service, ...update, lastUpdate: new Date() }
        : service
    ));
  };

  const updateServiceMetrics = () => {
    setServices(prev => prev.map(service => ({
      ...service,
      metrics: {
        ...service.metrics,
        cpu: Math.max(0, Math.min(100, service.metrics.cpu + (Math.random() - 0.5) * 10)),
        memory: Math.max(0, Math.min(100, service.metrics.memory + (Math.random() - 0.5) * 10)),
        requests: service.metrics.requests + Math.floor(Math.random() * 100)
      },
      lastUpdate: new Date()
    })));
  };

  const getQuickActions = (): QuickAction[] => [
    {
      id: 'restart-all',
      label: 'Restart All',
      icon: '🔄',
      action: () => console.log('Restarting all services'),
      color: '#00ffff'
    },
    {
      id: 'update-all',
      label: 'Update All',
      icon: '⬆️',
      action: () => console.log('Updating all services'),
      color: '#ff00ff'
    },
    {
      id: 'backup',
      label: 'Backup',
      icon: '💾',
      action: () => console.log('Creating backup'),
      color: '#ffff00'
    },
    {
      id: 'scan-media',
      label: 'Scan Media',
      icon: '🔍',
      action: () => console.log('Scanning media libraries'),
      color: '#00ff00'
    }
  ];

  const filteredServices = services
    .filter(service => filterType === 'all' || service.type === filterType)
    .filter(service => 
      service.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      service.features.some(f => f.toLowerCase().includes(searchQuery.toLowerCase()))
    )
    .sort((a, b) => {
      switch (sortBy) {
        case 'status':
          return a.status.localeCompare(b.status);
        case 'cpu':
          return b.metrics.cpu - a.metrics.cpu;
        case 'memory':
          return b.metrics.memory - a.metrics.memory;
        default:
          return a.name.localeCompare(b.name);
      }
    });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return '#00ff00';
      case 'offline': return '#ff0000';
      case 'degraded': return '#ffff00';
      case 'maintenance': return '#ff00ff';
      default: return '#666666';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'media': return '#00ffff';
      case 'download': return '#ff00ff';
      case 'management': return '#ffff00';
      case 'monitoring': return '#00ff00';
      case 'storage': return '#ff8800';
      case 'utility': return '#8888ff';
      default: return '#ffffff';
    }
  };

  const handleServiceClick = (service: ServiceCard) => {
    setSelectedService(service);
  };

  const handleServiceAction = (action: string, serviceId: string) => {
    console.log(`Executing ${action} on ${serviceId}`);
    // Implement service actions
  };

  return (
    <div className="service-grid-dashboard cyberpunk-theme">
      <div className="dashboard-header">
        <div className="header-title">
          <h1 className="title glitch-text" data-text="SERVICE NEXUS">
            SERVICE NEXUS
          </h1>
          <div className="service-stats">
            <span className="stat">
              <span className="stat-value">{services.filter(s => s.status === 'online').length}</span>
              <span className="stat-label">Online</span>
            </span>
            <span className="stat">
              <span className="stat-value">{services.length}</span>
              <span className="stat-label">Total</span>
            </span>
            <span className="stat">
              <span className="stat-value">
                {Math.round(services.reduce((sum, s) => sum + s.metrics.cpu, 0) / services.length)}%
              </span>
              <span className="stat-label">Avg CPU</span>
            </span>
          </div>
        </div>
        
        <div className="header-controls">
          <div className="search-box">
            <input
              type="text"
              placeholder="Search services..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
            />
            <span className="search-icon">🔍</span>
          </div>
          
          <div className="view-controls">
            <button
              className={`view-btn ${viewMode === 'grid' ? 'active' : ''}`}
              onClick={() => setViewMode('grid')}
              title="Grid View"
            >
              ⊞
            </button>
            <button
              className={`view-btn ${viewMode === 'list' ? 'active' : ''}`}
              onClick={() => setViewMode('list')}
              title="List View"
            >
              ☰
            </button>
            <button
              className={`view-btn ${viewMode === 'compact' ? 'active' : ''}`}
              onClick={() => setViewMode('compact')}
              title="Compact View"
            >
              ⋮
            </button>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="quick-actions">
        {getQuickActions().map(action => (
          <button
            key={action.id}
            className="quick-action-btn"
            onClick={action.action}
            style={{ borderColor: action.color }}
          >
            <span className="action-icon">{action.icon}</span>
            <span className="action-label">{action.label}</span>
          </button>
        ))}
      </div>

      {/* Filter Bar */}
      <div className="filter-bar">
        <div className="filter-tabs">
          <button
            className={`filter-tab ${filterType === 'all' ? 'active' : ''}`}
            onClick={() => setFilterType('all')}
          >
            All Services
          </button>
          <button
            className={`filter-tab ${filterType === 'media' ? 'active' : ''}`}
            onClick={() => setFilterType('media')}
          >
            Media
          </button>
          <button
            className={`filter-tab ${filterType === 'download' ? 'active' : ''}`}
            onClick={() => setFilterType('download')}
          >
            Download
          </button>
          <button
            className={`filter-tab ${filterType === 'management' ? 'active' : ''}`}
            onClick={() => setFilterType('management')}
          >
            Management
          </button>
          <button
            className={`filter-tab ${filterType === 'monitoring' ? 'active' : ''}`}
            onClick={() => setFilterType('monitoring')}
          >
            Monitoring
          </button>
          <button
            className={`filter-tab ${filterType === 'storage' ? 'active' : ''}`}
            onClick={() => setFilterType('storage')}
          >
            Storage
          </button>
          <button
            className={`filter-tab ${filterType === 'utility' ? 'active' : ''}`}
            onClick={() => setFilterType('utility')}
          >
            Utilities
          </button>
        </div>
        
        <div className="sort-controls">
          <label>Sort by:</label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="sort-select"
          >
            <option value="name">Name</option>
            <option value="status">Status</option>
            <option value="cpu">CPU Usage</option>
            <option value="memory">Memory Usage</option>
          </select>
        </div>
      </div>

      {/* Service Grid */}
      <div className={`service-grid ${viewMode}`} ref={gridRef}>
        <AnimatePresence>
          {filteredServices.map((service, index) => (
            <motion.div
              key={service.id}
              className="service-card"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ delay: index * 0.02 }}
              onClick={() => handleServiceClick(service)}
              style={{
                borderColor: getStatusColor(service.status),
                '--type-color': getTypeColor(service.type)
              } as any}
            >
              <div className="card-header">
                <div className="service-icon">{service.icon}</div>
                <div className="service-info">
                  <h3 className="service-name">{service.name}</h3>
                  <span className="service-version">v{service.version}</span>
                </div>
                <div className={`status-badge ${service.status}`}>
                  {service.status}
                </div>
              </div>
              
              {showMetrics && viewMode !== 'compact' && (
                <div className="card-metrics">
                  <div className="metric-row">
                    <div className="metric">
                      <span className="metric-label">CPU</span>
                      <div className="metric-bar">
                        <div 
                          className="metric-fill cpu"
                          style={{ width: `${service.metrics.cpu}%` }}
                        />
                      </div>
                      <span className="metric-value">{service.metrics.cpu}%</span>
                    </div>
                  </div>
                  
                  <div className="metric-row">
                    <div className="metric">
                      <span className="metric-label">Memory</span>
                      <div className="metric-bar">
                        <div 
                          className="metric-fill memory"
                          style={{ width: `${service.metrics.memory}%` }}
                        />
                      </div>
                      <span className="metric-value">{service.metrics.memory}%</span>
                    </div>
                  </div>
                  
                  {viewMode === 'grid' && (
                    <div className="metric-row">
                      <div className="metric">
                        <span className="metric-label">Disk</span>
                        <div className="metric-bar">
                          <div 
                            className="metric-fill disk"
                            style={{ width: `${service.metrics.disk}%` }}
                          />
                        </div>
                        <span className="metric-value">{service.metrics.disk}%</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              {viewMode === 'grid' && (
                <>
                  <div className="card-features">
                    {service.features.slice(0, 3).map(feature => (
                      <span key={feature} className="feature-tag">{feature}</span>
                    ))}
                    {service.features.length > 3 && (
                      <span className="feature-tag">+{service.features.length - 3}</span>
                    )}
                  </div>
                  
                  <div className="card-stats">
                    <span className="stat-item">
                      <span className="stat-icon">📊</span>
                      {service.metrics.requests.toLocaleString()} req
                    </span>
                    <span className="stat-item">
                      <span className="stat-icon">⚠️</span>
                      {service.metrics.errors} err
                    </span>
                    <span className="stat-item">
                      <span className="stat-icon">⏱️</span>
                      {service.metrics.uptime}%
                    </span>
                  </div>
                </>
              )}
              
              <div className="card-actions">
                <button 
                  className="action-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    window.open(service.url, '_blank');
                  }}
                  title="Open"
                >
                  🔗
                </button>
                <button 
                  className="action-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleServiceAction('restart', service.id);
                  }}
                  title="Restart"
                >
                  🔄
                </button>
                <button 
                  className="action-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleServiceAction('logs', service.id);
                  }}
                  title="Logs"
                >
                  📋
                </button>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Service Detail Modal */}
      <AnimatePresence>
        {selectedService && (
          <motion.div
            className="service-detail-modal"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedService(null)}
          >
            <motion.div
              className="modal-content"
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              onClick={(e) => e.stopPropagation()}
            >
              <button 
                className="close-modal"
                onClick={() => setSelectedService(null)}
              >
                ×
              </button>
              
              <div className="modal-header">
                <span className="modal-icon">{selectedService.icon}</span>
                <div>
                  <h2>{selectedService.name}</h2>
                  <p>Version {selectedService.version} | Port {selectedService.port}</p>
                </div>
                <div className={`modal-status ${selectedService.status}`}>
                  {selectedService.status.toUpperCase()}
                </div>
              </div>
              
              <div className="modal-metrics">
                <h3>Performance Metrics</h3>
                <div className="metrics-grid">
                  <div className="metric-card">
                    <span className="metric-title">CPU Usage</span>
                    <span className="metric-big">{selectedService.metrics.cpu}%</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-title">Memory Usage</span>
                    <span className="metric-big">{selectedService.metrics.memory}%</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-title">Disk Usage</span>
                    <span className="metric-big">{selectedService.metrics.disk}%</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-title">Uptime</span>
                    <span className="metric-big">{selectedService.metrics.uptime}%</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-title">Total Requests</span>
                    <span className="metric-big">{selectedService.metrics.requests.toLocaleString()}</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-title">Errors</span>
                    <span className="metric-big">{selectedService.metrics.errors}</span>
                  </div>
                </div>
              </div>
              
              <div className="modal-features">
                <h3>Features</h3>
                <div className="features-list">
                  {selectedService.features.map(feature => (
                    <span key={feature} className="feature-badge">{feature}</span>
                  ))}
                </div>
              </div>
              
              <div className="modal-actions">
                <button className="modal-action-btn primary">
                  Open Dashboard
                </button>
                <button className="modal-action-btn">
                  Restart Service
                </button>
                <button className="modal-action-btn">
                  View Logs
                </button>
                <button className="modal-action-btn">
                  Settings
                </button>
              </div>
              
              <div className="modal-footer">
                <span>Last Updated: {new Date(selectedService.lastUpdate).toLocaleString()}</span>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Settings Panel */}
      <div className="settings-panel">
        <label className="setting-item">
          <input
            type="checkbox"
            checked={showMetrics}
            onChange={(e) => setShowMetrics(e.target.checked)}
          />
          <span>Show Metrics</span>
        </label>
        
        <label className="setting-item">
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.checked)}
          />
          <span>Auto Refresh</span>
        </label>
        
        {autoRefresh && (
          <div className="setting-item">
            <label>Refresh Interval:</label>
            <select
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
              className="setting-select"
            >
              <option value={5000}>5s</option>
              <option value={10000}>10s</option>
              <option value={30000}>30s</option>
              <option value={60000}>1m</option>
            </select>
          </div>
        )}
      </div>
    </div>
  );
};

export default ServiceGridDashboard;