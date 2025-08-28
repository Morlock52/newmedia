import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as THREE from 'three';
import './RealTimeMonitoringSystem.css';

interface ServiceMetric {
  id: string;
  name: string;
  status: 'online' | 'offline' | 'degraded' | 'maintenance';
  cpu: number;
  memory: number;
  disk: number;
  network: {
    in: number;
    out: number;
  };
  latency: number;
  uptime: number;
  errors: number;
  requests: number;
  lastUpdate: Date;
}

interface Alert {
  id: string;
  severity: 'critical' | 'warning' | 'info' | 'success';
  service: string;
  message: string;
  timestamp: Date;
  resolved: boolean;
}

interface SystemHealth {
  overall: number;
  services: number;
  performance: number;
  security: number;
  availability: number;
}

const RealTimeMonitoringSystem: React.FC = () => {
  const [services, setServices] = useState<ServiceMetric[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [systemHealth, setSystemHealth] = useState<SystemHealth>({
    overall: 100,
    services: 100,
    performance: 100,
    security: 100,
    availability: 100
  });
  const [selectedService, setSelectedService] = useState<ServiceMetric | null>(null);
  const [view, setView] = useState<'grid' | '3d' | 'matrix'>('grid');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000);
  const [showAlertPanel, setShowAlertPanel] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'reconnecting'>('connected');
  
  const wsRef = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const threeDRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const animationFrameRef = useRef<number>(0);

  useEffect(() => {
    initializeWebSocket();
    initializeServices();
    setup3DVisualization();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        fetchMetrics();
      }, refreshInterval);
      
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  const initializeWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8080/monitoring');
      
      wsRef.current.onopen = () => {
        setConnectionStatus('connected');
        console.log('WebSocket connected to monitoring system');
      };
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleRealtimeUpdate(data);
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('disconnected');
      };
      
      wsRef.current.onclose = () => {
        setConnectionStatus('reconnecting');
        setTimeout(() => initializeWebSocket(), 5000);
      };
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
      setConnectionStatus('disconnected');
    }
  };

  const handleRealtimeUpdate = (data: any) => {
    switch (data.type) {
      case 'metric_update':
        updateServiceMetric(data.payload);
        break;
      case 'alert':
        addAlert(data.payload);
        break;
      case 'health_update':
        setSystemHealth(data.payload);
        break;
      case 'service_status':
        updateServiceStatus(data.payload);
        break;
    }
  };

  const initializeServices = () => {
    // Initialize 30+ services
    const serviceList = [
      'Jellyfin', 'Plex', 'Emby', 'Sonarr', 'Radarr', 'Lidarr', 'Readarr',
      'Bazarr', 'Prowlarr', 'qBittorrent', 'SABnzbd', 'Transmission',
      'Overseerr', 'Jellyseerr', 'Tautulli', 'Organizr', 'Heimdall', 'Homer',
      'Portainer', 'Nginx Proxy Manager', 'Uptime Kuma', 'Grafana', 'Prometheus',
      'Watchtower', 'Duplicati', 'Nextcloud', 'Syncthing', 'FreshRSS',
      'Calibre-Web', 'PhotoPrism'
    ];
    
    const initialServices: ServiceMetric[] = serviceList.map((name, index) => ({
      id: `service-${index}`,
      name,
      status: Math.random() > 0.9 ? 'degraded' : 'online',
      cpu: Math.random() * 100,
      memory: Math.random() * 100,
      disk: Math.random() * 100,
      network: {
        in: Math.random() * 1000,
        out: Math.random() * 1000
      },
      latency: Math.random() * 100,
      uptime: Math.random() * 100,
      errors: Math.floor(Math.random() * 10),
      requests: Math.floor(Math.random() * 10000),
      lastUpdate: new Date()
    }));
    
    setServices(initialServices);
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch('/api/monitoring/metrics');
      const data = await response.json();
      setServices(data.services);
      setSystemHealth(data.health);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  };

  const updateServiceMetric = (update: Partial<ServiceMetric> & { id: string }) => {
    setServices(prev => prev.map(service => 
      service.id === update.id 
        ? { ...service, ...update, lastUpdate: new Date() }
        : service
    ));
  };

  const updateServiceStatus = (update: { id: string; status: string }) => {
    setServices(prev => prev.map(service => 
      service.id === update.id 
        ? { ...service, status: update.status as ServiceMetric['status'] }
        : service
    ));
  };

  const addAlert = (alert: Alert) => {
    setAlerts(prev => [alert, ...prev].slice(0, 50)); // Keep last 50 alerts
  };

  const setup3DVisualization = () => {
    if (!threeDRef.current) return;
    
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    scene.fog = new THREE.Fog(0x0a0a0a, 10, 100);
    
    const camera = new THREE.PerspectiveCamera(
      75,
      threeDRef.current.clientWidth / threeDRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 10, 30);
    camera.lookAt(0, 0, 0);
    
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(threeDRef.current.clientWidth, threeDRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    // Create service nodes
    const nodeGeometry = new THREE.SphereGeometry(0.5, 32, 32);
    const nodeGroup = new THREE.Group();
    
    for (let i = 0; i < 30; i++) {
      const material = new THREE.MeshPhongMaterial({
        color: new THREE.Color(`hsl(${180 + Math.random() * 60}, 100%, 50%)`),
        emissive: new THREE.Color(0x00ffff),
        emissiveIntensity: 0.2,
        transparent: true,
        opacity: 0.8
      });
      
      const node = new THREE.Mesh(nodeGeometry, material);
      const angle = (i / 30) * Math.PI * 2;
      const radius = 10 + Math.random() * 5;
      
      node.position.x = Math.cos(angle) * radius;
      node.position.z = Math.sin(angle) * radius;
      node.position.y = Math.random() * 5 - 2.5;
      
      nodeGroup.add(node);
    }
    
    scene.add(nodeGroup);
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
    
    const pointLight = new THREE.PointLight(0x00ffff, 1, 100);
    pointLight.position.set(0, 20, 0);
    scene.add(pointLight);
    
    // Add grid
    const gridHelper = new THREE.GridHelper(40, 40, 0x00ffff, 0x004444);
    scene.add(gridHelper);
    
    sceneRef.current = scene;
    rendererRef.current = renderer;
    
    const animate = () => {
      if (!sceneRef.current || !rendererRef.current) return;
      
      nodeGroup.rotation.y += 0.001;
      
      // Pulse nodes based on metrics
      nodeGroup.children.forEach((node, index) => {
        if (services[index]) {
          const scale = 1 + (services[index].cpu / 100) * 0.5;
          node.scale.setScalar(scale);
          
          // Update color based on status
          const mesh = node as THREE.Mesh;
          const material = mesh.material as THREE.MeshPhongMaterial;
          
          switch (services[index].status) {
            case 'online':
              material.emissive = new THREE.Color(0x00ff00);
              break;
            case 'degraded':
              material.emissive = new THREE.Color(0xffff00);
              break;
            case 'offline':
              material.emissive = new THREE.Color(0xff0000);
              break;
            case 'maintenance':
              material.emissive = new THREE.Color(0xff00ff);
              break;
          }
        }
      });
      
      renderer.render(scene, camera);
      animationFrameRef.current = requestAnimationFrame(animate);
    };
    
    animate();
  };

  const drawMatrixView = () => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Matrix rain effect
    const fontSize = 10;
    const columns = canvas.width / fontSize;
    const drops: number[] = [];
    
    for (let i = 0; i < columns; i++) {
      drops[i] = Math.random() * -100;
    }
    
    const drawMatrix = () => {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      ctx.fillStyle = '#00ff00';
      ctx.font = fontSize + 'px monospace';
      
      for (let i = 0; i < drops.length; i++) {
        const text = Math.random() > 0.5 ? '1' : '0';
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);
        
        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }
        drops[i]++;
      }
      
      // Overlay service status
      services.forEach((service, index) => {
        const x = (index % 6) * (canvas.width / 6) + 20;
        const y = Math.floor(index / 6) * 50 + 50;
        
        ctx.fillStyle = service.status === 'online' ? '#00ff00' : '#ff0000';
        ctx.font = 'bold 12px monospace';
        ctx.fillText(service.name.substring(0, 10), x, y);
        ctx.font = '10px monospace';
        ctx.fillText(`CPU: ${service.cpu.toFixed(0)}%`, x, y + 15);
      });
      
      requestAnimationFrame(drawMatrix);
    };
    
    drawMatrix();
  };

  const getStatusColor = (status: ServiceMetric['status']) => {
    switch (status) {
      case 'online': return '#00ff00';
      case 'offline': return '#ff0000';
      case 'degraded': return '#ffff00';
      case 'maintenance': return '#ff00ff';
      default: return '#666666';
    }
  };

  const getSeverityIcon = (severity: Alert['severity']) => {
    switch (severity) {
      case 'critical': return '🚨';
      case 'warning': return '⚠️';
      case 'info': return 'ℹ️';
      case 'success': return '✅';
      default: return '📝';
    }
  };

  const formatUptime = (uptime: number) => {
    const days = Math.floor(uptime / 86400);
    const hours = Math.floor((uptime % 86400) / 3600);
    return `${days}d ${hours}h`;
  };

  const resolveAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, resolved: true } : alert
    ));
  };

  const restartService = async (serviceId: string) => {
    try {
      await fetch(`/api/services/${serviceId}/restart`, { method: 'POST' });
      
      // Update status
      updateServiceStatus({ id: serviceId, status: 'maintenance' });
      
      setTimeout(() => {
        updateServiceStatus({ id: serviceId, status: 'online' });
      }, 5000);
    } catch (error) {
      console.error('Failed to restart service:', error);
    }
  };

  return (
    <div className="realtime-monitoring-system cyberpunk-theme">
      <div className="monitoring-header">
        <h1 className="title glitch-text" data-text="SYSTEM MONITORING">
          SYSTEM MONITORING
        </h1>
        
        <div className="connection-status">
          <div className={`status-dot ${connectionStatus}`}></div>
          <span>{connectionStatus.toUpperCase()}</span>
        </div>
        
        <div className="view-selector">
          <button 
            className={`view-btn ${view === 'grid' ? 'active' : ''}`}
            onClick={() => setView('grid')}
          >
            Grid View
          </button>
          <button 
            className={`view-btn ${view === '3d' ? 'active' : ''}`}
            onClick={() => setView('3d')}
          >
            3D View
          </button>
          <button 
            className={`view-btn ${view === 'matrix' ? 'active' : ''}`}
            onClick={() => {
              setView('matrix');
              setTimeout(() => drawMatrixView(), 100);
            }}
          >
            Matrix View
          </button>
        </div>
      </div>

      {/* System Health Overview */}
      <div className="health-overview">
        <div className="health-card">
          <div className="health-metric">
            <span className="metric-label">Overall Health</span>
            <div className="metric-value" style={{ color: systemHealth.overall > 80 ? '#00ff00' : '#ffff00' }}>
              {systemHealth.overall.toFixed(0)}%
            </div>
            <div className="health-bar">
              <div 
                className="health-fill"
                style={{ 
                  width: `${systemHealth.overall}%`,
                  background: `linear-gradient(90deg, #00ffff, ${systemHealth.overall > 80 ? '#00ff00' : '#ffff00'})`
                }}
              />
            </div>
          </div>
        </div>
        
        <div className="health-card">
          <div className="health-metric">
            <span className="metric-label">Services</span>
            <div className="metric-value">{services.filter(s => s.status === 'online').length}/{services.length}</div>
          </div>
        </div>
        
        <div className="health-card">
          <div className="health-metric">
            <span className="metric-label">Avg Latency</span>
            <div className="metric-value">
              {(services.reduce((sum, s) => sum + s.latency, 0) / services.length).toFixed(0)}ms
            </div>
          </div>
        </div>
        
        <div className="health-card">
          <div className="health-metric">
            <span className="metric-label">Active Alerts</span>
            <div className="metric-value" style={{ color: alerts.filter(a => !a.resolved).length > 0 ? '#ff0000' : '#00ff00' }}>
              {alerts.filter(a => !a.resolved).length}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="monitoring-content">
        {/* Services View */}
        <div className="services-container">
          {view === 'grid' && (
            <div className="services-grid">
              <AnimatePresence>
                {services.map((service, index) => (
                  <motion.div
                    key={service.id}
                    className="service-card"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    transition={{ delay: index * 0.02 }}
                    onClick={() => setSelectedService(service)}
                    style={{
                      borderColor: getStatusColor(service.status),
                      boxShadow: `0 0 20px ${getStatusColor(service.status)}40`
                    }}
                  >
                    <div className="service-header">
                      <div className="service-status">
                        <div 
                          className="status-indicator"
                          style={{ backgroundColor: getStatusColor(service.status) }}
                        />
                        <span className="service-name">{service.name}</span>
                      </div>
                      <button 
                        className="restart-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          restartService(service.id);
                        }}
                        title="Restart service"
                      >
                        🔄
                      </button>
                    </div>
                    
                    <div className="service-metrics">
                      <div className="metric">
                        <span className="metric-icon">💻</span>
                        <span className="metric-value">{service.cpu.toFixed(0)}%</span>
                      </div>
                      <div className="metric">
                        <span className="metric-icon">🧠</span>
                        <span className="metric-value">{service.memory.toFixed(0)}%</span>
                      </div>
                      <div className="metric">
                        <span className="metric-icon">💾</span>
                        <span className="metric-value">{service.disk.toFixed(0)}%</span>
                      </div>
                    </div>
                    
                    <div className="service-network">
                      <span className="network-in">↓ {(service.network.in / 1024).toFixed(1)} MB/s</span>
                      <span className="network-out">↑ {(service.network.out / 1024).toFixed(1)} MB/s</span>
                    </div>
                    
                    <div className="service-footer">
                      <span className="uptime">⏱ {formatUptime(service.uptime * 86400)}</span>
                      <span className="requests">{service.requests.toLocaleString()} req</span>
                    </div>
                    
                    {service.errors > 0 && (
                      <div className="error-badge">{service.errors} errors</div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          )}
          
          {view === '3d' && (
            <div ref={threeDRef} className="three-d-view" />
          )}
          
          {view === 'matrix' && (
            <canvas 
              ref={canvasRef}
              className="matrix-canvas"
              width={800}
              height={600}
            />
          )}
        </div>

        {/* Alerts Panel */}
        {showAlertPanel && (
          <div className="alerts-panel">
            <div className="panel-header">
              <h3>System Alerts</h3>
              <button 
                className="panel-toggle"
                onClick={() => setShowAlertPanel(false)}
              >
                ×
              </button>
            </div>
            
            <div className="alerts-list">
              <AnimatePresence>
                {alerts.map((alert, index) => (
                  <motion.div
                    key={alert.id}
                    className={`alert-item ${alert.severity} ${alert.resolved ? 'resolved' : ''}`}
                    initial={{ opacity: 0, x: 50 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -50 }}
                    transition={{ delay: index * 0.05 }}
                  >
                    <div className="alert-icon">
                      {getSeverityIcon(alert.severity)}
                    </div>
                    
                    <div className="alert-content">
                      <div className="alert-service">{alert.service}</div>
                      <div className="alert-message">{alert.message}</div>
                      <div className="alert-time">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                    
                    {!alert.resolved && (
                      <button
                        className="resolve-btn"
                        onClick={() => resolveAlert(alert.id)}
                      >
                        ✓
                      </button>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>
        )}
      </div>

      {/* Service Detail Modal */}
      <AnimatePresence>
        {selectedService && (
          <motion.div
            className="service-detail-modal"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
          >
            <div className="modal-content">
              <button 
                className="close-modal"
                onClick={() => setSelectedService(null)}
              >
                ×
              </button>
              
              <h2>{selectedService.name}</h2>
              
              <div className="detail-metrics">
                <div className="metric-chart">
                  <span className="chart-label">CPU Usage</span>
                  <div className="chart-bar">
                    <div 
                      className="chart-fill"
                      style={{ width: `${selectedService.cpu}%` }}
                    />
                  </div>
                  <span className="chart-value">{selectedService.cpu.toFixed(1)}%</span>
                </div>
                
                <div className="metric-chart">
                  <span className="chart-label">Memory Usage</span>
                  <div className="chart-bar">
                    <div 
                      className="chart-fill"
                      style={{ width: `${selectedService.memory}%` }}
                    />
                  </div>
                  <span className="chart-value">{selectedService.memory.toFixed(1)}%</span>
                </div>
                
                <div className="metric-chart">
                  <span className="chart-label">Disk Usage</span>
                  <div className="chart-bar">
                    <div 
                      className="chart-fill"
                      style={{ width: `${selectedService.disk}%` }}
                    />
                  </div>
                  <span className="chart-value">{selectedService.disk.toFixed(1)}%</span>
                </div>
              </div>
              
              <div className="detail-stats">
                <div className="stat">
                  <span className="stat-label">Status:</span>
                  <span className="stat-value" style={{ color: getStatusColor(selectedService.status) }}>
                    {selectedService.status.toUpperCase()}
                  </span>
                </div>
                <div className="stat">
                  <span className="stat-label">Latency:</span>
                  <span className="stat-value">{selectedService.latency.toFixed(0)}ms</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Uptime:</span>
                  <span className="stat-value">{formatUptime(selectedService.uptime * 86400)}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Requests:</span>
                  <span className="stat-value">{selectedService.requests.toLocaleString()}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Errors:</span>
                  <span className="stat-value" style={{ color: selectedService.errors > 0 ? '#ff0000' : '#00ff00' }}>
                    {selectedService.errors}
                  </span>
                </div>
                <div className="stat">
                  <span className="stat-label">Last Update:</span>
                  <span className="stat-value">
                    {new Date(selectedService.lastUpdate).toLocaleTimeString()}
                  </span>
                </div>
              </div>
              
              <div className="detail-actions">
                <button className="action-btn restart">Restart Service</button>
                <button className="action-btn logs">View Logs</button>
                <button className="action-btn config">Configuration</button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Settings Panel */}
      <div className="monitoring-settings">
        <label className="setting-item">
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.checked)}
          />
          <span>Auto Refresh</span>
        </label>
        
        <div className="setting-item">
          <label>Refresh Interval:</label>
          <select
            value={refreshInterval}
            onChange={(e) => setRefreshInterval(Number(e.target.value))}
            disabled={!autoRefresh}
          >
            <option value={1000}>1s</option>
            <option value={5000}>5s</option>
            <option value={10000}>10s</option>
            <option value={30000}>30s</option>
          </select>
        </div>
        
        {!showAlertPanel && (
          <button 
            className="show-alerts-btn"
            onClick={() => setShowAlertPanel(true)}
          >
            Show Alerts ({alerts.filter(a => !a.resolved).length})
          </button>
        )}
      </div>

      {/* Cyberpunk decorations */}
      <div className="cyberpunk-decorations">
        <div className="grid-overlay"></div>
        <div className="scan-lines"></div>
        <div className="data-stream"></div>
      </div>
    </div>
  );
};

export default RealTimeMonitoringSystem;