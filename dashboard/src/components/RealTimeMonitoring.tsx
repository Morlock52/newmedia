import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Line, Bar, Doughnut, Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  RadialLinearScale,
  Filler
} from 'chart.js';
import io from 'socket.io-client';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  RadialLinearScale,
  Filler
);

interface SystemMetrics {
  timestamp: number;
  cpu: number;
  memory: number;
  disk: number;
  network: {
    upload: number;
    download: number;
  };
  temperature: number;
  power: number;
}

interface ServiceStatus {
  id: string;
  name: string;
  status: 'online' | 'offline' | 'degraded' | 'maintenance';
  uptime: number;
  responseTime: number;
  errorRate: number;
  lastCheck: number;
  version: string;
  port: number;
  health: number; // 0-100
}

interface Alert {
  id: string;
  type: 'critical' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: number;
  acknowledged: boolean;
  service?: string;
  metric?: string;
  value?: number;
  threshold?: number;
}

interface NetworkConnection {
  id: string;
  source: string;
  target: string;
  type: 'api' | 'websocket' | 'database' | 'external';
  status: 'active' | 'idle' | 'error';
  latency: number;
  throughput: number;
}

interface RealTimeMonitoringProps {
  autoRefresh?: boolean;
  refreshInterval?: number;
  enableAlerts?: boolean;
  compactMode?: boolean;
}

const RealTimeMonitoring: React.FC<RealTimeMonitoringProps> = ({
  autoRefresh = true,
  refreshInterval = 1000,
  enableAlerts = true,
  compactMode = false
}) => {
  // State management
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics[]>([]);
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [networkConnections, setNetworkConnections] = useState<NetworkConnection[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1m' | '5m' | '1h' | '24h'>('5m');
  const [selectedMetric, setSelectedMetric] = useState<'cpu' | 'memory' | 'disk' | 'network'>('cpu');
  const [activeView, setActiveView] = useState<'overview' | 'services' | 'alerts' | 'network'>('overview');
  const [showAdvancedMetrics, setShowAdvancedMetrics] = useState(false);
  const [anomalyDetection, setAnomalyDetection] = useState(true);

  // Refs
  const socketRef = useRef<any>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const chartRef = useRef<any>(null);

  // WebSocket connection setup
  useEffect(() => {
    if (autoRefresh) {
      socketRef.current = io('ws://localhost:3001', {
        transports: ['websocket', 'polling']
      });

      socketRef.current.on('connect', () => {
        setIsConnected(true);
        console.log('Connected to monitoring WebSocket');
      });

      socketRef.current.on('disconnect', () => {
        setIsConnected(false);
        console.log('Disconnected from monitoring WebSocket');
      });

      socketRef.current.on('system-metrics', (data: SystemMetrics) => {
        setSystemMetrics(prev => {
          const newMetrics = [...prev, data].slice(-100); // Keep last 100 points
          return newMetrics;
        });
      });

      socketRef.current.on('service-status', (data: ServiceStatus[]) => {
        setServices(data);
      });

      socketRef.current.on('alert', (alert: Alert) => {
        if (enableAlerts) {
          setAlerts(prev => [alert, ...prev].slice(0, 50)); // Keep last 50 alerts
        }
      });

      socketRef.current.on('network-status', (data: NetworkConnection[]) => {
        setNetworkConnections(data);
      });

      return () => {
        socketRef.current?.disconnect();
      };
    }
  }, [autoRefresh, enableAlerts]);

  // Fallback data fetching when WebSocket is not available
  useEffect(() => {
    if (!isConnected && autoRefresh) {
      const interval = setInterval(() => {
        fetchSystemMetrics();
        fetchServiceStatus();
        fetchAlerts();
        fetchNetworkStatus();
      }, refreshInterval);

      return () => clearInterval(interval);
    }
  }, [isConnected, autoRefresh, refreshInterval]);

  // Real-time visualization animation
  useEffect(() => {
    if (systemMetrics.length > 0) {
      animateNetworkGraph();
    }
  }, [systemMetrics, networkConnections]);

  const fetchSystemMetrics = async () => {
    try {
      // Simulate system metrics
      const metrics: SystemMetrics = {
        timestamp: Date.now(),
        cpu: Math.random() * 100,
        memory: 60 + Math.random() * 30,
        disk: 45 + Math.random() * 20,
        network: {
          upload: Math.random() * 1000,
          download: Math.random() * 5000
        },
        temperature: 35 + Math.random() * 20,
        power: 80 + Math.random() * 20
      };

      setSystemMetrics(prev => [...prev, metrics].slice(-100));
    } catch (error) {
      console.error('Failed to fetch system metrics:', error);
    }
  };

  const fetchServiceStatus = async () => {
    try {
      // Simulate service status
      const mockServices: ServiceStatus[] = [
        {
          id: 'jellyfin',
          name: 'Jellyfin Media Server',
          status: Math.random() > 0.1 ? 'online' : 'degraded',
          uptime: Date.now() - (Math.random() * 86400000),
          responseTime: 50 + Math.random() * 200,
          errorRate: Math.random() * 5,
          lastCheck: Date.now(),
          version: '10.8.13',
          port: 8096,
          health: 80 + Math.random() * 20
        },
        {
          id: 'sonarr',
          name: 'Sonarr',
          status: Math.random() > 0.05 ? 'online' : 'offline',
          uptime: Date.now() - (Math.random() * 86400000),
          responseTime: 30 + Math.random() * 150,
          errorRate: Math.random() * 3,
          lastCheck: Date.now(),
          version: '4.0.2',
          port: 8989,
          health: 90 + Math.random() * 10
        },
        {
          id: 'radarr',
          name: 'Radarr',
          status: 'online',
          uptime: Date.now() - (Math.random() * 86400000),
          responseTime: 40 + Math.random() * 120,
          errorRate: Math.random() * 2,
          lastCheck: Date.now(),
          version: '5.3.6',
          port: 7878,
          health: 85 + Math.random() * 15
        },
        {
          id: 'prowlarr',
          name: 'Prowlarr',
          status: 'online',
          uptime: Date.now() - (Math.random() * 86400000),
          responseTime: 60 + Math.random() * 180,
          errorRate: Math.random() * 4,
          lastCheck: Date.now(),
          version: '1.12.2',
          port: 9696,
          health: 75 + Math.random() * 25
        },
        {
          id: 'qbittorrent',
          name: 'qBittorrent',
          status: 'online',
          uptime: Date.now() - (Math.random() * 86400000),
          responseTime: 25 + Math.random() * 100,
          errorRate: Math.random() * 1,
          lastCheck: Date.now(),
          version: '4.6.3',
          port: 8080,
          health: 95 + Math.random() * 5
        }
      ];

      setServices(mockServices);
    } catch (error) {
      console.error('Failed to fetch service status:', error);
    }
  };

  const fetchAlerts = async () => {
    try {
      // Generate random alerts
      if (Math.random() > 0.95) {
        const alertTypes = ['critical', 'warning', 'info'] as const;
        const newAlert: Alert = {
          id: Date.now().toString(),
          type: alertTypes[Math.floor(Math.random() * alertTypes.length)],
          title: 'System Alert',
          message: 'High CPU usage detected on media server',
          timestamp: Date.now(),
          acknowledged: false,
          service: 'jellyfin',
          metric: 'cpu',
          value: 95,
          threshold: 90
        };

        setAlerts(prev => [newAlert, ...prev].slice(0, 50));
      }
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
    }
  };

  const fetchNetworkStatus = async () => {
    try {
      const connections: NetworkConnection[] = [
        {
          id: 'jellyfin-api',
          source: 'Dashboard',
          target: 'Jellyfin',
          type: 'api',
          status: 'active',
          latency: 10 + Math.random() * 50,
          throughput: Math.random() * 1000
        },
        {
          id: 'sonarr-websocket',
          source: 'Dashboard',
          target: 'Sonarr',
          type: 'websocket',
          status: 'active',
          latency: 15 + Math.random() * 30,
          throughput: Math.random() * 500
        },
        {
          id: 'database-connection',
          source: 'API',
          target: 'Database',
          type: 'database',
          status: 'active',
          latency: 5 + Math.random() * 20,
          throughput: Math.random() * 2000
        }
      ];

      setNetworkConnections(connections);
    } catch (error) {
      console.error('Failed to fetch network status:', error);
    }
  };

  const animateNetworkGraph = () => {
    if (!canvasRef.current || networkConnections.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw network connections
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = 80;

    // Draw central hub
    ctx.beginPath();
    ctx.arc(centerX, centerY, 20, 0, Math.PI * 2);
    ctx.fillStyle = '#ff00ff';
    ctx.fill();
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 3;
    ctx.stroke();

    // Draw nodes and connections
    networkConnections.forEach((conn, index) => {
      const angle = (index / networkConnections.length) * Math.PI * 2;
      const nodeX = centerX + Math.cos(angle) * radius;
      const nodeY = centerY + Math.sin(angle) * radius;

      // Draw connection line
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(nodeX, nodeY);
      ctx.strokeStyle = conn.status === 'active' ? '#00ff00' : 
                       conn.status === 'idle' ? '#ffff00' : '#ff0000';
      ctx.lineWidth = 2 + (conn.throughput / 1000) * 3;
      ctx.stroke();

      // Draw node
      ctx.beginPath();
      ctx.arc(nodeX, nodeY, 10, 0, Math.PI * 2);
      ctx.fillStyle = conn.status === 'active' ? '#00ff00' : 
                     conn.status === 'idle' ? '#ffff00' : '#ff0000';
      ctx.fill();

      // Draw latency indicator
      ctx.fillStyle = '#ffffff';
      ctx.font = '8px monospace';
      ctx.fillText(
        `${conn.latency.toFixed(0)}ms`,
        nodeX - 15,
        nodeY + 25
      );
    });

    animationRef.current = requestAnimationFrame(animateNetworkGraph);
  };

  const acknowledgeAlert = (alertId: string) => {
    setAlerts(prev => 
      prev.map(alert => 
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      )
    );
  };

  const clearAlert = (alertId: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return '#00ff00';
      case 'offline': return '#ff0000';
      case 'degraded': return '#ffff00';
      case 'maintenance': return '#ff00ff';
      default: return '#cccccc';
    }
  };

  const formatUptime = (uptime: number) => {
    const seconds = Math.floor(uptime / 1000);
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  // Chart configurations
  const systemChartData = {
    labels: systemMetrics.slice(-20).map((_, index) => index),
    datasets: [
      {
        label: 'CPU %',
        data: systemMetrics.slice(-20).map(m => m.cpu),
        borderColor: '#00ffff',
        backgroundColor: 'rgba(0, 255, 255, 0.1)',
        fill: true,
        tension: 0.4
      },
      {
        label: 'Memory %',
        data: systemMetrics.slice(-20).map(m => m.memory),
        borderColor: '#ff00ff',
        backgroundColor: 'rgba(255, 0, 255, 0.1)',
        fill: true,
        tension: 0.4
      },
      {
        label: 'Disk %',
        data: systemMetrics.slice(-20).map(m => m.disk),
        borderColor: '#ffff00',
        backgroundColor: 'rgba(255, 255, 0, 0.1)',
        fill: true,
        tension: 0.4
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: '#ffffff'
        }
      }
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: '#ffffff'
        }
      },
      y: {
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: '#ffffff'
        },
        min: 0,
        max: 100
      }
    }
  };

  return (
    <div className="real-time-monitoring" style={{
      background: 'linear-gradient(135deg, rgba(0,0,0,0.95) 0%, rgba(20,20,40,0.95) 100%)',
      border: '2px solid #00ffff',
      borderRadius: '15px',
      padding: compactMode ? '10px' : '20px',
      color: '#ffffff',
      fontFamily: 'monospace',
      minHeight: compactMode ? '400px' : '600px',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Cyberpunk background effects */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: `
          repeating-linear-gradient(
            90deg,
            transparent,
            transparent 2px,
            rgba(0, 255, 255, 0.02) 2px,
            rgba(0, 255, 255, 0.02) 4px
          ),
          repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(255, 0, 255, 0.02) 2px,
            rgba(255, 0, 255, 0.02) 4px
          )
        `,
        pointerEvents: 'none'
      }} />

      {/* Header */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        marginBottom: '20px',
        position: 'relative',
        zIndex: 2
      }}>
        <h2 style={{
          color: '#00ffff',
          textShadow: '0 0 10px #00ffff, 0 0 20px #00ffff',
          margin: 0,
          fontSize: compactMode ? '18px' : '24px'
        }}>
          📊 REAL-TIME MONITORING
        </h2>

        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          {/* Connection Status */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '5px 10px',
            background: isConnected ? 'rgba(0,255,0,0.2)' : 'rgba(255,0,0,0.2)',
            border: `1px solid ${isConnected ? '#00ff00' : '#ff0000'}`,
            borderRadius: '15px',
            fontSize: '11px'
          }}>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: isConnected ? '#00ff00' : '#ff0000',
              animation: isConnected ? 'pulse 2s infinite' : 'none'
            }} />
            {isConnected ? 'LIVE' : 'OFFLINE'}
          </div>

          {/* View Selector */}
          <div style={{ display: 'flex', gap: '5px' }}>
            {(['overview', 'services', 'alerts', 'network'] as const).map(view => (
              <button
                key={view}
                onClick={() => setActiveView(view)}
                style={{
                  padding: '5px 10px',
                  background: activeView === view ? 'rgba(0,255,255,0.3)' : 'rgba(0,0,0,0.3)',
                  border: `1px solid ${activeView === view ? '#00ffff' : '#666'}`,
                  borderRadius: '5px',
                  color: activeView === view ? '#00ffff' : '#cccccc',
                  cursor: 'pointer',
                  fontSize: '10px',
                  textTransform: 'uppercase'
                }}
              >
                {view}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content based on active view */}
      <div style={{ position: 'relative', zIndex: 2 }}>
        {activeView === 'overview' && (
          <div style={{ display: 'grid', gridTemplateColumns: compactMode ? '1fr' : '1fr 1fr', gap: '20px' }}>
            {/* System Metrics Chart */}
            <div style={{
              background: 'rgba(0,0,0,0.5)',
              border: '1px solid rgba(0,255,255,0.3)',
              borderRadius: '10px',
              padding: '15px',
              height: '300px'
            }}>
              <h3 style={{ color: '#ffff00', fontSize: '14px', marginBottom: '10px' }}>
                System Performance
              </h3>
              <Line data={systemChartData} options={chartOptions} />
            </div>

            {/* Network Visualization */}
            <div style={{
              background: 'rgba(0,0,0,0.5)',
              border: '1px solid rgba(255,0,255,0.3)',
              borderRadius: '10px',
              padding: '15px',
              height: '300px'
            }}>
              <h3 style={{ color: '#ff00ff', fontSize: '14px', marginBottom: '10px' }}>
                Network Topology
              </h3>
              <canvas
                ref={canvasRef}
                width={250}
                height={200}
                style={{ width: '100%', height: 'calc(100% - 30px)' }}
              />
            </div>
          </div>
        )}

        {activeView === 'services' && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: compactMode ? '1fr' : 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '15px'
          }}>
            <AnimatePresence>
              {services.map((service, index) => (
                <motion.div
                  key={service.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  style={{
                    background: 'linear-gradient(135deg, rgba(0,255,255,0.1) 0%, rgba(255,0,255,0.1) 100%)',
                    border: `2px solid ${getStatusColor(service.status)}`,
                    borderRadius: '10px',
                    padding: '15px',
                    boxShadow: `0 0 20px ${getStatusColor(service.status)}30`
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                    <h4 style={{ color: '#ffffff', margin: 0, fontSize: '14px' }}>
                      {service.name}
                    </h4>
                    <div style={{
                      padding: '2px 8px',
                      background: getStatusColor(service.status) + '30',
                      border: `1px solid ${getStatusColor(service.status)}`,
                      borderRadius: '10px',
                      fontSize: '10px',
                      textTransform: 'uppercase'
                    }}>
                      {service.status}
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '11px' }}>
                    <div>
                      <span style={{ color: '#cccccc' }}>Uptime:</span>
                      <span style={{ color: '#00ff00', marginLeft: '5px' }}>
                        {formatUptime(service.uptime)}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: '#cccccc' }}>Port:</span>
                      <span style={{ color: '#ffff00', marginLeft: '5px' }}>
                        {service.port}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: '#cccccc' }}>Response:</span>
                      <span style={{ color: '#ff00ff', marginLeft: '5px' }}>
                        {service.responseTime.toFixed(0)}ms
                      </span>
                    </div>
                    <div>
                      <span style={{ color: '#cccccc' }}>Health:</span>
                      <span style={{ color: '#00ffff', marginLeft: '5px' }}>
                        {service.health.toFixed(0)}%
                      </span>
                    </div>
                  </div>

                  {/* Health bar */}
                  <div style={{
                    width: '100%',
                    height: '4px',
                    background: 'rgba(255,255,255,0.1)',
                    borderRadius: '2px',
                    marginTop: '10px',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      width: `${service.health}%`,
                      height: '100%',
                      background: `linear-gradient(90deg, ${getStatusColor(service.status)}, #00ffff)`,
                      borderRadius: '2px',
                      transition: 'width 0.3s ease'
                    }} />
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}

        {activeView === 'alerts' && (
          <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
            <AnimatePresence>
              {alerts.map((alert, index) => (
                <motion.div
                  key={alert.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ delay: index * 0.05 }}
                  style={{
                    background: alert.acknowledged ? 'rgba(128,128,128,0.2)' : 
                               alert.type === 'critical' ? 'rgba(255,0,0,0.2)' :
                               alert.type === 'warning' ? 'rgba(255,255,0,0.2)' :
                               'rgba(0,255,255,0.2)',
                    border: `1px solid ${
                      alert.type === 'critical' ? '#ff0000' :
                      alert.type === 'warning' ? '#ffff00' :
                      '#00ffff'
                    }`,
                    borderRadius: '8px',
                    padding: '15px',
                    marginBottom: '10px',
                    opacity: alert.acknowledged ? 0.6 : 1
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '5px' }}>
                        <span style={{
                          fontSize: '16px'
                        }}>
                          {alert.type === 'critical' ? '🚨' : alert.type === 'warning' ? '⚠️' : 'ℹ️'}
                        </span>
                        <h4 style={{ margin: 0, fontSize: '14px', color: '#ffffff' }}>
                          {alert.title}
                        </h4>
                        <span style={{
                          fontSize: '10px',
                          color: '#cccccc'
                        }}>
                          {new Date(alert.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <p style={{ margin: '5px 0', fontSize: '12px', color: '#cccccc' }}>
                        {alert.message}
                      </p>
                      {alert.service && (
                        <span style={{
                          fontSize: '10px',
                          color: '#ffff00',
                          background: 'rgba(255,255,0,0.2)',
                          padding: '2px 6px',
                          borderRadius: '3px'
                        }}>
                          {alert.service}
                        </span>
                      )}
                    </div>
                    <div style={{ display: 'flex', gap: '5px' }}>
                      {!alert.acknowledged && (
                        <button
                          onClick={() => acknowledgeAlert(alert.id)}
                          style={{
                            padding: '5px 10px',
                            background: 'rgba(0,255,0,0.2)',
                            border: '1px solid #00ff00',
                            borderRadius: '3px',
                            color: '#00ff00',
                            cursor: 'pointer',
                            fontSize: '10px'
                          }}
                        >
                          ACK
                        </button>
                      )}
                      <button
                        onClick={() => clearAlert(alert.id)}
                        style={{
                          padding: '5px 10px',
                          background: 'rgba(255,0,0,0.2)',
                          border: '1px solid #ff0000',
                          borderRadius: '3px',
                          color: '#ff0000',
                          cursor: 'pointer',
                          fontSize: '10px'
                        }}
                      >
                        ✕
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            {alerts.length === 0 && (
              <div style={{
                textAlign: 'center',
                color: '#cccccc',
                fontSize: '14px',
                padding: '50px'
              }}>
                No alerts at the moment. System is running smoothly! 🎉
              </div>
            )}
          </div>
        )}

        {activeView === 'network' && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: compactMode ? '1fr' : '2fr 1fr',
            gap: '20px'
          }}>
            {/* Network Graph */}
            <div style={{
              background: 'rgba(0,0,0,0.5)',
              border: '1px solid rgba(0,255,255,0.3)',
              borderRadius: '10px',
              padding: '15px',
              height: '400px'
            }}>
              <h3 style={{ color: '#00ffff', fontSize: '14px', marginBottom: '10px' }}>
                Network Activity
              </h3>
              <canvas
                ref={canvasRef}
                width={400}
                height={300}
                style={{ width: '100%', height: 'calc(100% - 30px)' }}
              />
            </div>

            {/* Connection Details */}
            <div style={{
              background: 'rgba(0,0,0,0.5)',
              border: '1px solid rgba(255,0,255,0.3)',
              borderRadius: '10px',
              padding: '15px',
              height: '400px',
              overflowY: 'auto'
            }}>
              <h3 style={{ color: '#ff00ff', fontSize: '14px', marginBottom: '10px' }}>
                Active Connections
              </h3>
              {networkConnections.map((conn, index) => (
                <div
                  key={conn.id}
                  style={{
                    border: `1px solid ${getStatusColor(conn.status)}`,
                    borderRadius: '5px',
                    padding: '10px',
                    marginBottom: '10px',
                    fontSize: '11px'
                  }}
                >
                  <div style={{ color: '#ffffff', fontWeight: 'bold', marginBottom: '5px' }}>
                    {conn.source} → {conn.target}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '5px' }}>
                    <div>
                      <span style={{ color: '#cccccc' }}>Type:</span>
                      <span style={{ color: '#ffff00', marginLeft: '5px' }}>
                        {conn.type}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: '#cccccc' }}>Status:</span>
                      <span style={{ color: getStatusColor(conn.status), marginLeft: '5px' }}>
                        {conn.status}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: '#cccccc' }}>Latency:</span>
                      <span style={{ color: '#00ffff', marginLeft: '5px' }}>
                        {conn.latency.toFixed(0)}ms
                      </span>
                    </div>
                    <div>
                      <span style={{ color: '#cccccc' }}>Throughput:</span>
                      <span style={{ color: '#ff00ff', marginLeft: '5px' }}>
                        {(conn.throughput / 1000).toFixed(1)}KB/s
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Floating metrics */}
      {systemMetrics.length > 0 && (
        <div style={{
          position: 'absolute',
          top: '20px',
          right: '20px',
          display: 'flex',
          gap: '10px',
          zIndex: 3
        }}>
          <div style={{
            background: 'rgba(0,255,255,0.2)',
            border: '1px solid #00ffff',
            borderRadius: '5px',
            padding: '5px 10px',
            fontSize: '10px'
          }}>
            CPU: {systemMetrics[systemMetrics.length - 1]?.cpu.toFixed(0)}%
          </div>
          <div style={{
            background: 'rgba(255,0,255,0.2)',
            border: '1px solid #ff00ff',
            borderRadius: '5px',
            padding: '5px 10px',
            fontSize: '10px'
          }}>
            MEM: {systemMetrics[systemMetrics.length - 1]?.memory.toFixed(0)}%
          </div>
          <div style={{
            background: 'rgba(255,255,0,0.2)',
            border: '1px solid #ffff00',
            borderRadius: '5px',
            padding: '5px 10px',
            fontSize: '10px'
          }}>
            DISK: {systemMetrics[systemMetrics.length - 1]?.disk.toFixed(0)}%
          </div>
        </div>
      )}

      {/* Scan line effect */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '2px',
        background: 'linear-gradient(90deg, transparent, #00ffff, transparent)',
        animation: 'scan 3s linear infinite',
        zIndex: 1
      }} />

      <style jsx>{`
        @keyframes scan {
          0% { transform: translateY(-2px); }
          100% { transform: translateY(600px); }
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
};

export default RealTimeMonitoring;