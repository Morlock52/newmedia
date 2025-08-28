import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface InfrastructureComponent {
  id: string;
  name: string;
  type: 'server' | 'database' | 'load-balancer' | 'cache' | 'cdn' | 'storage' | 'network';
  status: 'healthy' | 'degraded' | 'critical' | 'offline';
  location: string;
  metrics: {
    cpu: number;
    memory: number;
    disk: number;
    network: number;
    uptime: number;
    latency: number;
    throughput: number;
  };
  dependencies: string[];
  alerts: Alert[];
  lastCheck: Date;
}

interface Alert {
  id: string;
  level: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  timestamp: Date;
  acknowledged: boolean;
  component: string;
}

interface NetworkTopology {
  nodes: Array<{
    id: string;
    name: string;
    x: number;
    y: number;
    status: string;
    type: string;
  }>;
  connections: Array<{
    from: string;
    to: string;
    status: 'active' | 'inactive' | 'congested';
    bandwidth: number;
    latency: number;
  }>;
}

interface PerformanceMetric {
  timestamp: Date;
  cpu: number;
  memory: number;
  network: number;
  responseTime: number;
}

const InfrastructureMonitor: React.FC = () => {
  const [components, setComponents] = useState<InfrastructureComponent[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [topology, setTopology] = useState<NetworkTopology>({ nodes: [], connections: [] });
  const [selectedComponent, setSelectedComponent] = useState<InfrastructureComponent | null>(null);
  const [performanceHistory, setPerformanceHistory] = useState<PerformanceMetric[]>([]);
  const [systemHealth, setSystemHealth] = useState({
    overall: 0,
    availability: 0,
    performance: 0,
    security: 0
  });
  const [isMonitoring, setIsMonitoring] = useState(true);
  const [viewMode, setViewMode] = useState<'grid' | 'topology' | 'metrics'>('grid');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const metricsCanvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const animationRef = useRef<number>(0);

  useEffect(() => {
    initializeInfrastructure();
    setupWebSocket();
    startTopologyVisualization();
    startMetricsVisualization();
    
    if (isMonitoring) {
      const interval = setInterval(updateMetrics, 2000);
      return () => clearInterval(interval);
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [isMonitoring]);

  const initializeInfrastructure = () => {
    // Initialize infrastructure components
    const infrastructureComponents: InfrastructureComponent[] = [
      {
        id: 'web-server-1',
        name: 'Web Server 1',
        type: 'server',
        status: 'healthy',
        location: 'US-East-1',
        metrics: {
          cpu: 45,
          memory: 68,
          disk: 73,
          network: 34,
          uptime: 99.8,
          latency: 12,
          throughput: 156
        },
        dependencies: ['database-cluster', 'cache-redis'],
        alerts: [],
        lastCheck: new Date()
      },
      {
        id: 'web-server-2',
        name: 'Web Server 2',
        type: 'server',
        status: 'healthy',
        location: 'US-West-2',
        metrics: {
          cpu: 52,
          memory: 71,
          disk: 68,
          network: 42,
          uptime: 99.9,
          latency: 8,
          throughput: 189
        },
        dependencies: ['database-cluster', 'cache-redis'],
        alerts: [],
        lastCheck: new Date()
      },
      {
        id: 'load-balancer',
        name: 'Load Balancer',
        type: 'load-balancer',
        status: 'healthy',
        location: 'Global',
        metrics: {
          cpu: 25,
          memory: 35,
          disk: 15,
          network: 78,
          uptime: 99.95,
          latency: 2,
          throughput: 845
        },
        dependencies: ['web-server-1', 'web-server-2'],
        alerts: [],
        lastCheck: new Date()
      },
      {
        id: 'database-cluster',
        name: 'Database Cluster',
        type: 'database',
        status: 'degraded',
        location: 'US-Central',
        metrics: {
          cpu: 82,
          memory: 89,
          disk: 91,
          network: 67,
          uptime: 99.2,
          latency: 45,
          throughput: 234
        },
        dependencies: ['storage-primary', 'storage-replica'],
        alerts: [
          {
            id: 'db-alert-1',
            level: 'warning',
            message: 'High memory usage detected',
            timestamp: new Date(Date.now() - 1800000),
            acknowledged: false,
            component: 'database-cluster'
          }
        ],
        lastCheck: new Date()
      },
      {
        id: 'cache-redis',
        name: 'Redis Cache',
        type: 'cache',
        status: 'healthy',
        location: 'US-East-1',
        metrics: {
          cpu: 28,
          memory: 45,
          disk: 12,
          network: 89,
          uptime: 99.7,
          latency: 1,
          throughput: 2456
        },
        dependencies: [],
        alerts: [],
        lastCheck: new Date()
      },
      {
        id: 'cdn-cloudflare',
        name: 'CDN (CloudFlare)',
        type: 'cdn',
        status: 'healthy',
        location: 'Global',
        metrics: {
          cpu: 15,
          memory: 22,
          disk: 8,
          network: 92,
          uptime: 99.99,
          latency: 5,
          throughput: 1234
        },
        dependencies: ['load-balancer'],
        alerts: [],
        lastCheck: new Date()
      },
      {
        id: 'storage-primary',
        name: 'Primary Storage',
        type: 'storage',
        status: 'healthy',
        location: 'US-Central',
        metrics: {
          cpu: 12,
          memory: 18,
          disk: 76,
          network: 23,
          uptime: 99.8,
          latency: 3,
          throughput: 567
        },
        dependencies: [],
        alerts: [],
        lastCheck: new Date()
      },
      {
        id: 'storage-replica',
        name: 'Replica Storage',
        type: 'storage',
        status: 'critical',
        location: 'US-West-2',
        metrics: {
          cpu: 95,
          memory: 97,
          disk: 98,
          network: 89,
          uptime: 87.3,
          latency: 156,
          throughput: 23
        },
        dependencies: ['storage-primary'],
        alerts: [
          {
            id: 'storage-alert-1',
            level: 'critical',
            message: 'Storage replica experiencing critical failures',
            timestamp: new Date(Date.now() - 300000),
            acknowledged: false,
            component: 'storage-replica'
          },
          {
            id: 'storage-alert-2',
            level: 'error',
            message: 'Disk usage above 95%',
            timestamp: new Date(Date.now() - 600000),
            acknowledged: false,
            component: 'storage-replica'
          }
        ],
        lastCheck: new Date()
      },
      {
        id: 'monitoring-system',
        name: 'Monitoring System',
        type: 'server',
        status: 'healthy',
        location: 'US-East-1',
        metrics: {
          cpu: 34,
          memory: 42,
          disk: 28,
          network: 56,
          uptime: 99.9,
          latency: 8,
          throughput: 123
        },
        dependencies: [],
        alerts: [],
        lastCheck: new Date()
      }
    ];
    
    // Initialize network topology
    const networkTopology: NetworkTopology = {
      nodes: infrastructureComponents.map((comp, index) => ({
        id: comp.id,
        name: comp.name,
        x: 200 + (index % 3) * 200,
        y: 100 + Math.floor(index / 3) * 150,
        status: comp.status,
        type: comp.type
      })),
      connections: [
        { from: 'cdn-cloudflare', to: 'load-balancer', status: 'active', bandwidth: 1000, latency: 5 },
        { from: 'load-balancer', to: 'web-server-1', status: 'active', bandwidth: 500, latency: 2 },
        { from: 'load-balancer', to: 'web-server-2', status: 'active', bandwidth: 500, latency: 3 },
        { from: 'web-server-1', to: 'database-cluster', status: 'congested', bandwidth: 200, latency: 45 },
        { from: 'web-server-2', to: 'database-cluster', status: 'active', bandwidth: 200, latency: 42 },
        { from: 'web-server-1', to: 'cache-redis', status: 'active', bandwidth: 100, latency: 1 },
        { from: 'web-server-2', to: 'cache-redis', status: 'active', bandwidth: 100, latency: 1 },
        { from: 'database-cluster', to: 'storage-primary', status: 'active', bandwidth: 300, latency: 3 },
        { from: 'storage-primary', to: 'storage-replica', status: 'inactive', bandwidth: 0, latency: 156 }
      ]
    };
    
    setComponents(infrastructureComponents);
    setTopology(networkTopology);
    
    // Aggregate all alerts
    const allAlerts = infrastructureComponents.flatMap(comp => comp.alerts);
    setAlerts(allAlerts);
    
    // Calculate system health
    const healthyCount = infrastructureComponents.filter(c => c.status === 'healthy').length;
    const totalCount = infrastructureComponents.length;
    const avgUptime = infrastructureComponents.reduce((sum, c) => sum + c.metrics.uptime, 0) / totalCount;
    const avgLatency = infrastructureComponents.reduce((sum, c) => sum + c.metrics.latency, 0) / totalCount;
    
    setSystemHealth({
      overall: Math.round((healthyCount / totalCount) * 100),
      availability: Math.round(avgUptime),
      performance: Math.round(Math.max(0, 100 - avgLatency * 2)),
      security: Math.round(100 - (allAlerts.filter(a => a.level === 'critical').length * 20))
    });
    
    // Initialize performance history
    const history: PerformanceMetric[] = [];
    for (let i = 0; i < 20; i++) {
      history.push({
        timestamp: new Date(Date.now() - (19 - i) * 30000),
        cpu: 40 + Math.random() * 30,
        memory: 50 + Math.random() * 30,
        network: 30 + Math.random() * 40,
        responseTime: 10 + Math.random() * 20
      });
    }
    setPerformanceHistory(history);
  };

  const setupWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8080/infrastructure');
      
      wsRef.current.onmessage = (event) => {
        const update = JSON.parse(event.data);
        handleInfrastructureUpdate(update);
      };
    } catch (error) {
      console.warn('Infrastructure WebSocket not available');
    }
  };

  const handleInfrastructureUpdate = (update: any) => {
    if (update.type === 'component_update') {
      setComponents(prev => prev.map(comp => 
        comp.id === update.componentId 
          ? { ...comp, metrics: { ...comp.metrics, ...update.metrics }, lastCheck: new Date() }
          : comp
      ));
    } else if (update.type === 'alert') {
      const newAlert: Alert = {
        id: Date.now().toString(),
        level: update.level,
        message: update.message,
        timestamp: new Date(),
        acknowledged: false,
        component: update.component
      };
      setAlerts(prev => [newAlert, ...prev]);
    }
  };

  const updateMetrics = () => {
    setComponents(prev => prev.map(comp => {
      const newMetrics = { ...comp.metrics };
      
      // Simulate metric fluctuations
      Object.keys(newMetrics).forEach(key => {
        if (key !== 'uptime') {
          const current = newMetrics[key as keyof typeof newMetrics];
          const variation = (Math.random() - 0.5) * 10;
          newMetrics[key as keyof typeof newMetrics] = Math.max(0, Math.min(100, current + variation));
        }
      });
      
      // Simulate status changes based on metrics
      let newStatus = comp.status;
      if (comp.id === 'storage-replica') {
        newStatus = 'critical'; // Keep this critical for demo
      } else if (newMetrics.cpu > 90 || newMetrics.memory > 90) {
        newStatus = 'critical';
      } else if (newMetrics.cpu > 75 || newMetrics.memory > 80) {
        newStatus = 'degraded';
      } else {
        newStatus = 'healthy';
      }
      
      return {
        ...comp,
        metrics: newMetrics,
        status: newStatus,
        lastCheck: new Date()
      };
    }));
    
    // Add new performance metric
    const newMetric: PerformanceMetric = {
      timestamp: new Date(),
      cpu: 40 + Math.random() * 30,
      memory: 50 + Math.random() * 30,
      network: 30 + Math.random() * 40,
      responseTime: 10 + Math.random() * 20
    };
    
    setPerformanceHistory(prev => [...prev.slice(-19), newMetric]);
  };

  const startTopologyVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw connections
      topology.connections.forEach(conn => {
        const fromNode = topology.nodes.find(n => n.id === conn.from);
        const toNode = topology.nodes.find(n => n.id === conn.to);
        
        if (fromNode && toNode) {
          const color = conn.status === 'active' ? '#00FF00' : 
                       conn.status === 'congested' ? '#FFFF00' : '#FF0040';
          
          ctx.strokeStyle = color;
          ctx.lineWidth = Math.max(1, conn.bandwidth / 200);
          ctx.globalAlpha = conn.status === 'inactive' ? 0.3 : 0.8;
          
          ctx.beginPath();
          ctx.moveTo(fromNode.x, fromNode.y);
          ctx.lineTo(toNode.x, toNode.y);
          ctx.stroke();
          
          // Draw data flow animation
          if (conn.status === 'active') {
            const time = Date.now() / 1000;
            const progress = (time % 2) / 2;
            const x = fromNode.x + (toNode.x - fromNode.x) * progress;
            const y = fromNode.y + (toNode.y - fromNode.y) * progress;
            
            ctx.fillStyle = '#FF00FF';
            ctx.globalAlpha = 1;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
          }
          
          ctx.globalAlpha = 1;
        }
      });
      
      // Draw nodes
      topology.nodes.forEach(node => {
        const color = getStatusColor(node.status);
        const size = node.type === 'load-balancer' || node.type === 'cdn' ? 25 : 20;
        
        ctx.fillStyle = color;
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        
        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        
        // Add pulsing effect for critical nodes
        if (node.status === 'critical') {
          const pulseRadius = size + Math.sin(Date.now() / 200) * 8;
          ctx.strokeStyle = '#FF0040';
          ctx.globalAlpha = 0.5;
          ctx.lineWidth = 2;
          
          ctx.beginPath();
          ctx.arc(node.x, node.y, pulseRadius, 0, 2 * Math.PI);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
        
        // Draw node label
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '12px Orbitron';
        ctx.textAlign = 'center';
        ctx.fillText(node.name, node.x, node.y + size + 20);
      });
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
  };

  const startMetricsVisualization = () => {
    const canvas = metricsCanvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const drawMetrics = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const metrics = ['cpu', 'memory', 'network', 'responseTime'];
      const colors = ['#00FFFF', '#FF00FF', '#FFFF00', '#00FF00'];
      
      metrics.forEach((metric, index) => {
        const yOffset = index * (canvas.height / metrics.length);
        const height = canvas.height / metrics.length;
        
        ctx.strokeStyle = colors[index];
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.8;
        
        ctx.beginPath();
        performanceHistory.forEach((point, i) => {
          const x = (i / (performanceHistory.length - 1)) * canvas.width;
          const value = point[metric as keyof PerformanceMetric] as number;
          const y = yOffset + height - (value / 100) * height;
          
          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.stroke();
        
        // Add glow effect
        ctx.shadowColor = colors[index];
        ctx.shadowBlur = 10;
        ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1;
        
        // Add metric label
        ctx.fillStyle = colors[index];
        ctx.font = '14px Orbitron';
        ctx.fillText(metric.toUpperCase(), 10, yOffset + 20);
      });
      
      setTimeout(drawMetrics, 1000);
    };
    
    drawMetrics();
  };

  const acknowledgeAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    ));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return '#00FF00';
      case 'degraded': return '#FFFF00';
      case 'critical': return '#FF0040';
      case 'offline': return '#666666';
      default: return '#00FFFF';
    }
  };

  const getAlertColor = (level: string) => {
    switch (level) {
      case 'info': return '#00FFFF';
      case 'warning': return '#FFFF00';
      case 'error': return '#FF6600';
      case 'critical': return '#FF0040';
      default: return '#666666';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'server': return '🖥️';
      case 'database': return '🗄';
      case 'load-balancer': return '⚖️';
      case 'cache': return '💾';
      case 'cdn': return '🌐';
      case 'storage': return '🗜️';
      case 'network': return '🌐';
      default: return '💻';
    }
  };

  const filteredComponents = components.filter(comp => 
    filterStatus === 'all' || comp.status === filterStatus
  );

  return (
    <div style={{
      background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%)',
      color: '#00FFFF',
      fontFamily: 'Orbitron, monospace',
      minHeight: '100vh',
      padding: '20px',
      position: 'relative'
    }}>
      {/* Infrastructure Grid Background */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundImage: `
          linear-gradient(90deg, rgba(0,255,255,0.05) 1px, transparent 1px),
          linear-gradient(rgba(0,255,255,0.05) 1px, transparent 1px)
        `,
        backgroundSize: '50px 50px',
        opacity: 0.4,
        pointerEvents: 'none'
      }} />

      {/* Header */}
      <motion.header
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '30px',
          zIndex: 10,
          position: 'relative'
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
            animation: 'infraGlow 4s ease-in-out infinite alternate'
          }}>
            INFRASTRUCTURE MONITOR
          </h1>
          <div style={{
            display: 'flex',
            gap: '20px',
            marginTop: '10px',
            fontSize: '0.9rem'
          }}>
            <span>Overall Health: <strong style={{ color: systemHealth.overall > 80 ? '#00FF00' : systemHealth.overall > 60 ? '#FFFF00' : '#FF0040' }}>{systemHealth.overall}%</strong></span>
            <span>Availability: <strong style={{ color: '#00FFFF' }}>{systemHealth.availability}%</strong></span>
            <span>Active Alerts: <strong style={{ color: alerts.filter(a => !a.acknowledged).length > 0 ? '#FF0040' : '#00FF00' }}>{alerts.filter(a => !a.acknowledged).length}</strong></span>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: '10px' }}>
            {['grid', 'topology', 'metrics'].map(mode => (
              <button
                key={mode}
                onClick={() => setViewMode(mode as any)}
                style={{
                  padding: '8px 15px',
                  background: viewMode === mode ? 'linear-gradient(45deg, #00FFFF, #FF00FF)' : 'rgba(0,255,255,0.2)',
                  border: '1px solid #00FFFF',
                  borderRadius: '6px',
                  color: viewMode === mode ? '#000' : '#00FFFF',
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                  textTransform: 'capitalize'
                }}
              >
                {mode}
              </button>
            ))}
          </div>
          
          <button
            onClick={() => setIsMonitoring(!isMonitoring)}
            style={{
              padding: '12px 25px',
              background: isMonitoring ? 'linear-gradient(45deg, #00FF00, #00FFFF)' : 'rgba(255,0,64,0.2)',
              border: 'none',
              borderRadius: '8px',
              color: isMonitoring ? '#000' : '#FF0040',
              fontWeight: 'bold',
              cursor: 'pointer',
              fontSize: '1rem'
            }}
          >
            {isMonitoring ? '⏸️ MONITORING' : '▶️ START MONITOR'}
          </button>
        </div>
      </motion.header>

      {/* System Health Dashboard */}
      <motion.section
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2 }}
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '20px',
          marginBottom: '30px',
          zIndex: 10,
          position: 'relative'
        }}
      >
        {[
          { label: 'Overall Health', value: systemHealth.overall, color: '#00FFFF', icon: '🎯' },
          { label: 'Availability', value: systemHealth.availability, color: '#00FF00', icon: '✅' },
          { label: 'Performance', value: systemHealth.performance, color: '#FFFF00', icon: '⚡' },
          { label: 'Security', value: systemHealth.security, color: '#FF00FF', icon: '🔒' }
        ].map((health, index) => (
          <div
            key={health.label}
            style={{
              background: 'rgba(0,0,0,0.8)',
              border: `3px solid ${health.color}`,
              borderRadius: '15px',
              padding: '25px',
              textAlign: 'center',
              position: 'relative',
              overflow: 'hidden',
              boxShadow: `0 0 30px rgba(${health.color === '#00FFFF' ? '0,255,255' : health.color === '#00FF00' ? '0,255,0' : health.color === '#FFFF00' ? '255,255,0' : '255,0,255'},0.4)`
            }}
          >
            <div style={{ fontSize: '3rem', marginBottom: '15px' }}>{health.icon}</div>
            <div style={{
              fontSize: '2.5rem',
              fontWeight: 'bold',
              color: health.color,
              marginBottom: '8px'
            }}>
              {health.value}%
            </div>
            <div style={{ fontSize: '1rem', opacity: 0.8 }}>{health.label}</div>
            
            {/* Circular Progress */}
            <div style={{
              position: 'absolute',
              top: '15px',
              right: '15px',
              width: '40px',
              height: '40px',
              borderRadius: '50%',
              background: `conic-gradient(${health.color} ${health.value * 3.6}deg, rgba(255,255,255,0.1) 0deg)`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '0.8rem',
              fontWeight: 'bold'
            }}>
              {health.value}%
            </div>
          </div>
        ))}
      </motion.section>

      {/* Alert Panel */}
      <AnimatePresence>
        {alerts.filter(a => !a.acknowledged).length > 0 && (
          <motion.section
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            style={
            {
              background: 'rgba(0,0,0,0.9)',
              border: '2px solid #FF0040',
              borderRadius: '12px',
              padding: '20px',
              marginBottom: '30px',
              position: 'relative',
              zIndex: 10,
              animation: 'alertPulse 2s infinite'
            }}
          >
            <h3 style={{ color: '#FF0040', marginBottom: '15px' }}>ACTIVE ALERTS</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {alerts.filter(a => !a.acknowledged).slice(0, 3).map(alert => (
                <div
                  key={alert.id}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '10px',
                    background: 'rgba(0,0,0,0.7)',
                    border: `1px solid ${getAlertColor(alert.level)}`,
                    borderRadius: '8px'
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '5px' }}>
                      <span style={{
                        padding: '3px 8px',
                        background: getAlertColor(alert.level),
                        color: '#000',
                        borderRadius: '4px',
                        fontSize: '0.7rem',
                        fontWeight: 'bold',
                        textTransform: 'uppercase'
                      }}>
                        {alert.level}
                      </span>
                      <span style={{ color: '#00FFFF' }}>{alert.component}</span>
                      <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>{alert.timestamp.toLocaleTimeString()}</span>
                    </div>
                    <div style={{ fontSize: '0.9rem' }}>{alert.message}</div>
                  </div>
                  <button
                    onClick={() => acknowledgeAlert(alert.id)}
                    style={{
                      padding: '6px 12px',
                      background: 'linear-gradient(45deg, #FFFF00, #FF00FF)',
                      border: 'none',
                      borderRadius: '6px',
                      color: '#000',
                      fontWeight: 'bold',
                      cursor: 'pointer',
                      fontSize: '0.8rem'
                    }}
                  >
                    ACK
                  </button>
                </div>
              ))}
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      {/* Main Content Area */}
      {viewMode === 'grid' && (
        <motion.section
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{
            position: 'relative',
            zIndex: 10
          }}
        >
          {/* Filter Bar */}
          <div style={{
            display: 'flex',
            gap: '10px',
            marginBottom: '20px',
            flexWrap: 'wrap'
          }}>
            {['all', 'healthy', 'degraded', 'critical', 'offline'].map(status => (
              <button
                key={status}
                onClick={() => setFilterStatus(status)}
                style={{
                  padding: '8px 15px',
                  background: filterStatus === status ? 'linear-gradient(45deg, #00FFFF, #FF00FF)' : 'rgba(0,255,255,0.1)',
                  border: '1px solid #00FFFF',
                  borderRadius: '20px',
                  color: filterStatus === status ? '#000' : '#00FFFF',
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                  textTransform: 'capitalize'
                }}
              >
                {status === 'all' ? 'All Components' : status} 
                {status !== 'all' && (
                  <span style={{ marginLeft: '5px', fontWeight: 'bold' }}>
                    ({components.filter(c => c.status === status).length})
                  </span>
                )}
              </button>
            ))}
          </div>
          
          {/* Components Grid */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
            gap: '20px'
          }}>
            {filteredComponents.map((component, index) => (
              <motion.div
                key={component.id}
                initial={{ opacity: 0, scale: 0.9, y: 20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                whileHover={{ scale: 1.02, y: -5 }}
                style={{
                  background: 'rgba(0,0,0,0.8)',
                  border: `3px solid ${getStatusColor(component.status)}`,
                  borderRadius: '15px',
                  padding: '25px',
                  cursor: 'pointer',
                  position: 'relative',
                  overflow: 'hidden'
                }}
                onClick={() => setSelectedComponent(component)}
              >
                {/* Component Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                    <div style={{
                      fontSize: '3rem',
                      filter: `drop-shadow(0 0 10px ${getStatusColor(component.status)})`
                    }}>
                      {getTypeIcon(component.type)}
                    </div>
                    <div>
                      <h3 style={{ margin: 0, color: '#00FFFF', fontSize: '1.3rem' }}>{component.name}</h3>
                      <div style={{ display: 'flex', gap: '10px', marginTop: '5px' }}>
                        <span style={{
                          padding: '3px 8px',
                          background: getStatusColor(component.status),
                          color: component.status === 'healthy' ? '#000' : '#FFF',
                          borderRadius: '10px',
                          fontSize: '0.8rem',
                          fontWeight: 'bold',
                          textTransform: 'uppercase'
                        }}>
                          {component.status}
                        </span>
                        <span style={{
                          padding: '3px 8px',
                          background: 'rgba(255,255,0,0.2)',
                          color: '#FFFF00',
                          borderRadius: '10px',
                          fontSize: '0.8rem',
                          textTransform: 'capitalize'
                        }}>
                          {component.type}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ color: '#00FFFF', fontSize: '0.9rem' }}>{component.location}</div>
                    <div style={{ fontSize: '0.8rem', opacity: 0.7 }}>
                      Uptime: {component.metrics.uptime}%
                    </div>
                  </div>
                </div>
                
                {/* Metrics */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '15px',
                  marginBottom: '20px'
                }}>
                  {[
                    { label: 'CPU', value: component.metrics.cpu, unit: '%', color: '#FF00FF' },
                    { label: 'Memory', value: component.metrics.memory, unit: '%', color: '#FFFF00' },
                    { label: 'Disk', value: component.metrics.disk, unit: '%', color: '#FF6600' },
                    { label: 'Network', value: component.metrics.network, unit: '%', color: '#00FF00' }
                  ].map(metric => (
                    <div key={metric.label} style={{ textAlign: 'center' }}>
                      <div style={{
                        fontSize: '1.2rem',
                        fontWeight: 'bold',
                        color: metric.color
                      }}>
                        {metric.value.toFixed(1)}{metric.unit}
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
                
                {/* Performance Indicators */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '15px'
                }}>
                  <span>Latency: <strong style={{ color: '#00FFFF' }}>{component.metrics.latency}ms</strong></span>
                  <span>Throughput: <strong style={{ color: '#FF00FF' }}>{component.metrics.throughput}/s</strong></span>
                </div>
                
                {/* Alerts Count */}
                {component.alerts.length > 0 && (
                  <div style={{
                    padding: '10px',
                    background: 'rgba(255,0,64,0.2)',
                    border: '1px solid #FF0040',
                    borderRadius: '8px',
                    marginBottom: '15px'
                  }}>
                    <strong style={{ color: '#FF0040' }}>Active Alerts: {component.alerts.length}</strong>
                  </div>
                )}
                
                {/* Dependencies */}
                {component.dependencies.length > 0 && (
                  <div style={{ fontSize: '0.8rem', opacity: 0.7 }}>
                    Dependencies: {component.dependencies.join(', ')}
                  </div>
                )}
                
                {/* Last Check */}
                <div style={{
                  position: 'absolute',
                  bottom: '10px',
                  right: '15px',
                  fontSize: '0.7rem',
                  opacity: 0.6
                }}>
                  Updated: {component.lastCheck.toLocaleTimeString()}
                </div>
                
                {/* Status Pulse */}
                {component.status !== 'healthy' && (
                  <div style={{
                    position: 'absolute',
                    top: '15px',
                    right: '15px',
                    width: '12px',
                    height: '12px',
                    background: getStatusColor(component.status),
                    borderRadius: '50%',
                    animation: 'statusPulse 2s infinite'
                  }} />
                )}
              </motion.div>
            ))}
          </div>
        </motion.section>
      )}

      {viewMode === 'topology' && (
        <motion.section
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{
            background: 'rgba(0,0,0,0.8)',
            border: '2px solid #00FFFF',
            borderRadius: '15px',
            padding: '30px',
            position: 'relative',
            zIndex: 10
          }}
        >
          <h2 style={{ color: '#00FFFF', marginBottom: '25px', textAlign: 'center' }}>NETWORK TOPOLOGY</h2>
          
          <canvas
            ref={canvasRef}
            width={1000}
            height={600}
            style={{
              width: '100%',
              height: '600px',
              border: '1px solid #333',
              borderRadius: '8px',
              background: 'rgba(0,0,0,0.5)'
            }}
          />
          
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '30px',
            marginTop: '20px',
            fontSize: '0.9rem'
          }}>
            {[
              { status: 'active', color: '#00FF00', label: 'Active Connection' },
              { status: 'congested', color: '#FFFF00', label: 'Congested' },
              { status: 'inactive', color: '#FF0040', label: 'Inactive' },
              { status: 'healthy', color: '#00FF00', label: 'Healthy Node' },
              { status: 'critical', color: '#FF0040', label: 'Critical Node' }
            ].map(item => (
              <div key={item.status} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{
                  width: '12px',
                  height: '12px',
                  background: item.color,
                  borderRadius: '50%',
                  boxShadow: `0 0 8px ${item.color}`
                }} />
                <span>{item.label}</span>
              </div>
            ))}
          </div>
        </motion.section>
      )}

      {viewMode === 'metrics' && (
        <motion.section
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{
            background: 'rgba(0,0,0,0.8)',
            border: '2px solid #FFFF00',
            borderRadius: '15px',
            padding: '30px',
            position: 'relative',
            zIndex: 10
          }}
        >
          <h2 style={{ color: '#FFFF00', marginBottom: '25px', textAlign: 'center' }}>PERFORMANCE METRICS</h2>
          
          <canvas
            ref={metricsCanvasRef}
            width={1000}
            height={400}
            style={{
              width: '100%',
              height: '400px',
              border: '1px solid #333',
              borderRadius: '8px',
              background: 'rgba(0,0,0,0.5)'
            }}
          />
          
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '40px',
            marginTop: '20px',
            fontSize: '0.9rem'
          }}>
            {[
              { metric: 'CPU', color: '#00FFFF' },
              { metric: 'Memory', color: '#FF00FF' },
              { metric: 'Network', color: '#FFFF00' },
              { metric: 'Response Time', color: '#00FF00' }
            ].map(item => (
              <div key={item.metric} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{
                  width: '4px',
                  height: '20px',
                  background: item.color,
                  borderRadius: '2px'
                }} />
                <span>{item.metric}</span>
              </div>
            ))}
          </div>
        </motion.section>
      )}

      <style jsx>{`
        @keyframes infraGlow {
          0% { filter: hue-rotate(0deg) brightness(1); }
          100% { filter: hue-rotate(180deg) brightness(1.4); }
        }
        
        @keyframes alertPulse {
          0%, 100% { box-shadow: 0 0 20px rgba(255,0,64,0.5); }
          50% { box-shadow: 0 0 40px rgba(255,0,64,1); }
        }
        
        @keyframes statusPulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.6; transform: scale(1.4); }
        }
      `}</style>
    </div>
  );
};

export default InfrastructureMonitor;