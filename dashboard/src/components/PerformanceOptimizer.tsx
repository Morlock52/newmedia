import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface PerformanceMetric {
  name: string;
  current: number;
  target: number;
  unit: string;
  status: 'optimal' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable';
  history: number[];
}

interface OptimizationRule {
  id: string;
  name: string;
  description: string;
  category: 'cpu' | 'memory' | 'network' | 'storage' | 'cache';
  priority: 'low' | 'medium' | 'high' | 'critical';
  enabled: boolean;
  impact: number;
  lastApplied?: Date;
}

interface SystemResource {
  id: string;
  name: string;
  type: 'cpu' | 'memory' | 'disk' | 'network';
  usage: number;
  capacity: number;
  temperature?: number;
  processes: Array<{
    name: string;
    usage: number;
    priority: number;
  }>;
}

interface BottleneckAnalysis {
  component: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  impact: number;
  recommendation: string;
  estimatedGain: string;
}

const PerformanceOptimizer: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetric[]>([]);
  const [systemResources, setSystemResources] = useState<SystemResource[]>([]);
  const [optimizationRules, setOptimizationRules] = useState<OptimizationRule[]>([]);
  const [bottlenecks, setBottlenecks] = useState<BottleneckAnalysis[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [autoOptimize, setAutoOptimize] = useState(false);
  const [optimizationScore, setOptimizationScore] = useState(0);
  const [realTimeMode, setRealTimeMode] = useState(true);
  const [selectedResource, setSelectedResource] = useState<SystemResource | null>(null);
  const [performanceHistory, setPerformanceHistory] = useState<number[]>([]);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    initializePerformanceData();
    setupWebSocket();
    startPerformanceVisualization();
    
    if (realTimeMode) {
      const interval = setInterval(updateMetrics, 1000);
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
  }, [realTimeMode]);

  const initializePerformanceData = () => {
    // Initialize performance metrics
    const performanceMetrics: PerformanceMetric[] = [
      {
        name: 'CPU Usage',
        current: 45,
        target: 70,
        unit: '%',
        status: 'optimal',
        trend: 'stable',
        history: Array.from({ length: 20 }, () => 40 + Math.random() * 20)
      },
      {
        name: 'Memory Usage',
        current: 68,
        target: 80,
        unit: '%',
        status: 'warning',
        trend: 'up',
        history: Array.from({ length: 20 }, () => 60 + Math.random() * 20)
      },
      {
        name: 'Network Latency',
        current: 25,
        target: 50,
        unit: 'ms',
        status: 'optimal',
        trend: 'down',
        history: Array.from({ length: 20 }, () => 20 + Math.random() * 20)
      },
      {
        name: 'Disk I/O',
        current: 150,
        target: 200,
        unit: 'MB/s',
        status: 'optimal',
        trend: 'stable',
        history: Array.from({ length: 20 }, () => 140 + Math.random() * 30)
      },
      {
        name: 'Cache Hit Rate',
        current: 94,
        target: 95,
        unit: '%',
        status: 'warning',
        trend: 'up',
        history: Array.from({ length: 20 }, () => 90 + Math.random() * 8)
      },
      {
        name: 'Response Time',
        current: 120,
        target: 100,
        unit: 'ms',
        status: 'warning',
        trend: 'down',
        history: Array.from({ length: 20 }, () => 110 + Math.random() * 20)
      }
    ];
    
    // Initialize system resources
    const resources: SystemResource[] = [
      {
        id: 'cpu',
        name: 'CPU',
        type: 'cpu',
        usage: 45,
        capacity: 100,
        temperature: 62,
        processes: [
          { name: 'jellyfin', usage: 15, priority: 8 },
          { name: 'plex', usage: 12, priority: 7 },
          { name: 'sonarr', usage: 8, priority: 6 },
          { name: 'qbittorrent', usage: 10, priority: 5 }
        ]
      },
      {
        id: 'memory',
        name: 'RAM',
        type: 'memory',
        usage: 68,
        capacity: 100,
        processes: [
          { name: 'jellyfin', usage: 25, priority: 8 },
          { name: 'plex', usage: 20, priority: 7 },
          { name: 'database', usage: 15, priority: 9 },
          { name: 'cache', usage: 8, priority: 6 }
        ]
      },
      {
        id: 'disk',
        name: 'Storage',
        type: 'disk',
        usage: 73,
        capacity: 100,
        processes: [
          { name: 'media-scan', usage: 40, priority: 6 },
          { name: 'backup', usage: 20, priority: 4 },
          { name: 'downloads', usage: 13, priority: 5 }
        ]
      },
      {
        id: 'network',
        name: 'Network',
        type: 'network',
        usage: 35,
        capacity: 100,
        processes: [
          { name: 'streaming', usage: 60, priority: 9 },
          { name: 'downloads', usage: 25, priority: 7 },
          { name: 'api-calls', usage: 15, priority: 6 }
        ]
      }
    ];
    
    // Initialize optimization rules
    const rules: OptimizationRule[] = [
      {
        id: 'cpu-throttle',
        name: 'CPU Throttling',
        description: 'Reduce CPU frequency during low usage periods',
        category: 'cpu',
        priority: 'medium',
        enabled: true,
        impact: 15
      },
      {
        id: 'memory-compress',
        name: 'Memory Compression',
        description: 'Compress inactive memory pages',
        category: 'memory',
        priority: 'high',
        enabled: true,
        impact: 25
      },
      {
        id: 'cache-preload',
        name: 'Smart Cache Preloading',
        description: 'Preload frequently accessed media',
        category: 'cache',
        priority: 'medium',
        enabled: false,
        impact: 20
      },
      {
        id: 'network-qos',
        name: 'Network QoS',
        description: 'Prioritize streaming traffic',
        category: 'network',
        priority: 'high',
        enabled: true,
        impact: 30
      },
      {
        id: 'disk-defrag',
        name: 'Smart Defragmentation',
        description: 'Defragment storage during idle time',
        category: 'storage',
        priority: 'low',
        enabled: false,
        impact: 10
      },
      {
        id: 'process-priority',
        name: 'Process Priority Optimization',
        description: 'Adjust process priorities dynamically',
        category: 'cpu',
        priority: 'critical',
        enabled: true,
        impact: 35
      }
    ];
    
    // Initialize bottleneck analysis
    const bottleneckAnalysis: BottleneckAnalysis[] = [
      {
        component: 'Database Connection Pool',
        severity: 'high',
        impact: 85,
        recommendation: 'Increase connection pool size to 20 connections',
        estimatedGain: '40% faster response time'
      },
      {
        component: 'Transcoding Pipeline',
        severity: 'medium',
        impact: 60,
        recommendation: 'Enable hardware acceleration for H.264/H.265',
        estimatedGain: '60% reduction in CPU usage'
      },
      {
        component: 'Cache Invalidation',
        severity: 'low',
        impact: 25,
        recommendation: 'Implement smart cache warming strategies',
        estimatedGain: '15% better cache hit rate'
      },
      {
        component: 'Memory Leaks',
        severity: 'critical',
        impact: 95,
        recommendation: 'Update affected services to latest versions',
        estimatedGain: '30% memory usage reduction'
      }
    ];
    
    setMetrics(performanceMetrics);
    setSystemResources(resources);
    setOptimizationRules(rules);
    setBottlenecks(bottleneckAnalysis);
    
    // Calculate optimization score
    const enabledRules = rules.filter(r => r.enabled);
    const totalImpact = enabledRules.reduce((sum, r) => sum + r.impact, 0);
    setOptimizationScore(Math.min(100, totalImpact));
  };

  const setupWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8080/performance');
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handlePerformanceUpdate(data);
      };
    } catch (error) {
      console.warn('Performance WebSocket not available');
    }
  };

  const handlePerformanceUpdate = (data: any) => {
    if (data.type === 'metrics') {
      setMetrics(prev => prev.map(metric => {
        const update = data.metrics.find((m: any) => m.name === metric.name);
        if (update) {
          const newHistory = [...metric.history.slice(1), update.current];
          return {
            ...metric,
            current: update.current,
            history: newHistory,
            status: update.current > metric.target ? 'warning' : 'optimal',
            trend: update.current > metric.current ? 'up' : update.current < metric.current ? 'down' : 'stable'
          };
        }
        return metric;
      }));
    }
  };

  const updateMetrics = () => {
    setMetrics(prev => prev.map(metric => {
      const variation = (Math.random() - 0.5) * 10;
      const newValue = Math.max(0, metric.current + variation);
      const newHistory = [...metric.history.slice(1), newValue];
      
      return {
        ...metric,
        current: newValue,
        history: newHistory,
        status: newValue > metric.target ? 'warning' : newValue > metric.target * 0.9 ? 'warning' : 'optimal',
        trend: newValue > metric.current ? 'up' : newValue < metric.current ? 'down' : 'stable'
      };
    }));
    
    setSystemResources(prev => prev.map(resource => ({
      ...resource,
      usage: Math.max(0, Math.min(100, resource.usage + (Math.random() - 0.5) * 5)),
      temperature: resource.temperature ? Math.max(40, Math.min(80, resource.temperature + (Math.random() - 0.5) * 3)) : undefined
    })));
    
    // Update performance history
    const avgPerformance = metrics.reduce((sum, m) => sum + (m.current / m.target * 100), 0) / metrics.length;
    setPerformanceHistory(prev => [...prev.slice(-19), avgPerformance]);
  };

  const startPerformanceVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw performance metrics as waves
      metrics.forEach((metric, index) => {
        const yOffset = (index * canvas.height) / metrics.length;
        const height = canvas.height / metrics.length;
        
        ctx.strokeStyle = getMetricColor(metric.status);
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.8;
        
        ctx.beginPath();
        for (let x = 0; x < canvas.width; x++) {
          const progress = x / canvas.width;
          const historyIndex = Math.floor(progress * (metric.history.length - 1));
          const value = metric.history[historyIndex] || 0;
          const y = yOffset + height - (value / 100) * height;
          
          if (x === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
        
        // Add glow effect
        ctx.shadowColor = getMetricColor(metric.status);
        ctx.shadowBlur = 10;
        ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1;
      });
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
  };

  const runOptimization = async (ruleId?: string) => {
    setIsAnalyzing(true);
    
    const rulesToApply = ruleId ? [ruleId] : optimizationRules.filter(r => r.enabled).map(r => r.id);
    
    for (const id of rulesToApply) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setOptimizationRules(prev => prev.map(rule => 
        rule.id === id ? { ...rule, lastApplied: new Date() } : rule
      ));
      
      // Simulate performance improvement
      const rule = optimizationRules.find(r => r.id === id);
      if (rule) {
        setMetrics(prev => prev.map(metric => {
          if ((rule.category === 'cpu' && metric.name.includes('CPU')) ||
              (rule.category === 'memory' && metric.name.includes('Memory')) ||
              (rule.category === 'network' && metric.name.includes('Network')) ||
              (rule.category === 'cache' && metric.name.includes('Cache'))) {
            return {
              ...metric,
              current: Math.max(0, metric.current - (rule.impact * 0.5)),
              status: 'optimal'
            };
          }
          return metric;
        }));
      }
    }
    
    setIsAnalyzing(false);
  };

  const toggleOptimizationRule = (ruleId: string) => {
    setOptimizationRules(prev => prev.map(rule => 
      rule.id === ruleId ? { ...rule, enabled: !rule.enabled } : rule
    ));
    
    // Recalculate optimization score
    const enabledRules = optimizationRules.filter(r => r.enabled || r.id === ruleId);
    const totalImpact = enabledRules.reduce((sum, r) => sum + (r.enabled || r.id === ruleId ? r.impact : 0), 0);
    setOptimizationScore(Math.min(100, totalImpact));
  };

  const getMetricColor = (status: string) => {
    switch (status) {
      case 'optimal': return '#00FF00';
      case 'warning': return '#FFFF00';
      case 'critical': return '#FF0040';
      default: return '#00FFFF';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return '#00FF00';
      case 'medium': return '#FFFF00';
      case 'high': return '#FF6600';
      case 'critical': return '#FF0040';
      default: return '#666666';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'low': return '#00FFFF';
      case 'medium': return '#FFFF00';
      case 'high': return '#FF00FF';
      case 'critical': return '#FF0040';
      default: return '#666666';
    }
  };

  return (
    <div style={{
      background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%)',
      color: '#00FFFF',
      fontFamily: 'Orbitron, monospace',
      minHeight: '100vh',
      padding: '20px',
      position: 'relative'
    }}>
      {/* Circuit Pattern Background */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundImage: `
          linear-gradient(90deg, rgba(0,255,255,0.1) 1px, transparent 1px),
          linear-gradient(rgba(0,255,255,0.1) 1px, transparent 1px),
          radial-gradient(circle at 25% 25%, rgba(255,0,255,0.1) 2px, transparent 2px)
        `,
        backgroundSize: '40px 40px, 40px 40px, 80px 80px',
        opacity: 0.3,
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
            animation: 'performanceGlow 3s ease-in-out infinite alternate'
          }}>
            PERFORMANCE OPTIMIZER
          </h1>
          <div style={{
            display: 'flex',
            gap: '20px',
            marginTop: '10px',
            fontSize: '0.9rem'
          }}>
            <span>Optimization Score: <strong style={{ color: '#00FF00' }}>{optimizationScore}%</strong></span>
            <span>Active Rules: <strong style={{ color: '#FFFF00' }}>{optimizationRules.filter(r => r.enabled).length}</strong></span>
            <span>Status: <strong style={{ color: isAnalyzing ? '#FFFF00' : '#00FF00' }}>{isAnalyzing ? 'OPTIMIZING' : 'READY'}</strong></span>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <button
            onClick={() => setRealTimeMode(!realTimeMode)}
            style={{
              padding: '10px 20px',
              background: realTimeMode ? 'linear-gradient(45deg, #00FF00, #00FFFF)' : 'rgba(0,255,0,0.2)',
              border: '2px solid #00FF00',
              borderRadius: '8px',
              color: realTimeMode ? '#000' : '#00FF00',
              fontWeight: 'bold',
              cursor: 'pointer'
            }}
          >
            {realTimeMode ? 'REAL-TIME ON' : 'REAL-TIME OFF'}
          </button>
          
          <button
            onClick={() => runOptimization()}
            disabled={isAnalyzing}
            style={{
              padding: '12px 25px',
              background: isAnalyzing ? 'rgba(255,255,0,0.3)' : 'linear-gradient(45deg, #FF00FF, #FFFF00)',
              border: 'none',
              borderRadius: '8px',
              color: isAnalyzing ? '#FFFF00' : '#000',
              fontWeight: 'bold',
              cursor: 'pointer',
              fontSize: '1rem'
            }}
          >
            {isAnalyzing ? 'OPTIMIZING...' : '⚡ OPTIMIZE ALL'}
          </button>
        </div>
      </motion.header>

      {/* Performance Metrics Dashboard */}
      <motion.section
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2 }}
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
          gap: '20px',
          marginBottom: '30px',
          zIndex: 10,
          position: 'relative'
        }}
      >
        {metrics.map((metric, index) => (
          <div
            key={metric.name}
            style={{
              background: 'rgba(0,0,0,0.8)',
              border: `3px solid ${getMetricColor(metric.status)}`,
              borderRadius: '15px',
              padding: '25px',
              position: 'relative',
              overflow: 'hidden',
              boxShadow: `0 0 30px rgba(${getMetricColor(metric.status) === '#00FF00' ? '0,255,0' : getMetricColor(metric.status) === '#FFFF00' ? '255,255,0' : '255,0,64'},0.4)`
            }}
          >
            {/* Metric Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '15px' }}>
              <h3 style={{ margin: 0, color: '#00FFFF', fontSize: '1.1rem' }}>{metric.name}</h3>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                color: getMetricColor(metric.status)
              }}>
                <span style={{ fontSize: '1.2rem' }}>
                  {metric.trend === 'up' ? '↗️' : metric.trend === 'down' ? '↘️' : '➡️'}
                </span>
                <span style={{
                  padding: '3px 8px',
                  background: getMetricColor(metric.status),
                  color: '#000',
                  borderRadius: '4px',
                  fontSize: '0.7rem',
                  fontWeight: 'bold',
                  textTransform: 'uppercase'
                }}>
                  {metric.status}
                </span>
              </div>
            </div>
            
            {/* Current Value */}
            <div style={{
              display: 'flex',
              alignItems: 'baseline',
              gap: '8px',
              marginBottom: '15px'
            }}>
              <span style={{
                fontSize: '2.5rem',
                fontWeight: 'bold',
                color: getMetricColor(metric.status)
              }}>
                {metric.current.toFixed(1)}
              </span>
              <span style={{
                fontSize: '1.2rem',
                opacity: 0.8
              }}>
                {metric.unit}
              </span>
            </div>
            
            {/* Target vs Current */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '15px',
              fontSize: '0.9rem'
            }}>
              <span>Target: <strong>{metric.target}{metric.unit}</strong></span>
              <span style={{
                color: metric.current <= metric.target ? '#00FF00' : '#FFFF00'
              }}>
                {metric.current <= metric.target ? '✓ Within Target' : `${(((metric.current - metric.target) / metric.target) * 100).toFixed(1)}% over`}
              </span>
            </div>
            
            {/* Mini Chart */}
            <div style={{
              height: '40px',
              background: 'rgba(0,0,0,0.5)',
              borderRadius: '8px',
              padding: '5px',
              position: 'relative',
              overflow: 'hidden'
            }}>
              <svg width="100%" height="100%" style={{ display: 'block' }}>
                <polyline
                  fill="none"
                  stroke={getMetricColor(metric.status)}
                  strokeWidth="2"
                  points={metric.history.map((value, i) => {
                    const x = (i / (metric.history.length - 1)) * 100;
                    const y = 100 - (value / Math.max(...metric.history)) * 100;
                    return `${x}%,${y}%`;
                  }).join(' ')}
                />
              </svg>
            </div>
            
            {/* Pulse Animation */}
            <div style={{
              position: 'absolute',
              top: '15px',
              right: '15px',
              width: '10px',
              height: '10px',
              background: getMetricColor(metric.status),
              borderRadius: '50%',
              animation: 'metricPulse 2s infinite'
            }} />
          </div>
        ))}
      </motion.section>

      {/* System Resources */}
      <motion.section
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.4 }}
        style={{
          background: 'rgba(0,0,0,0.8)',
          border: '2px solid #FF00FF',
          borderRadius: '15px',
          padding: '30px',
          marginBottom: '30px',
          position: 'relative',
          zIndex: 10
        }}
      >
        <h2 style={{ color: '#FF00FF', marginBottom: '25px' }}>SYSTEM RESOURCES</h2>
        
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '20px'
        }}>
          {systemResources.map((resource, index) => (
            <motion.div
              key={resource.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5 + index * 0.1 }}
              whileHover={{ scale: 1.02 }}
              style={{
                background: 'rgba(0,0,0,0.9)',
                border: '2px solid #FF00FF',
                borderRadius: '12px',
                padding: '20px',
                cursor: 'pointer'
              }}
              onClick={() => setSelectedResource(resource)}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                <h3 style={{ margin: 0, color: '#00FFFF' }}>{resource.name}</h3>
                <div style={{ fontSize: '2rem' }}>
                  {resource.type === 'cpu' ? '💻' : 
                   resource.type === 'memory' ? '🧠' :
                   resource.type === 'disk' ? '💾' : '🌐'}
                </div>
              </div>
              
              {/* Usage Bar */}
              <div style={{
                background: 'rgba(0,0,0,0.5)',
                borderRadius: '10px',
                padding: '3px',
                marginBottom: '15px'
              }}>
                <div style={{
                  width: `${resource.usage}%`,
                  height: '20px',
                  background: resource.usage > 80 ? 'linear-gradient(90deg, #FF0040, #FF6600)' : 
                             resource.usage > 60 ? 'linear-gradient(90deg, #FFFF00, #FF6600)' :
                             'linear-gradient(90deg, #00FF00, #00FFFF)',
                  borderRadius: '8px',
                  transition: 'all 0.3s ease',
                  boxShadow: resource.usage > 80 ? '0 0 15px #FF0040' :
                            resource.usage > 60 ? '0 0 15px #FFFF00' :
                            '0 0 15px #00FF00'
                }} />
              </div>
              
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '15px'
              }}>
                <span style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#FFFF00' }}>
                  {resource.usage.toFixed(1)}%
                </span>
                {resource.temperature && (
                  <span style={{
                    color: resource.temperature > 70 ? '#FF0040' : resource.temperature > 60 ? '#FFFF00' : '#00FF00'
                  }}>
                    🌡️ {resource.temperature.toFixed(1)}°C
                  </span>
                )}
              </div>
              
              {/* Top Processes */}
              <div>
                <h4 style={{ margin: '0 0 10px 0', color: '#FFFF00', fontSize: '0.9rem' }}>Top Processes</h4>
                {resource.processes.slice(0, 3).map(process => (
                  <div key={process.name} style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '5px 0',
                    fontSize: '0.8rem'
                  }}>
                    <span>{process.name}</span>
                    <span style={{ color: '#00FFFF' }}>{process.usage}%</span>
                  </div>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Performance Visualization */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        style={{
          background: 'rgba(0,0,0,0.8)',
          border: '2px solid #FFFF00',
          borderRadius: '15px',
          padding: '30px',
          marginBottom: '30px',
          position: 'relative',
          zIndex: 10
        }}
      >
        <h2 style={{ color: '#FFFF00', marginBottom: '25px' }}>PERFORMANCE ANALYTICS</h2>
        
        <canvas
          ref={canvasRef}
          width={1000}
          height={300}
          style={{
            width: '100%',
            height: '300px',
            border: '1px solid #333',
            borderRadius: '8px',
            background: 'rgba(0,0,0,0.5)'
          }}
        />
      </motion.section>

      {/* Optimization Rules */}
      <motion.section
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        style={{
          background: 'rgba(0,0,0,0.8)',
          border: '2px solid #00FF00',
          borderRadius: '15px',
          padding: '30px',
          marginBottom: '30px',
          position: 'relative',
          zIndex: 10
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '25px' }}>
          <h2 style={{ color: '#00FF00', margin: 0 }}>OPTIMIZATION RULES</h2>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input
                type="checkbox"
                checked={autoOptimize}
                onChange={(e) => setAutoOptimize(e.target.checked)}
              />
              <span>Auto-optimize</span>
            </label>
          </div>
        </div>
        
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
          gap: '20px'
        }}>
          {optimizationRules.map((rule, index) => (
            <motion.div
              key={rule.id}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.9 + index * 0.05 }}
              style={{
                background: 'rgba(0,0,0,0.9)',
                border: `2px solid ${rule.enabled ? getPriorityColor(rule.priority) : '#666'}`,
                borderRadius: '12px',
                padding: '20px',
                opacity: rule.enabled ? 1 : 0.7
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '15px' }}>
                <div>
                  <h3 style={{ margin: 0, color: '#00FFFF', fontSize: '1.1rem' }}>{rule.name}</h3>
                  <div style={{ display: 'flex', gap: '10px', marginTop: '8px' }}>
                    <span style={{
                      padding: '3px 8px',
                      background: getPriorityColor(rule.priority),
                      color: '#000',
                      borderRadius: '4px',
                      fontSize: '0.7rem',
                      fontWeight: 'bold',
                      textTransform: 'uppercase'
                    }}>
                      {rule.priority}
                    </span>
                    <span style={{
                      padding: '3px 8px',
                      background: 'rgba(255,255,0,0.2)',
                      color: '#FFFF00',
                      borderRadius: '4px',
                      fontSize: '0.7rem',
                      textTransform: 'capitalize'
                    }}>
                      {rule.category}
                    </span>
                  </div>
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{
                    fontSize: '1.2rem',
                    fontWeight: 'bold',
                    color: '#00FF00'
                  }}>
                    +{rule.impact}%
                  </span>
                  <button
                    onClick={() => toggleOptimizationRule(rule.id)}
                    style={{
                      padding: '6px 12px',
                      background: rule.enabled ? 'linear-gradient(45deg, #00FF00, #00FFFF)' : 'rgba(255,0,64,0.2)',
                      border: 'none',
                      borderRadius: '6px',
                      color: rule.enabled ? '#000' : '#FF0040',
                      fontWeight: 'bold',
                      cursor: 'pointer',
                      fontSize: '0.8rem'
                    }}
                  >
                    {rule.enabled ? 'ENABLED' : 'DISABLED'}
                  </button>
                </div>
              </div>
              
              <p style={{
                margin: '0 0 15px 0',
                fontSize: '0.9rem',
                opacity: 0.8,
                lineHeight: 1.4
              }}>
                {rule.description}
              </p>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                {rule.lastApplied ? (
                  <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>
                    Last applied: {rule.lastApplied.toLocaleString()}
                  </span>
                ) : (
                  <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>Never applied</span>
                )}
                
                <button
                  onClick={() => runOptimization(rule.id)}
                  disabled={!rule.enabled || isAnalyzing}
                  style={{
                    padding: '6px 12px',
                    background: rule.enabled && !isAnalyzing ? 'linear-gradient(45deg, #FF00FF, #FFFF00)' : 'rgba(100,100,100,0.3)',
                    border: 'none',
                    borderRadius: '6px',
                    color: rule.enabled && !isAnalyzing ? '#000' : '#666',
                    fontWeight: 'bold',
                    cursor: rule.enabled && !isAnalyzing ? 'pointer' : 'not-allowed',
                    fontSize: '0.8rem'
                  }}
                >
                  APPLY NOW
                </button>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Bottleneck Analysis */}
      <motion.section
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.0 }}
        style={{
          background: 'rgba(0,0,0,0.8)',
          border: '2px solid #FF0040',
          borderRadius: '15px',
          padding: '30px',
          position: 'relative',
          zIndex: 10
        }}
      >
        <h2 style={{ color: '#FF0040', marginBottom: '25px' }}>BOTTLENECK ANALYSIS</h2>
        
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
          gap: '20px'
        }}>
          {bottlenecks.map((bottleneck, index) => (
            <motion.div
              key={bottleneck.component}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 1.1 + index * 0.1 }}
              style={{
                background: 'rgba(0,0,0,0.9)',
                border: `2px solid ${getSeverityColor(bottleneck.severity)}`,
                borderRadius: '12px',
                padding: '25px',
                position: 'relative'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '15px' }}>
                <h3 style={{ margin: 0, color: '#00FFFF', fontSize: '1.2rem' }}>{bottleneck.component}</h3>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px'
                }}>
                  <div style={{
                    width: '60px',
                    height: '60px',
                    borderRadius: '50%',
                    background: `conic-gradient(${getSeverityColor(bottleneck.severity)} ${bottleneck.impact * 3.6}deg, rgba(255,255,255,0.1) 0deg)`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '1rem',
                    fontWeight: 'bold',
                    color: '#fff'
                  }}>
                    {bottleneck.impact}%
                  </div>
                  <span style={{
                    padding: '6px 12px',
                    background: getSeverityColor(bottleneck.severity),
                    color: '#000',
                    borderRadius: '6px',
                    fontSize: '0.8rem',
                    fontWeight: 'bold',
                    textTransform: 'uppercase'
                  }}>
                    {bottleneck.severity}
                  </span>
                </div>
              </div>
              
              <div style={{ marginBottom: '15px' }}>
                <h4 style={{ margin: '0 0 8px 0', color: '#FFFF00', fontSize: '0.9rem' }}>Recommendation:</h4>
                <p style={{ margin: 0, fontSize: '0.9rem', opacity: 0.9, lineHeight: 1.4 }}>
                  {bottleneck.recommendation}
                </p>
              </div>
              
              <div style={{
                padding: '12px',
                background: 'rgba(0,255,0,0.1)',
                border: '1px solid #00FF00',
                borderRadius: '8px',
                fontSize: '0.9rem'
              }}>
                <strong style={{ color: '#00FF00' }}>Estimated Gain:</strong> {bottleneck.estimatedGain}
              </div>
            </motion.div>
          ))}
        </div>
      </motion.section>

      <style jsx>{`
        @keyframes performanceGlow {
          0% { filter: hue-rotate(0deg) brightness(1); }
          100% { filter: hue-rotate(90deg) brightness(1.3); }
        }
        
        @keyframes metricPulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.7; transform: scale(1.3); }
        }
      `}</style>
    </div>
  );
};

export default PerformanceOptimizer;