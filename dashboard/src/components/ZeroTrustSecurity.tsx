import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface SecurityEvent {
  id: string;
  timestamp: Date;
  type: 'authentication' | 'authorization' | 'breach' | 'scan' | 'access';
  severity: 'low' | 'medium' | 'high' | 'critical';
  source: string;
  target: string;
  description: string;
  status: 'resolved' | 'investigating' | 'open';
  metadata: Record<string, any>;
}

interface SecurityMetrics {
  threatLevel: number;
  authenticatedUsers: number;
  blockedAttempts: number;
  successfulLogins: number;
  failedLogins: number;
  activeScans: number;
  vulnerabilities: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
}

interface NetworkNode {
  id: string;
  name: string;
  type: 'endpoint' | 'server' | 'gateway' | 'database' | 'application';
  status: 'secure' | 'warning' | 'compromised' | 'scanning';
  trustScore: number;
  lastVerified: Date;
  connections: string[];
  position: { x: number; y: number };
}

const ZeroTrustSecurity: React.FC = () => {
  const [securityEvents, setSecurityEvents] = useState<SecurityEvent[]>([]);
  const [metrics, setMetrics] = useState<SecurityMetrics>({
    threatLevel: 0,
    authenticatedUsers: 0,
    blockedAttempts: 0,
    successfulLogins: 0,
    failedLogins: 0,
    activeScans: 0,
    vulnerabilities: { critical: 0, high: 0, medium: 0, low: 0 }
  });
  const [networkNodes, setNetworkNodes] = useState<NetworkNode[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<SecurityEvent | null>(null);
  const [scanningMode, setScanningMode] = useState(false);
  const [realTimeMode, setRealTimeMode] = useState(true);
  const [threatAnalysis, setThreatAnalysis] = useState('');
  const [encryptionStatus, setEncryptionStatus] = useState({
    inTransit: true,
    atRest: true,
    endToEnd: true,
    quantumReady: false
  });
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    initializeSecuritySystem();
    setupWebSocket();
    startNetworkVisualization();
    
    if (realTimeMode) {
      const interval = setInterval(generateSecurityEvent, 3000 + Math.random() * 7000);
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

  const initializeSecuritySystem = () => {
    // Initialize network topology
    const nodes: NetworkNode[] = [
      {
        id: 'gateway',
        name: 'Security Gateway',
        type: 'gateway',
        status: 'secure',
        trustScore: 95,
        lastVerified: new Date(),
        connections: ['web-server', 'api-server', 'database'],
        position: { x: 400, y: 200 }
      },
      {
        id: 'web-server',
        name: 'Web Server',
        type: 'server',
        status: 'secure',
        trustScore: 88,
        lastVerified: new Date(),
        connections: ['gateway', 'api-server'],
        position: { x: 200, y: 100 }
      },
      {
        id: 'api-server',
        name: 'API Server',
        type: 'server',
        status: 'warning',
        trustScore: 72,
        lastVerified: new Date(Date.now() - 300000),
        connections: ['gateway', 'web-server', 'database'],
        position: { x: 600, y: 100 }
      },
      {
        id: 'database',
        name: 'Database',
        type: 'database',
        status: 'secure',
        trustScore: 92,
        lastVerified: new Date(),
        connections: ['gateway', 'api-server'],
        position: { x: 400, y: 350 }
      },
      {
        id: 'media-app',
        name: 'Media Application',
        type: 'application',
        status: 'secure',
        trustScore: 85,
        lastVerified: new Date(),
        connections: ['api-server'],
        position: { x: 150, y: 300 }
      },
      {
        id: 'admin-endpoint',
        name: 'Admin Endpoint',
        type: 'endpoint',
        status: 'scanning',
        trustScore: 78,
        lastVerified: new Date(Date.now() - 600000),
        connections: ['gateway'],
        position: { x: 650, y: 300 }
      }
    ];
    
    setNetworkNodes(nodes);
    
    // Initialize security metrics
    setMetrics({
      threatLevel: Math.round(Math.random() * 30 + 10),
      authenticatedUsers: Math.round(Math.random() * 50 + 20),
      blockedAttempts: Math.round(Math.random() * 100 + 50),
      successfulLogins: Math.round(Math.random() * 200 + 100),
      failedLogins: Math.round(Math.random() * 50 + 10),
      activeScans: Math.round(Math.random() * 5 + 2),
      vulnerabilities: {
        critical: Math.round(Math.random() * 3),
        high: Math.round(Math.random() * 8 + 2),
        medium: Math.round(Math.random() * 15 + 5),
        low: Math.round(Math.random() * 25 + 10)
      }
    });
    
    // Generate initial security events
    const events: SecurityEvent[] = [];
    for (let i = 0; i < 10; i++) {
      events.push(generateRandomSecurityEvent());
    }
    setSecurityEvents(events);
  };

  const setupWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8080/security/events');
      
      wsRef.current.onmessage = (event) => {
        const securityEvent = JSON.parse(event.data);
        handleSecurityEvent(securityEvent);
      };
      
      wsRef.current.onerror = () => {
        console.warn('Security WebSocket connection failed');
      };
    } catch (error) {
      console.warn('WebSocket not available for security events');
    }
  };

  const handleSecurityEvent = (event: SecurityEvent) => {
    setSecurityEvents(prev => [event, ...prev.slice(0, 49)]);
    
    // Update threat analysis
    if (event.severity === 'critical' || event.severity === 'high') {
      setThreatAnalysis(`HIGH ALERT: ${event.description} detected from ${event.source}`);
      setTimeout(() => setThreatAnalysis(''), 10000);
    }
    
    // Update network nodes if affected
    if (event.type === 'breach' || event.type === 'scan') {
      setNetworkNodes(prev => prev.map(node => 
        node.name === event.target
          ? { ...node, status: 'warning', trustScore: Math.max(0, node.trustScore - 10) }
          : node
      ));
    }
  };

  const generateRandomSecurityEvent = (): SecurityEvent => {
    const types: SecurityEvent['type'][] = ['authentication', 'authorization', 'breach', 'scan', 'access'];
    const severities: SecurityEvent['severity'][] = ['low', 'medium', 'high', 'critical'];
    const statuses: SecurityEvent['status'][] = ['resolved', 'investigating', 'open'];
    
    const sources = ['192.168.1.100', '10.0.0.50', 'user@example.com', 'admin@localhost', 'unknown'];
    const targets = ['Web Server', 'API Server', 'Database', 'Media Application', 'Admin Panel'];
    
    const descriptions = {
      authentication: [
        'Multi-factor authentication challenge',
        'Failed login attempt detected',
        'Successful biometric verification',
        'Password change request'
      ],
      authorization: [
        'Privilege escalation attempt',
        'Unauthorized resource access',
        'Permission granted after verification',
        'Role-based access control violation'
      ],
      breach: [
        'Potential data exfiltration detected',
        'Unusual network traffic patterns',
        'Suspicious file access behavior',
        'Malware signature detected'
      ],
      scan: [
        'Port scan activity detected',
        'Vulnerability assessment completed',
        'Network reconnaissance attempt',
        'Security health check initiated'
      ],
      access: [
        'Resource access granted',
        'New device registration',
        'Session established',
        'API endpoint accessed'
      ]
    };
    
    const type = types[Math.floor(Math.random() * types.length)];
    const severity = severities[Math.floor(Math.random() * severities.length)];
    
    return {
      id: Date.now().toString() + Math.random(),
      timestamp: new Date(Date.now() - Math.random() * 3600000),
      type,
      severity,
      source: sources[Math.floor(Math.random() * sources.length)],
      target: targets[Math.floor(Math.random() * targets.length)],
      description: descriptions[type][Math.floor(Math.random() * descriptions[type].length)],
      status: statuses[Math.floor(Math.random() * statuses.length)],
      metadata: {
        timestamp: Date.now(),
        confidence: Math.random() * 100,
        risk: Math.random() * 10
      }
    };
  };

  const generateSecurityEvent = () => {
    const event = generateRandomSecurityEvent();
    handleSecurityEvent(event);
  };

  const startNetworkVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw connections
      networkNodes.forEach(node => {
        node.connections.forEach(connectionId => {
          const targetNode = networkNodes.find(n => n.id === connectionId);
          if (targetNode) {
            drawConnection(ctx, node, targetNode);
          }
        });
      });
      
      // Draw nodes
      networkNodes.forEach(node => {
        drawNode(ctx, node);
      });
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
  };

  const drawConnection = (ctx: CanvasRenderingContext2D, from: NetworkNode, to: NetworkNode) => {
    const gradient = ctx.createLinearGradient(from.position.x, from.position.y, to.position.x, to.position.y);
    gradient.addColorStop(0, getNodeColor(from.status));
    gradient.addColorStop(1, getNodeColor(to.status));
    
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.6;
    
    ctx.beginPath();
    ctx.moveTo(from.position.x, from.position.y);
    ctx.lineTo(to.position.x, to.position.y);
    ctx.stroke();
    
    ctx.globalAlpha = 1;
  };

  const drawNode = (ctx: CanvasRenderingContext2D, node: NetworkNode) => {
    const color = getNodeColor(node.status);
    
    // Draw node circle
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    
    ctx.beginPath();
    ctx.arc(node.position.x, node.position.y, 20, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    
    // Draw pulsing effect for active nodes
    if (node.status === 'scanning') {
      const pulseRadius = 20 + Math.sin(Date.now() / 200) * 10;
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.3;
      ctx.lineWidth = 2;
      
      ctx.beginPath();
      ctx.arc(node.position.x, node.position.y, pulseRadius, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    
    // Draw node label
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '12px Orbitron';
    ctx.textAlign = 'center';
    ctx.fillText(node.name, node.position.x, node.position.y + 40);
    
    // Draw trust score
    ctx.fillStyle = color;
    ctx.font = '10px Orbitron';
    ctx.fillText(`${node.trustScore}%`, node.position.x, node.position.y + 5);
  };

  const getNodeColor = (status: string) => {
    switch (status) {
      case 'secure': return '#00FF00';
      case 'warning': return '#FFFF00';
      case 'compromised': return '#FF0040';
      case 'scanning': return '#00FFFF';
      default: return '#666666';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return '#FF0040';
      case 'high': return '#FF6B00';
      case 'medium': return '#FFFF00';
      case 'low': return '#00FF00';
      default: return '#666666';
    }
  };

  const getThreatLevelColor = (level: number) => {
    if (level < 20) return '#00FF00';
    if (level < 40) return '#FFFF00';
    if (level < 60) return '#FF6B00';
    return '#FF0040';
  };

  const performSecurityScan = async () => {
    setScanningMode(true);
    
    // Update all nodes to scanning status
    setNetworkNodes(prev => prev.map(node => ({ ...node, status: 'scanning' })));
    
    // Simulate scan progress
    for (let i = 0; i < networkNodes.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setNetworkNodes(prev => prev.map((node, index) => {
        if (index === i) {
          const newTrustScore = Math.max(70, Math.min(100, node.trustScore + Math.random() * 20 - 5));
          return {
            ...node,
            status: 'secure',
            trustScore: newTrustScore,
            lastVerified: new Date()
          };
        }
        return node;
      }));
    }
    
    setScanningMode(false);
    
    // Generate scan completion event
    const scanEvent: SecurityEvent = {
      id: Date.now().toString(),
      timestamp: new Date(),
      type: 'scan',
      severity: 'low',
      source: 'Security System',
      target: 'All Nodes',
      description: 'Comprehensive security scan completed successfully',
      status: 'resolved',
      metadata: { scanDuration: networkNodes.length * 1000 }
    };
    
    handleSecurityEvent(scanEvent);
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
      {/* Security Grid Background */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundImage: 'linear-gradient(rgba(0,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,255,0.1) 1px, transparent 1px)',
        backgroundSize: '30px 30px',
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
            background: 'linear-gradient(45deg, #00FFFF, #FF0040)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            textShadow: '0 0 20px #FF0040',
            animation: 'securityPulse 2s infinite'
          }}>
            ZERO TRUST SECURITY
          </h1>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '20px',
            marginTop: '10px'
          }}>
            <div style={{
              padding: '8px 15px',
              background: `linear-gradient(45deg, ${getThreatLevelColor(metrics.threatLevel)}, #000)`,
              borderRadius: '20px',
              fontSize: '0.9rem',
              fontWeight: 'bold'
            }}>
              THREAT LEVEL: {metrics.threatLevel}%
            </div>
            <div style={{ fontSize: '0.9rem', opacity: 0.8 }}>
              {metrics.authenticatedUsers} Active Users | {metrics.activeScans} Active Scans
            </div>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '15px' }}>
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
            onClick={performSecurityScan}
            disabled={scanningMode}
            style={{
              padding: '10px 20px',
              background: scanningMode ? 'rgba(255,255,0,0.3)' : 'linear-gradient(45deg, #FFFF00, #FF00FF)',
              border: 'none',
              borderRadius: '8px',
              color: scanningMode ? '#FFFF00' : '#000',
              fontWeight: 'bold',
              cursor: 'pointer'
            }}
          >
            {scanningMode ? 'SCANNING...' : '🔍 FULL SCAN'}
          </button>
        </div>
      </motion.header>

      {/* Security Metrics Dashboard */}
      <motion.section
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2 }}
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '20px',
          marginBottom: '30px'
        }}
      >
        {[
          { label: 'Blocked Attempts', value: metrics.blockedAttempts, color: '#FF0040', icon: '🛡️' },
          { label: 'Successful Logins', value: metrics.successfulLogins, color: '#00FF00', icon: '✓' },
          { label: 'Failed Logins', value: metrics.failedLogins, color: '#FFFF00', icon: '✗' },
          { label: 'Critical Vulnerabilities', value: metrics.vulnerabilities.critical, color: '#FF0040', icon: '⚠️' }
        ].map((metric, index) => (
          <div
            key={metric.label}
            style={{
              background: 'rgba(0,0,0,0.8)',
              border: `2px solid ${metric.color}`,
              borderRadius: '12px',
              padding: '20px',
              textAlign: 'center',
              position: 'relative',
              overflow: 'hidden',
              boxShadow: `0 0 20px rgba(${metric.color === '#FF0040' ? '255,0,64' : metric.color === '#00FF00' ? '0,255,0' : '255,255,0'},0.3)`
            }}
          >
            <div style={{ fontSize: '2rem', marginBottom: '10px' }}>{metric.icon}</div>
            <div style={{
              fontSize: '2.5rem',
              fontWeight: 'bold',
              color: metric.color,
              marginBottom: '5px'
            }}>
              {metric.value}
            </div>
            <div style={{ fontSize: '0.9rem', opacity: 0.8 }}>{metric.label}</div>
            
            {/* Animated background effect */}
            <div style={
              {
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              height: '4px',
              background: `linear-gradient(90deg, transparent, ${metric.color}, transparent)`,
              animation: 'securityScan 3s linear infinite'
            }} />
          </div>
        ))}
      </motion.section>

      {/* Threat Analysis Alert */}
      <AnimatePresence>
        {threatAnalysis && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            style={{
              background: 'linear-gradient(45deg, #FF0040, #FF00FF)',
              color: '#000',
              padding: '15px 25px',
              borderRadius: '8px',
              marginBottom: '30px',
              fontWeight: 'bold',
              textAlign: 'center',
              animation: 'criticalAlert 1s infinite alternate'
            }}
          >
            ⚠️ {threatAnalysis}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Network Topology Visualization */}
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        style={{
          background: 'rgba(0,0,0,0.8)',
          border: '2px solid #00FFFF',
          borderRadius: '15px',
          padding: '30px',
          marginBottom: '30px',
          position: 'relative'
        }}
      >
        <h2 style={{ color: '#00FFFF', marginBottom: '20px', textAlign: 'center' }}>NETWORK TOPOLOGY</h2>
        
        <canvas
          ref={canvasRef}
          width={800}
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
          gap: '30px',
          marginTop: '20px',
          fontSize: '0.9rem'
        }}>
          {[
            { status: 'secure', color: '#00FF00', label: 'Secure' },
            { status: 'warning', color: '#FFFF00', label: 'Warning' },
            { status: 'compromised', color: '#FF0040', label: 'Compromised' },
            { status: 'scanning', color: '#00FFFF', label: 'Scanning' }
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

      {/* Security Events Log */}
      <motion.section
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        style={{
          background: 'rgba(0,0,0,0.8)',
          border: '2px solid #FF00FF',
          borderRadius: '15px',
          padding: '30px'
        }}
      >
        <h2 style={{ color: '#FF00FF', marginBottom: '20px' }}>SECURITY EVENTS LOG</h2>
        
        <div style={{
          maxHeight: '400px',
          overflowY: 'auto',
          border: '1px solid #333',
          borderRadius: '8px',
          background: 'rgba(0,0,0,0.5)'
        }}>
          <AnimatePresence>
            {securityEvents.map((event, index) => (
              <motion.div
                key={event.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ delay: index * 0.05 }}
                style={{
                  padding: '15px 20px',
                  borderBottom: '1px solid #333',
                  cursor: 'pointer',
                  background: selectedEvent?.id === event.id ? 'rgba(0,255,255,0.1)' : 'transparent'
                }}
                onClick={() => setSelectedEvent(event)}
                whileHover={{ background: 'rgba(0,255,255,0.05)' }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '5px' }}>
                      <span style={{
                        padding: '3px 8px',
                        background: getSeverityColor(event.severity),
                        color: '#000',
                        borderRadius: '4px',
                        fontSize: '0.8rem',
                        fontWeight: 'bold',
                        textTransform: 'uppercase'
                      }}>
                        {event.severity}
                      </span>
                      <span style={{ color: '#FFFF00', textTransform: 'uppercase' }}>{event.type}</span>
                      <span style={{ opacity: 0.7, fontSize: '0.9rem' }}>{event.timestamp.toLocaleTimeString()}</span>
                    </div>
                    <div style={{ marginBottom: '5px' }}>{event.description}</div>
                    <div style={{ fontSize: '0.8rem', opacity: 0.7 }}>
                      {event.source} → {event.target}
                    </div>
                  </div>
                  
                  <div style={{
                    padding: '4px 8px',
                    background: event.status === 'resolved' ? 'rgba(0,255,0,0.2)' : event.status === 'investigating' ? 'rgba(255,255,0,0.2)' : 'rgba(255,0,64,0.2)',
                    border: `1px solid ${event.status === 'resolved' ? '#00FF00' : event.status === 'investigating' ? '#FFFF00' : '#FF0040'}`,
                    borderRadius: '4px',
                    fontSize: '0.8rem',
                    textTransform: 'uppercase'
                  }}>
                    {event.status}
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </motion.section>

      {/* Encryption Status */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          background: 'rgba(0,0,0,0.9)',
          border: '2px solid #FFFF00',
          borderRadius: '12px',
          padding: '15px',
          minWidth: '250px'
        }}
      >
        <h4 style={{ color: '#FFFF00', marginBottom: '10px' }}>ENCRYPTION STATUS</h4>
        {Object.entries(encryptionStatus).map(([key, status]) => (
          <div key={key} style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '8px',
            fontSize: '0.9rem'
          }}>
            <span style={{ textTransform: 'capitalize' }}>{key.replace(/([A-Z])/g, ' $1')}</span>
            <span style={{
              color: status ? '#00FF00' : '#FF0040',
              fontWeight: 'bold'
            }}>
              {status ? '✓ ACTIVE' : '✗ INACTIVE'}
            </span>
          </div>
        ))}
      </motion.div>

      <style jsx>{`
        @keyframes securityPulse {
          0%, 100% { text-shadow: 0 0 20px #FF0040; }
          50% { text-shadow: 0 0 40px #FF0040, 0 0 60px #FF0040; }
        }
        
        @keyframes securityScan {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        
        @keyframes criticalAlert {
          0% { box-shadow: 0 0 20px rgba(255,0,64,0.5); }
          100% { box-shadow: 0 0 40px rgba(255,0,64,1); }
        }
      `}</style>
    </div>
  );
};

export default ZeroTrustSecurity;