import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface SyncTask {
  id: string;
  type: 'download' | 'upload' | 'metadata' | 'thumbnails' | 'subtitles';
  title: string;
  progress: number;
  status: 'queued' | 'syncing' | 'completed' | 'failed' | 'paused';
  fileSize: number;
  syncedSize: number;
  speed: number;
  eta: number;
  device: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  createdAt: Date;
}

interface MobileDevice {
  id: string;
  name: string;
  type: 'phone' | 'tablet' | 'laptop';
  platform: 'ios' | 'android' | 'windows' | 'macos';
  status: 'online' | 'offline' | 'syncing' | 'limited';
  batteryLevel: number;
  storageUsed: number;
  storageTotal: number;
  connection: 'wifi' | 'cellular' | 'bluetooth';
  lastSync: Date;
  syncSettings: {
    autoSync: boolean;
    syncOnWifi: boolean;
    syncOnCellular: boolean;
    maxFileSize: number;
    quality: 'original' | 'high' | 'medium' | 'low';
  };
}

interface OfflineContent {
  id: string;
  title: string;
  type: 'movie' | 'episode' | 'music' | 'book';
  size: number;
  quality: string;
  downloadedAt: Date;
  expiresAt?: Date;
  viewCount: number;
  deviceId: string;
}

const MobileSync: React.FC = () => {
  const [devices, setDevices] = useState<MobileDevice[]>([]);
  const [syncTasks, setSyncTasks] = useState<SyncTask[]>([]);
  const [offlineContent, setOfflineContent] = useState<OfflineContent[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<MobileDevice | null>(null);
  const [syncMode, setSyncMode] = useState<'auto' | 'manual' | 'scheduled'>('auto');
  const [isQRScannerOpen, setIsQRScannerOpen] = useState(false);
  const [networkStats, setNetworkStats] = useState({
    totalSync: 0,
    dailySync: 0,
    activeDownloads: 0,
    bandwidth: 0
  });
  const [syncAnalytics, setSyncAnalytics] = useState({
    successRate: 0,
    avgSpeed: 0,
    totalDevices: 0,
    contentSynced: 0
  });
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const qrRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    initializeMobileSync();
    setupWebSocket();
    startSyncVisualization();
    generateQRCode();
    
    const interval = setInterval(updateSyncProgress, 1000);
    return () => clearInterval(interval);
  }, []);

  const initializeMobileSync = () => {
    // Initialize mobile devices
    const mobileDevices: MobileDevice[] = [
      {
        id: 'iphone-12',
        name: 'iPhone 12 Pro',
        type: 'phone',
        platform: 'ios',
        status: 'syncing',
        batteryLevel: 78,
        storageUsed: 45,
        storageTotal: 128,
        connection: 'wifi',
        lastSync: new Date(Date.now() - 300000),
        syncSettings: {
          autoSync: true,
          syncOnWifi: true,
          syncOnCellular: false,
          maxFileSize: 2000,
          quality: 'high'
        }
      },
      {
        id: 'samsung-tab',
        name: 'Galaxy Tab S8',
        type: 'tablet',
        platform: 'android',
        status: 'online',
        batteryLevel: 92,
        storageUsed: 67,
        storageTotal: 256,
        connection: 'wifi',
        lastSync: new Date(Date.now() - 1800000),
        syncSettings: {
          autoSync: true,
          syncOnWifi: true,
          syncOnCellular: true,
          maxFileSize: 5000,
          quality: 'original'
        }
      },
      {
        id: 'macbook-pro',
        name: 'MacBook Pro 16"',
        type: 'laptop',
        platform: 'macos',
        status: 'limited',
        batteryLevel: 34,
        storageUsed: 89,
        storageTotal: 512,
        connection: 'wifi',
        lastSync: new Date(Date.now() - 7200000),
        syncSettings: {
          autoSync: false,
          syncOnWifi: true,
          syncOnCellular: false,
          maxFileSize: 10000,
          quality: 'original'
        }
      },
      {
        id: 'pixel-7',
        name: 'Pixel 7 Pro',
        type: 'phone',
        platform: 'android',
        status: 'offline',
        batteryLevel: 15,
        storageUsed: 23,
        storageTotal: 128,
        connection: 'cellular',
        lastSync: new Date(Date.now() - 86400000),
        syncSettings: {
          autoSync: true,
          syncOnWifi: true,
          syncOnCellular: false,
          maxFileSize: 1000,
          quality: 'medium'
        }
      }
    ];
    
    // Initialize sync tasks
    const tasks: SyncTask[] = [
      {
        id: 'sync-1',
        type: 'download',
        title: 'Avengers: Endgame (2019)',
        progress: 67,
        status: 'syncing',
        fileSize: 4200,
        syncedSize: 2814,
        speed: 15.6,
        eta: 89,
        device: 'iphone-12',
        priority: 'high',
        createdAt: new Date(Date.now() - 1800000)
      },
      {
        id: 'sync-2',
        type: 'metadata',
        title: 'TV Show Library Sync',
        progress: 100,
        status: 'completed',
        fileSize: 12,
        syncedSize: 12,
        speed: 0,
        eta: 0,
        device: 'samsung-tab',
        priority: 'medium',
        createdAt: new Date(Date.now() - 3600000)
      },
      {
        id: 'sync-3',
        type: 'thumbnails',
        title: 'Media Thumbnails',
        progress: 45,
        status: 'syncing',
        fileSize: 156,
        syncedSize: 70,
        speed: 2.3,
        eta: 37,
        device: 'samsung-tab',
        priority: 'low',
        createdAt: new Date(Date.now() - 900000)
      },
      {
        id: 'sync-4',
        type: 'upload',
        title: 'Watch History Backup',
        progress: 23,
        status: 'paused',
        fileSize: 8,
        syncedSize: 2,
        speed: 0,
        eta: 0,
        device: 'pixel-7',
        priority: 'medium',
        createdAt: new Date(Date.now() - 600000)
      },
      {
        id: 'sync-5',
        type: 'download',
        title: 'Stranger Things S4E9',
        progress: 0,
        status: 'queued',
        fileSize: 2800,
        syncedSize: 0,
        speed: 0,
        eta: 0,
        device: 'macbook-pro',
        priority: 'high',
        createdAt: new Date()
      }
    ];
    
    // Initialize offline content
    const content: OfflineContent[] = [
      {
        id: 'offline-1',
        title: 'The Matrix (1999)',
        type: 'movie',
        size: 3500,
        quality: 'High (1080p)',
        downloadedAt: new Date(Date.now() - 86400000),
        viewCount: 2,
        deviceId: 'iphone-12'
      },
      {
        id: 'offline-2',
        title: 'Breaking Bad S1-S5',
        type: 'episode',
        size: 45000,
        quality: 'Original (4K)',
        downloadedAt: new Date(Date.now() - 172800000),
        expiresAt: new Date(Date.now() + 2592000000),
        viewCount: 15,
        deviceId: 'samsung-tab'
      },
      {
        id: 'offline-3',
        title: 'Synthwave Playlist',
        type: 'music',
        size: 850,
        quality: 'High (320kbps)',
        downloadedAt: new Date(Date.now() - 259200000),
        viewCount: 8,
        deviceId: 'pixel-7'
      }
    ];
    
    setDevices(mobileDevices);
    setSyncTasks(tasks);
    setOfflineContent(content);
    
    // Initialize network stats
    setNetworkStats({
      totalSync: 125.6,
      dailySync: 12.3,
      activeDownloads: tasks.filter(t => t.status === 'syncing').length,
      bandwidth: 45.8
    });
    
    // Initialize sync analytics
    setSyncAnalytics({
      successRate: 94.5,
      avgSpeed: 8.7,
      totalDevices: mobileDevices.length,
      contentSynced: content.length
    });
  };

  const setupWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8080/mobile-sync');
      
      wsRef.current.onmessage = (event) => {
        const update = JSON.parse(event.data);
        handleSyncUpdate(update);
      };
    } catch (error) {
      console.warn('Mobile sync WebSocket not available');
    }
  };

  const handleSyncUpdate = (update: any) => {
    if (update.type === 'sync_progress') {
      setSyncTasks(prev => prev.map(task => 
        task.id === update.taskId 
          ? { ...task, progress: update.progress, speed: update.speed, eta: update.eta }
          : task
      ));
    } else if (update.type === 'device_status') {
      setDevices(prev => prev.map(device => 
        device.id === update.deviceId 
          ? { ...device, status: update.status, batteryLevel: update.battery }
          : device
      ));
    }
  };

  const updateSyncProgress = () => {
    setSyncTasks(prev => prev.map(task => {
      if (task.status === 'syncing') {
        const increment = Math.random() * 3;
        const newProgress = Math.min(100, task.progress + increment);
        const newSyncedSize = (newProgress / 100) * task.fileSize;
        const newSpeed = task.speed + (Math.random() - 0.5) * 2;
        const remainingSize = task.fileSize - newSyncedSize;
        const newEta = newSpeed > 0 ? Math.round(remainingSize / newSpeed) : 0;
        
        return {
          ...task,
          progress: newProgress,
          syncedSize: newSyncedSize,
          speed: Math.max(0, newSpeed),
          eta: newEta,
          status: newProgress >= 100 ? 'completed' : 'syncing'
        };
      }
      return task;
    }));
    
    // Update network stats
    setNetworkStats(prev => ({
      ...prev,
      bandwidth: Math.max(0, prev.bandwidth + (Math.random() - 0.5) * 10),
      activeDownloads: syncTasks.filter(t => t.status === 'syncing').length
    }));
  };

  const startSyncVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const connections: Array<{
      x: number;
      y: number;
      targetX: number;
      targetY: number;
      progress: number;
      speed: number;
    }> = [];
    
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw central server
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      
      ctx.fillStyle = '#00FFFF';
      ctx.shadowColor = '#00FFFF';
      ctx.shadowBlur = 20;
      ctx.beginPath();
      ctx.arc(centerX, centerY, 30, 0, 2 * Math.PI);
      ctx.fill();
      ctx.shadowBlur = 0;
      
      // Draw devices
      const activeDevices = devices.filter(d => d.status !== 'offline');
      activeDevices.forEach((device, index) => {
        const angle = (index / activeDevices.length) * 2 * Math.PI;
        const radius = 150;
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;
        
        const color = device.status === 'syncing' ? '#FFFF00' : device.status === 'online' ? '#00FF00' : '#FF00FF';
        
        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 15;
        ctx.beginPath();
        ctx.arc(x, y, 15, 0, 2 * Math.PI);
        ctx.fill();
        ctx.shadowBlur = 0;
        
        // Draw connection line
        if (device.status === 'syncing') {
          ctx.strokeStyle = '#FFFF00';
          ctx.lineWidth = 2;
          ctx.globalAlpha = 0.6;
          ctx.beginPath();
          ctx.moveTo(centerX, centerY);
          ctx.lineTo(x, y);
          ctx.stroke();
          ctx.globalAlpha = 1;
          
          // Animate data packets
          if (Math.random() < 0.3) {
            connections.push({
              x: centerX,
              y: centerY,
              targetX: x,
              targetY: y,
              progress: 0,
              speed: 0.02 + Math.random() * 0.03
            });
          }
        }
      });
      
      // Animate data packets
      for (let i = connections.length - 1; i >= 0; i--) {
        const conn = connections[i];
        conn.progress += conn.speed;
        
        if (conn.progress >= 1) {
          connections.splice(i, 1);
          continue;
        }
        
        const x = conn.x + (conn.targetX - conn.x) * conn.progress;
        const y = conn.y + (conn.targetY - conn.y) * conn.progress;
        
        ctx.fillStyle = '#FF00FF';
        ctx.shadowColor = '#FF00FF';
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
      
      requestAnimationFrame(animate);
    };
    
    animate();
  };

  const generateQRCode = () => {
    const canvas = qrRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Simple QR code pattern (for demo)
    const size = canvas.width;
    const moduleSize = size / 25;
    
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, size, size);
    
    ctx.fillStyle = '#000000';
    
    // Create a simple pattern that looks like a QR code
    for (let row = 0; row < 25; row++) {
      for (let col = 0; col < 25; col++) {
        if (Math.random() < 0.5) {
          ctx.fillRect(col * moduleSize, row * moduleSize, moduleSize, moduleSize);
        }
      }
    }
    
    // Add position markers
    const markerSize = moduleSize * 7;
    const positions = [[0, 0], [18, 0], [0, 18]];
    
    ctx.fillStyle = '#000000';
    positions.forEach(([x, y]) => {
      ctx.fillRect(x * moduleSize, y * moduleSize, markerSize, markerSize);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect((x + 1) * moduleSize, (y + 1) * moduleSize, markerSize - 2 * moduleSize, markerSize - 2 * moduleSize);
      ctx.fillStyle = '#000000';
      ctx.fillRect((x + 2) * moduleSize, (y + 2) * moduleSize, markerSize - 4 * moduleSize, markerSize - 4 * moduleSize);
    });
  };

  const pauseTask = (taskId: string) => {
    setSyncTasks(prev => prev.map(task => 
      task.id === taskId ? { ...task, status: task.status === 'paused' ? 'syncing' : 'paused' } : task
    ));
  };

  const cancelTask = (taskId: string) => {
    setSyncTasks(prev => prev.filter(task => task.id !== taskId));
  };

  const updateDeviceSettings = (deviceId: string, settings: Partial<MobileDevice['syncSettings']>) => {
    setDevices(prev => prev.map(device => 
      device.id === deviceId 
        ? { ...device, syncSettings: { ...device.syncSettings, ...settings } }
        : device
    ));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return '#00FF00';
      case 'syncing': return '#FFFF00';
      case 'limited': return '#FF6600';
      case 'offline': return '#FF0040';
      default: return '#666666';
    }
  };

  const getTaskStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return '#00FF00';
      case 'syncing': return '#00FFFF';
      case 'queued': return '#FFFF00';
      case 'paused': return '#FF00FF';
      case 'failed': return '#FF0040';
      default: return '#666666';
    }
  };

  const getPlatformIcon = (platform: string) => {
    switch (platform) {
      case 'ios': return '🍎';
      case 'android': return '🤖';
      case 'windows': return '💻';
      case 'macos': return '🖥️';
      default: return '📱';
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const formatTime = (seconds: number) => {
    if (seconds === 0) return '0s';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
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
      {/* Mobile Network Background */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundImage: `
          radial-gradient(circle at 20% 20%, rgba(0,255,255,0.1) 1px, transparent 1px),
          radial-gradient(circle at 80% 80%, rgba(255,0,255,0.1) 1px, transparent 1px),
          radial-gradient(circle at 40% 70%, rgba(255,255,0,0.1) 1px, transparent 1px)
        `,
        backgroundSize: '60px 60px, 80px 80px, 100px 100px',
        opacity: 0.3,
        pointerEvents: 'none',
        animation: 'mobileFloat 10s linear infinite'
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
            animation: 'mobilePulse 4s ease-in-out infinite alternate'
          }}>
            MOBILE SYNC
          </h1>
          <div style={{
            display: 'flex',
            gap: '20px',
            marginTop: '10px',
            fontSize: '0.9rem'
          }}>
            <span>Mode: <strong style={{ color: '#FFFF00' }}>{syncMode.toUpperCase()}</strong></span>
            <span>Bandwidth: <strong style={{ color: '#00FF00' }}>{networkStats.bandwidth.toFixed(1)} MB/s</strong></span>
            <span>Success Rate: <strong style={{ color: '#FF00FF' }}>{syncAnalytics.successRate}%</strong></span>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <select
            value={syncMode}
            onChange={(e) => setSyncMode(e.target.value as any)}
            style={{
              padding: '12px',
              background: 'rgba(0,0,0,0.8)',
              border: '2px solid #00FFFF',
              borderRadius: '8px',
              color: '#00FFFF',
              fontSize: '1rem'
            }}
          >
            <option value="auto">Auto Sync</option>
            <option value="manual">Manual Sync</option>
            <option value="scheduled">Scheduled Sync</option>
          </select>
          
          <button
            onClick={() => setIsQRScannerOpen(!isQRScannerOpen)}
            style={{
              padding: '12px 20px',
              background: 'linear-gradient(45deg, #FF00FF, #FFFF00)',
              border: 'none',
              borderRadius: '8px',
              color: '#000',
              fontWeight: 'bold',
              cursor: 'pointer',
              fontSize: '1rem'
            }}
          >
            📱 ADD DEVICE
          </button>
        </div>
      </motion.header>

      {/* Sync Analytics Dashboard */}
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
          { label: 'Total Synced', value: `${networkStats.totalSync} GB`, color: '#00FFFF', icon: '⬇️' },
          { label: 'Active Downloads', value: networkStats.activeDownloads.toString(), color: '#FFFF00', icon: '🔄' },
          { label: 'Connected Devices', value: devices.filter(d => d.status !== 'offline').length.toString(), color: '#00FF00', icon: '📱' },
          { label: 'Avg Speed', value: `${syncAnalytics.avgSpeed} MB/s`, color: '#FF00FF', icon: '⚡' }
        ].map((metric, index) => (
          <div
            key={metric.label}
            style={{
              background: 'rgba(0,0,0,0.8)',
              border: `3px solid ${metric.color}`,
              borderRadius: '15px',
              padding: '25px',
              textAlign: 'center',
              position: 'relative',
              overflow: 'hidden',
              boxShadow: `0 0 30px rgba(${metric.color === '#00FFFF' ? '0,255,255' : metric.color === '#FFFF00' ? '255,255,0' : metric.color === '#00FF00' ? '0,255,0' : '255,0,255'},0.3)`
            }}
          >
            <div style={{ fontSize: '3rem', marginBottom: '15px' }}>{metric.icon}</div>
            <div style={{
              fontSize: '2.5rem',
              fontWeight: 'bold',
              color: metric.color,
              marginBottom: '8px'
            }}>
              {metric.value}
            </div>
            <div style={{ fontSize: '1rem', opacity: 0.8 }}>{metric.label}</div>
            
            {/* Animated wave effect */}
            <div style={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              height: '4px',
              background: `linear-gradient(90deg, transparent, ${metric.color}, transparent)`,
              animation: 'syncWave 3s linear infinite'
            }} />
          </div>
        ))}
      </motion.section>

      {/* Sync Visualization */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        style={{
          background: 'rgba(0,0,0,0.8)',
          border: '2px solid #00FFFF',
          borderRadius: '15px',
          padding: '30px',
          marginBottom: '30px',
          position: 'relative',
          zIndex: 10
        }}
      >
        <h2 style={{ color: '#00FFFF', marginBottom: '25px', textAlign: 'center' }}>SYNC NETWORK TOPOLOGY</h2>
        
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
      </motion.section>

      {/* Device Management */}
      <motion.section
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.6 }}
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
        <h2 style={{ color: '#FF00FF', marginBottom: '25px' }}>CONNECTED DEVICES</h2>
        
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
          gap: '20px'
        }}>
          {devices.map((device, index) => (
            <motion.div
              key={device.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.7 + index * 0.1 }}
              whileHover={{ scale: 1.02, y: -5 }}
              style={{
                background: 'rgba(0,0,0,0.9)',
                border: `2px solid ${getStatusColor(device.status)}`,
                borderRadius: '12px',
                padding: '25px',
                cursor: 'pointer',
                position: 'relative',
                overflow: 'hidden'
              }}
              onClick={() => setSelectedDevice(device)}
            >
              {/* Device Header */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                  <div style={{
                    fontSize: '3rem',
                    filter: `drop-shadow(0 0 10px ${getStatusColor(device.status)})`
                  }}>
                    {getPlatformIcon(device.platform)}
                  </div>
                  <div>
                    <h3 style={{ margin: 0, color: '#00FFFF', fontSize: '1.3rem' }}>{device.name}</h3>
                    <div style={{ display: 'flex', gap: '10px', marginTop: '5px' }}>
                      <span style={{
                        padding: '3px 8px',
                        background: getStatusColor(device.status),
                        color: device.status === 'online' || device.status === 'syncing' ? '#000' : '#FFF',
                        borderRadius: '10px',
                        fontSize: '0.8rem',
                        fontWeight: 'bold',
                        textTransform: 'uppercase'
                      }}>
                        {device.status}
                      </span>
                      <span style={{
                        padding: '3px 8px',
                        background: 'rgba(255,255,0,0.2)',
                        color: '#FFFF00',
                        borderRadius: '10px',
                        fontSize: '0.8rem',
                        textTransform: 'capitalize'
                      }}>
                        {device.type}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  fontSize: '0.9rem'
                }}>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{
                      color: device.batteryLevel > 50 ? '#00FF00' : device.batteryLevel > 20 ? '#FFFF00' : '#FF0040'
                    }}>
                      🔋 {device.batteryLevel}%
                    </div>
                    <div style={{ color: '#00FFFF' }}>
                      {device.connection.toUpperCase()}
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Storage Usage */}
              <div style={{ marginBottom: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '0.9rem', color: '#FFFF00' }}>Storage</span>
                  <span style={{ fontSize: '0.9rem' }}>
                    {device.storageUsed} GB / {device.storageTotal} GB
                  </span>
                </div>
                <div style={{
                  background: 'rgba(0,0,0,0.5)',
                  borderRadius: '10px',
                  padding: '3px'
                }}>
                  <div style={{
                    width: `${(device.storageUsed / device.storageTotal) * 100}%`,
                    height: '15px',
                    background: 'linear-gradient(90deg, #00FF00, #FFFF00, #FF6600)',
                    borderRadius: '8px',
                    transition: 'all 0.3s ease'
                  }} />
                </div>
              </div>
              
              {/* Sync Settings Quick Toggle */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '10px',
                marginBottom: '15px'
              }}>
                {[
                  { key: 'autoSync', label: 'Auto Sync', value: device.syncSettings.autoSync },
                  { key: 'syncOnWifi', label: 'WiFi Only', value: device.syncSettings.syncOnWifi }
                ].map(setting => (
                  <button
                    key={setting.key}
                    onClick={(e) => {
                      e.stopPropagation();
                      updateDeviceSettings(device.id, { [setting.key]: !setting.value });
                    }}
                    style={{
                      padding: '8px 12px',
                      background: setting.value 
                        ? 'linear-gradient(45deg, #00FF00, #00FFFF)' 
                        : 'rgba(255,0,64,0.2)',
                      border: 'none',
                      borderRadius: '6px',
                      color: setting.value ? '#000' : '#FF0040',
                      fontWeight: 'bold',
                      cursor: 'pointer',
                      fontSize: '0.8rem'
                    }}
                  >
                    {setting.label}
                  </button>
                ))}
              </div>
              
              {/* Last Sync */}
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: '0.8rem',
                opacity: 0.7
              }}>
                <span>Last sync: {device.lastSync.toLocaleString()}</span>
                <span>{device.syncSettings.quality} quality</span>
              </div>
              
              {/* Active sync indicator */}
              {device.status === 'syncing' && (
                <div style={{
                  position: 'absolute',
                  top: '10px',
                  right: '10px',
                  width: '12px',
                  height: '12px',
                  background: '#FFFF00',
                  borderRadius: '50%',
                  animation: 'syncPulse 1s infinite'
                }} />
              )}
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Active Sync Tasks */}
      <motion.section
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
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
        <h2 style={{ color: '#FFFF00', marginBottom: '25px' }}>ACTIVE SYNC TASKS</h2>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          <AnimatePresence>
            {syncTasks.map((task, index) => (
              <motion.div
                key={task.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ delay: index * 0.05 }}
                style={{
                  background: 'rgba(0,0,0,0.9)',
                  border: `2px solid ${getTaskStatusColor(task.status)}`,
                  borderRadius: '12px',
                  padding: '20px',
                  position: 'relative',
                  overflow: 'hidden'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '15px' }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                      <span style={{
                        fontSize: '1.5rem'
                      }}>
                        {task.type === 'download' ? '⬇️' : 
                         task.type === 'upload' ? '⬆️' :
                         task.type === 'metadata' ? '📄' :
                         task.type === 'thumbnails' ? '🖼️' : '📝'}
                      </span>
                      <h3 style={{ margin: 0, color: '#00FFFF', fontSize: '1.1rem' }}>{task.title}</h3>
                      <span style={{
                        padding: '3px 8px',
                        background: getTaskStatusColor(task.status),
                        color: task.status === 'completed' || task.status === 'syncing' ? '#000' : '#FFF',
                        borderRadius: '4px',
                        fontSize: '0.7rem',
                        fontWeight: 'bold',
                        textTransform: 'uppercase'
                      }}>
                        {task.status}
                      </span>
                    </div>
                    
                    <div style={{ display: 'flex', gap: '20px', fontSize: '0.9rem', opacity: 0.8 }}>
                      <span>Device: {devices.find(d => d.id === task.device)?.name}</span>
                      <span>Size: {formatFileSize(task.fileSize * 1024 * 1024)}</span>
                      {task.speed > 0 && <span>Speed: {task.speed.toFixed(1)} MB/s</span>}
                      {task.eta > 0 && <span>ETA: {formatTime(task.eta)}</span>}
                    </div>
                  </div>
                  
                  <div style={{ display: 'flex', gap: '10px' }}>
                    {task.status === 'syncing' || task.status === 'paused' ? (
                      <button
                        onClick={() => pauseTask(task.id)}
                        style={{
                          padding: '6px 12px',
                          background: task.status === 'paused' ? 'linear-gradient(45deg, #00FF00, #00FFFF)' : 'rgba(255,255,0,0.2)',
                          border: '1px solid #FFFF00',
                          borderRadius: '6px',
                          color: task.status === 'paused' ? '#000' : '#FFFF00',
                          cursor: 'pointer',
                          fontSize: '0.8rem'
                        }}
                      >
                        {task.status === 'paused' ? '▶️' : '⏸️'}
                      </button>
                    ) : null}
                    
                    <button
                      onClick={() => cancelTask(task.id)}
                      style={{
                        padding: '6px 12px',
                        background: 'rgba(255,0,64,0.2)',
                        border: '1px solid #FF0040',
                        borderRadius: '6px',
                        color: '#FF0040',
                        cursor: 'pointer',
                        fontSize: '0.8rem'
                      }}
                    >
                      ✕
                    </button>
                  </div>
                </div>
                
                {/* Progress Bar */}
                <div style={{
                  background: 'rgba(0,0,0,0.5)',
                  borderRadius: '10px',
                  padding: '4px',
                  marginBottom: '10px'
                }}>
                  <div style={{
                    width: `${task.progress}%`,
                    height: '20px',
                    background: `linear-gradient(90deg, ${getTaskStatusColor(task.status)}, rgba(255,255,255,0.3))`,
                    borderRadius: '8px',
                    transition: 'all 0.3s ease',
                    position: 'relative',
                    overflow: 'hidden'
                  }}>
                    {task.status === 'syncing' && (
                      <div style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
                        animation: 'progressShimmer 2s linear infinite'
                      }} />
                    )}
                  </div>
                </div>
                
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: '0.8rem',
                  opacity: 0.7
                }}>
                  <span>{task.progress.toFixed(1)}% complete</span>
                  <span>{formatFileSize(task.syncedSize * 1024 * 1024)} / {formatFileSize(task.fileSize * 1024 * 1024)}</span>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </motion.section>

      {/* QR Code Scanner Modal */}
      <AnimatePresence>
        {isQRScannerOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(0,0,0,0.9)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1000
            }}
            onClick={() => setIsQRScannerOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              style={{
                background: 'rgba(0,0,0,0.95)',
                border: '2px solid #00FFFF',
                borderRadius: '15px',
                padding: '40px',
                textAlign: 'center',
                maxWidth: '500px'
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <h2 style={{ color: '#00FFFF', marginBottom: '25px' }}>CONNECT NEW DEVICE</h2>
              
              <div style={{ marginBottom: '25px' }}>
                <canvas
                  ref={qrRef}
                  width={200}
                  height={200}
                  style={{
                    border: '2px solid #FFFF00',
                    borderRadius: '8px'
                  }}
                />
              </div>
              
              <p style={{ marginBottom: '20px', opacity: 0.8 }}>
                Scan this QR code with your mobile device to connect and start syncing.
              </p>
              
              <div style={{ marginBottom: '20px', fontSize: '0.9rem', opacity: 0.7 }}>
                Connection URL: <code style={{ color: '#FFFF00' }}>nexus://192.168.1.100:8080/sync</code>
              </div>
              
              <button
                onClick={() => setIsQRScannerOpen(false)}
                style={{
                  padding: '12px 25px',
                  background: 'linear-gradient(45deg, #FF00FF, #FFFF00)',
                  border: 'none',
                  borderRadius: '8px',
                  color: '#000',
                  fontWeight: 'bold',
                  cursor: 'pointer'
                }}
              >
                CLOSE
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <style jsx>{`
        @keyframes mobilePulse {
          0% { filter: hue-rotate(0deg) brightness(1); }
          100% { filter: hue-rotate(120deg) brightness(1.3); }
        }
        
        @keyframes syncWave {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(200%); }
        }
        
        @keyframes syncPulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.6; transform: scale(1.5); }
        }
        
        @keyframes progressShimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        
        @keyframes mobileFloat {
          0% { transform: translateY(0); }
          50% { transform: translateY(-10px); }
          100% { transform: translateY(0); }
        }
      `}</style>
    </div>
  );
};

export default MobileSync;