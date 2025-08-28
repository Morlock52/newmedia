import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface PWAFeature {
  id: string;
  name: string;
  icon: string;
  status: 'active' | 'inactive' | 'updating' | 'error';
  description: string;
  action: () => void;
}

interface MobileGesture {
  type: 'swipe' | 'pinch' | 'tap' | 'longPress';
  direction?: 'up' | 'down' | 'left' | 'right';
  callback: () => void;
}

interface OfflineCapability {
  id: string;
  feature: string;
  cached: boolean;
  lastSync: Date;
  size: number;
}

const MobilePWAInterface: React.FC = () => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [installPrompt, setInstallPrompt] = useState<any>(null);
  const [isInstallable, setIsInstallable] = useState(false);
  const [orientation, setOrientation] = useState(screen.orientation?.angle || 0);
  const [viewportHeight, setViewportHeight] = useState(window.innerHeight);
  const [swipeGesture, setSwipeGesture] = useState<{ startX: number; startY: number } | null>(null);
  const [notificationPermission, setNotificationPermission] = useState(Notification.permission);
  const [cacheStatus, setCacheStatus] = useState<'checking' | 'idle' | 'updating' | 'updated'>('idle');
  
  const pwaFeatures: PWAFeature[] = [
    {
      id: 'offline',
      name: 'Offline Mode',
      icon: '📱',
      status: isOnline ? 'active' : 'inactive',
      description: 'Access cached content without internet',
      action: () => toggleOfflineMode()
    },
    {
      id: 'notifications',
      name: 'Push Notifications',
      icon: '🔔',
      status: notificationPermission === 'granted' ? 'active' : 'inactive',
      description: 'Receive real-time updates',
      action: () => requestNotificationPermission()
    },
    {
      id: 'background-sync',
      name: 'Background Sync',
      icon: '🔄',
      status: 'active',
      description: 'Sync data when connection returns',
      action: () => triggerBackgroundSync()
    },
    {
      id: 'install',
      name: 'Install App',
      icon: '📲',
      status: isInstallable ? 'active' : 'inactive',
      description: 'Add to home screen',
      action: () => installPWA()
    }
  ];

  const offlineCapabilities: OfflineCapability[] = [
    {
      id: 'media-library',
      feature: 'Media Library',
      cached: true,
      lastSync: new Date(),
      size: 15.6
    },
    {
      id: 'user-preferences',
      feature: 'User Settings',
      cached: true,
      lastSync: new Date(),
      size: 0.2
    },
    {
      id: 'download-queue',
      feature: 'Download Queue',
      cached: true,
      lastSync: new Date(),
      size: 2.1
    }
  ];

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    const handleBeforeInstallPrompt = (e: Event) => {
      e.preventDefault();
      setInstallPrompt(e);
      setIsInstallable(true);
    };
    const handleOrientationChange = () => setOrientation(screen.orientation?.angle || 0);
    const handleResize = () => setViewportHeight(window.innerHeight);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    screen.orientation?.addEventListener('change', handleOrientationChange);
    window.addEventListener('resize', handleResize);

    // Register service worker
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js')
        .then(registration => {
          console.log('SW registered:', registration);
          registration.addEventListener('updatefound', () => {
            setCacheStatus('updating');
          });
        })
        .catch(error => console.log('SW registration failed:', error));
    }

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      screen.orientation?.removeEventListener('change', handleOrientationChange);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  const requestNotificationPermission = async () => {
    if ('Notification' in window) {
      const permission = await Notification.requestPermission();
      setNotificationPermission(permission);
      
      if (permission === 'granted') {
        new Notification('🚀 NEXUS PWA Activated', {
          body: 'Push notifications are now enabled',
          icon: '/icon-192x192.png',
          badge: '/badge-72x72.png',
          tag: 'pwa-enabled'
        });
      }
    }
  };

  const installPWA = async () => {
    if (installPrompt) {
      installPrompt.prompt();
      const result = await installPrompt.userChoice;
      
      if (result.outcome === 'accepted') {
        setIsInstallable(false);
        setInstallPrompt(null);
      }
    }
  };

  const toggleOfflineMode = () => {
    // Toggle service worker cache strategies
    if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({ type: 'TOGGLE_OFFLINE_MODE' });
    }
  };

  const triggerBackgroundSync = () => {
    if ('serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype) {
      navigator.serviceWorker.ready.then(registration => {
        return (registration as any).sync.register('background-sync');
      });
    }
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    const touch = e.touches[0];
    setSwipeGesture({ startX: touch.clientX, startY: touch.clientY });
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    if (!swipeGesture) return;
    
    const touch = e.changedTouches[0];
    const deltaX = touch.clientX - swipeGesture.startX;
    const deltaY = touch.clientY - swipeGesture.startY;
    const minSwipeDistance = 50;
    
    if (Math.abs(deltaX) > minSwipeDistance || Math.abs(deltaY) > minSwipeDistance) {
      if (Math.abs(deltaX) > Math.abs(deltaY)) {
        // Horizontal swipe
        if (deltaX > 0) {
          console.log('Swipe right detected');
        } else {
          console.log('Swipe left detected');
        }
      } else {
        // Vertical swipe
        if (deltaY > 0) {
          console.log('Swipe down detected');
        } else {
          console.log('Swipe up detected');
        }
      }
    }
    
    setSwipeGesture(null);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return '#00FFFF';
      case 'inactive': return '#FF00FF';
      case 'updating': return '#FFFF00';
      case 'error': return '#FF0040';
      default: return '#666666';
    }
  };

  return (
    <div 
      className="mobile-pwa-interface"
      style={{
        height: `${viewportHeight}px`,
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%)',
        color: '#00FFFF',
        fontFamily: 'Orbitron, monospace',
        overflow: 'hidden',
        position: 'relative'
      }}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
    >
      {/* Cyberpunk Grid Background */}
      <div 
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundImage: 'linear-gradient(rgba(0,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,255,0.1) 1px, transparent 1px)',
          backgroundSize: '20px 20px',
          opacity: 0.3,
          animation: 'gridPulse 4s infinite'
        }}
      />

      {/* Header */}
      <motion.header 
        style={{
          padding: '20px',
          borderBottom: '2px solid #00FFFF',
          background: 'rgba(0,0,0,0.7)',
          backdropFilter: 'blur(10px)'
        }}
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ 
              margin: 0, 
              fontSize: '1.5rem',
              textShadow: '0 0 10px #00FFFF',
              animation: 'glitch 2s infinite'
            }}>
              NEXUS PWA
            </h1>
            <p style={{ margin: 0, opacity: 0.7, fontSize: '0.9rem' }}>
              Orientation: {orientation}° | {isOnline ? 'ONLINE' : 'OFFLINE'}
            </p>
          </div>
          
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            background: isOnline ? 
              'radial-gradient(circle, #00FF00 0%, #00FFFF 100%)' : 
              'radial-gradient(circle, #FF0040 0%, #FF00FF 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '1.2rem',
            animation: 'pulse 2s infinite'
          }}>
            {isOnline ? '🌐' : '📱'}
          </div>
        </div>
      </motion.header>

      {/* PWA Features Grid */}
      <div style={{ padding: '20px', flex: 1, overflowY: 'auto' }}>
        <motion.section
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <h2 style={{ 
            marginBottom: '20px', 
            color: '#FFFF00',
            textShadow: '0 0 8px #FFFF00'
          }}>
            PWA CAPABILITIES
          </h2>
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '15px',
            marginBottom: '30px'
          }}>
            {pwaFeatures.map((feature, index) => (
              <motion.div
                key={feature.id}
                style={{
                  background: 'rgba(0,0,0,0.6)',
                  border: `2px solid ${getStatusColor(feature.status)}`,
                  borderRadius: '12px',
                  padding: '20px',
                  cursor: 'pointer',
                  backdropFilter: 'blur(5px)',
                  boxShadow: `0 0 20px rgba(${feature.status === 'active' ? '0,255,255' : '255,0,255'},0.3)`
                }}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: index * 0.1 }}
                whileTap={{ scale: 0.95 }}
                onClick={feature.action}
              >
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                  <span style={{ fontSize: '2rem', marginRight: '12px' }}>{feature.icon}</span>
                  <div>
                    <h3 style={{ margin: 0, color: getStatusColor(feature.status) }}>{feature.name}</h3>
                    <span style={{ 
                      fontSize: '0.8rem', 
                      color: feature.status === 'active' ? '#00FF00' : '#FF6B6B',
                      textTransform: 'uppercase',
                      fontWeight: 'bold'
                    }}>
                      {feature.status}
                    </span>
                  </div>
                </div>
                <p style={{ margin: 0, opacity: 0.8, fontSize: '0.9rem' }}>
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Offline Capabilities */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h2 style={{ 
            marginBottom: '20px', 
            color: '#FF00FF',
            textShadow: '0 0 8px #FF00FF'
          }}>
            CACHED DATA
          </h2>
          
          <div style={{
            background: 'rgba(0,0,0,0.6)',
            border: '2px solid #FF00FF',
            borderRadius: '12px',
            padding: '20px',
            backdropFilter: 'blur(5px)'
          }}>
            {offlineCapabilities.map((capability, index) => (
              <motion.div
                key={capability.id}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '12px 0',
                  borderBottom: index < offlineCapabilities.length - 1 ? '1px solid rgba(255,0,255,0.3)' : 'none'
                }}
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 0.5 + index * 0.1 }}
              >
                <div>
                  <span style={{ color: '#FFFF00', fontWeight: 'bold' }}>{capability.feature}</span>
                  <div style={{ fontSize: '0.8rem', opacity: 0.7 }}>
                    Last sync: {capability.lastSync.toLocaleTimeString()}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{
                    color: capability.cached ? '#00FF00' : '#FF6B6B',
                    fontWeight: 'bold'
                  }}>
                    {capability.cached ? '✓ CACHED' : '✗ NOT CACHED'}
                  </div>
                  <div style={{ fontSize: '0.8rem', opacity: 0.7 }}>
                    {capability.size} MB
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Install Prompt */}
        <AnimatePresence>
          {isInstallable && (
            <motion.div
              style={{
                position: 'fixed',
                bottom: '20px',
                left: '20px',
                right: '20px',
                background: 'linear-gradient(45deg, #FF00FF, #FFFF00)',
                color: '#000',
                padding: '20px',
                borderRadius: '12px',
                textAlign: 'center',
                fontWeight: 'bold',
                cursor: 'pointer',
                zIndex: 1000,
                boxShadow: '0 10px 30px rgba(255,0,255,0.5)'
              }}
              initial={{ y: 100, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: 100, opacity: 0 }}
              onClick={installPWA}
              whileTap={{ scale: 0.95 }}
            >
              <div style={{ fontSize: '1.5rem', marginBottom: '8px' }}>📲</div>
              <div>TAP TO INSTALL NEXUS PWA</div>
              <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>Add to your home screen for the best experience</div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <style jsx>{`
        @keyframes glitch {
          0%, 100% { text-shadow: 0 0 10px #00FFFF; }
          25% { text-shadow: -2px 0 10px #FF00FF, 2px 0 10px #FFFF00; }
          50% { text-shadow: 2px 0 10px #FF00FF, -2px 0 10px #FFFF00; }
          75% { text-shadow: 0 0 10px #00FFFF, 0 2px 10px #FF00FF; }
        }
        
        @keyframes gridPulse {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 0.6; }
        }
        
        @keyframes pulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.1); }
        }
      `}</style>
    </div>
  );
};

export default MobilePWAInterface;