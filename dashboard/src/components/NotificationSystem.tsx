import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { io, Socket } from 'socket.io-client';
import './NotificationSystem.css';

interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info' | 'download' | 'media';
  title: string;
  message: string;
  timestamp: Date;
  service?: string;
  icon?: string;
  progress?: number;
  actions?: Array<{
    label: string;
    action: () => void;
  }>;
}

const NotificationSystem: React.FC = () => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [socket, setSocket] = useState<Socket | null>(null);

  useEffect(() => {
    // Connect to WebSocket server
    const ws = io('ws://localhost:8181', {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    ws.on('connect', () => {
      console.log('Connected to notification server');
    });

    // Listen for different notification types
    ws.on('media:new', (data) => {
      addNotification({
        type: 'media',
        title: 'New Media Available',
        message: data.message,
        service: data.service,
        icon: '🎬',
      });
    });

    ws.on('download:complete', (data) => {
      addNotification({
        type: 'download',
        title: 'Download Complete',
        message: `${data.name} has finished downloading`,
        service: data.service,
        icon: '✅',
        progress: 100,
      });
    });

    ws.on('service:failure', (data) => {
      addNotification({
        type: 'error',
        title: 'Service Failure',
        message: `${data.service} is not responding`,
        service: data.service,
        icon: '⚠️',
        actions: [
          {
            label: 'Restart',
            action: () => restartService(data.service),
          },
        ],
      });
    });

    ws.on('system:alert', (data) => {
      addNotification({
        type: 'warning',
        title: 'System Alert',
        message: data.message,
        icon: '🚨',
      });
    });

    setSocket(ws);

    return () => {
      ws.disconnect();
    };
  }, []);

  const addNotification = useCallback((notification: Omit<Notification, 'id' | 'timestamp'>) => {
    const newNotification: Notification = {
      ...notification,
      id: `notif-${Date.now()}-${Math.random()}`,
      timestamp: new Date(),
    };

    setNotifications((prev) => [newNotification, ...prev].slice(0, 5));

    // Auto-remove after 10 seconds
    setTimeout(() => {
      removeNotification(newNotification.id);
    }, 10000);
  }, []);

  const removeNotification = (id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  };

  const restartService = async (serviceName: string) => {
    try {
      const response = await fetch(`/api/services/${serviceName}/restart`, {
        method: 'POST',
      });
      if (response.ok) {
        addNotification({
          type: 'success',
          title: 'Service Restarted',
          message: `${serviceName} has been restarted successfully`,
          icon: '🔄',
        });
      }
    } catch (error) {
      console.error('Failed to restart service:', error);
    }
  };

  return (
    <div className="notification-container">
      <AnimatePresence>
        {notifications.map((notification) => (
          <motion.div
            key={notification.id}
            className={`notification notification-${notification.type} cyberpunk-notification`}
            initial={{ x: 400, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 400, opacity: 0 }}
            transition={{
              type: 'spring',
              stiffness: 500,
              damping: 30,
            }}
          >
            {/* Holographic effect layers */}
            <div className="holographic-bg"></div>
            <div className="glitch-effect" data-text={notification.title}></div>
            
            {/* Notification content */}
            <div className="notification-content">
              <div className="notification-header">
                <span className="notification-icon neon-glow">
                  {notification.icon}
                </span>
                <div className="notification-text">
                  <h4 className="notification-title">{notification.title}</h4>
                  {notification.service && (
                    <span className="notification-service">{notification.service}</span>
                  )}
                </div>
                <button
                  className="notification-close"
                  onClick={() => removeNotification(notification.id)}
                >
                  ×
                </button>
              </div>
              
              <p className="notification-message">{notification.message}</p>
              
              {notification.progress !== undefined && (
                <div className="progress-bar">
                  <div
                    className="progress-fill neon-gradient"
                    style={{ width: `${notification.progress}%` }}
                  />
                </div>
              )}
              
              {notification.actions && notification.actions.length > 0 && (
                <div className="notification-actions">
                  {notification.actions.map((action, index) => (
                    <button
                      key={index}
                      className="notification-action cyberpunk-button"
                      onClick={action.action}
                    >
                      {action.label}
                    </button>
                  ))}
                </div>
              )}
              
              <div className="notification-timestamp">
                {notification.timestamp.toLocaleTimeString()}
              </div>
            </div>
            
            {/* Cyberpunk decorative elements */}
            <div className="cyber-corner top-left"></div>
            <div className="cyber-corner top-right"></div>
            <div className="cyber-corner bottom-left"></div>
            <div className="cyber-corner bottom-right"></div>
            <div className="scan-line"></div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
};

export default NotificationSystem;