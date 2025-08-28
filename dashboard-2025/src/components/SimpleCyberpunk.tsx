'use client'

import React, { useState, useEffect } from 'react'
import '@/app/simple-cyberpunk.css'

const SERVICES = [
  { name: 'JELLYFIN', port: 8096, icon: '🎬', color: '#00ffff' },
  { name: 'PROWLARR', port: 9696, icon: '🔍', color: '#ff00ff' },
  { name: 'SONARR', port: 8989, icon: '📺', color: '#9d00ff' },
  { name: 'RADARR', port: 7878, icon: '🎥', color: '#00ff88' },
  { name: 'LIDARR', port: 8686, icon: '🎵', color: '#ffff00' },
  { name: 'QBITTORRENT', port: 8080, icon: '⬇️', color: '#ff6600' },
]

export function SimpleCyberpunkDashboard() {
  const [services, setServices] = useState(SERVICES.map(s => ({ ...s, status: 'checking' })))
  const [terminalLines, setTerminalLines] = useState(['> SYSTEM ONLINE', '> AWAITING INPUT...'])
  const [metrics, setMetrics] = useState({ cpu: 45, memory: 67, disk: 82, network: 23 })

  useEffect(() => {
    // Simulate service status checks
    const timer = setTimeout(() => {
      setServices(SERVICES.map(s => ({ 
        ...s, 
        status: Math.random() > 0.3 ? 'online' : 'offline' 
      })))
      addTerminalLine('> ALL SERVICES CHECKED')
    }, 2000)

    // Simulate metrics updates
    const metricsInterval = setInterval(() => {
      setMetrics({
        cpu: Math.floor(Math.random() * 30 + 40),
        memory: Math.floor(Math.random() * 20 + 60),
        disk: Math.floor(Math.random() * 10 + 80),
        network: Math.floor(Math.random() * 50 + 10)
      })
    }, 3000)

    return () => {
      clearTimeout(timer)
      clearInterval(metricsInterval)
    }
  }, [])

  const addTerminalLine = (line: string) => {
    setTerminalLines(prev => [...prev.slice(-9), line])
  }

  const checkServices = () => {
    addTerminalLine('> SCANNING SERVICES...')
    setServices(SERVICES.map(s => ({ ...s, status: 'checking' })))
    
    setTimeout(() => {
      const updated = SERVICES.map(s => ({ 
        ...s, 
        status: Math.random() > 0.3 ? 'online' : 'offline' 
      }))
      setServices(updated)
      
      updated.forEach(s => {
        addTerminalLine(`> ${s.name}: ${s.status.toUpperCase()}`)
      })
      addTerminalLine('> SCAN COMPLETE')
    }, 1500)
  }

  return (
    <div className="cyber-container">
      {/* Animated Title */}
      <h1 className="cyber-title">
        🌆 MEDIA NEXUS CONTROL 🌆
      </h1>

      {/* Quick Actions */}
      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <button className="cyber-button" onClick={checkServices}>
          🔄 SCAN SERVICES
        </button>
        <button className="cyber-button" style={{ marginLeft: '1rem' }} onClick={() => addTerminalLine('> OPTIMIZING...')}>
          ⚡ OPTIMIZE
        </button>
        <button className="cyber-button" style={{ marginLeft: '1rem' }} onClick={() => setTerminalLines(['> TERMINAL CLEARED'])}>
          🗑️ CLEAR
        </button>
      </div>

      {/* Services Grid */}
      <div className="services-grid">
        {services.map((service, index) => (
          <div 
            key={service.name}
            className="service-card"
            style={{ 
              animationDelay: `${index * 0.1}s`,
              borderColor: service.status === 'online' ? '#00ff88' : 
                          service.status === 'offline' ? '#ff0040' : '#ffff00'
            }}
            onClick={() => {
              if (service.status === 'online') {
                window.open(`http://localhost:${service.port}`, '_blank')
              }
            }}
          >
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>{service.icon}</div>
            <div className="service-name">{service.name}</div>
            <div className={`status-${service.status}`} style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
              {service.status === 'checking' && <span className="cyber-loading" />}
              {service.status !== 'checking' && (
                <>
                  <span style={{ fontSize: '0.75rem' }}>●</span> {service.status.toUpperCase()}
                </>
              )}
            </div>
            <div style={{ fontSize: '0.75rem', opacity: 0.7, marginTop: '0.25rem' }}>
              PORT: {service.port}
            </div>
          </div>
        ))}
      </div>

      {/* System Metrics */}
      <div className="service-card" style={{ maxWidth: '600px', margin: '2rem auto' }}>
        <h2 style={{ 
          fontSize: '1.5rem', 
          marginBottom: '1rem', 
          color: '#ff00ff',
          textShadow: '0 0 10px currentColor' 
        }}>
          SYSTEM METRICS
        </h2>
        {Object.entries(metrics).map(([key, value]) => (
          <div key={key} style={{ marginBottom: '1rem' }}>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              marginBottom: '0.25rem',
              fontSize: '0.875rem',
              textTransform: 'uppercase'
            }}>
              <span>{key}</span>
              <span style={{ color: value > 80 ? '#ff0040' : '#00ff88' }}>{value}%</span>
            </div>
            <div className="metric-bar">
              <div 
                className="metric-fill" 
                style={{ 
                  width: `${value}%`,
                  background: value > 80 ? 'linear-gradient(90deg, #ff0040, #ff6600)' : 
                             value > 60 ? 'linear-gradient(90deg, #ffff00, #ff00ff)' :
                             'linear-gradient(90deg, #00ff88, #00ffff)'
                }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Terminal */}
      <div className="cyber-terminal">
        <div style={{ 
          borderBottom: '1px solid #00ff88', 
          paddingBottom: '0.5rem', 
          marginBottom: '0.5rem',
          fontSize: '0.875rem'
        }}>
          <span style={{ opacity: 0.7 }}>root@nexus:~$</span>
        </div>
        <div style={{ minHeight: '150px' }}>
          {terminalLines.map((line, index) => (
            <div key={index} style={{ marginBottom: '0.25rem', fontSize: '0.875rem' }}>
              {line}
            </div>
          ))}
          <span style={{ 
            display: 'inline-block',
            width: '10px',
            height: '15px',
            background: '#00ff88',
            animation: 'pulse 1s infinite'
          }} />
        </div>
      </div>

      {/* Footer */}
      <div style={{ 
        textAlign: 'center', 
        marginTop: '3rem', 
        opacity: 0.7,
        fontSize: '0.875rem'
      }}>
        <div style={{ marginBottom: '0.5rem' }}>
          🔒 SECURE CONNECTION | 🌐 NEURAL NETWORK ACTIVE | ⚡ QUANTUM ENCRYPTED
        </div>
        <div style={{ fontSize: '0.75rem' }}>
          CYBERPUNK MEDIA DASHBOARD v2025.1.0 | BUILT WITH REACT & THREE.JS
        </div>
      </div>
    </div>
  )
}