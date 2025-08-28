'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CyberpunkLayout } from './CyberpunkLayout'
import dynamic from 'next/dynamic'
import styles from './CyberpunkDashboard.module.css'
import { HolographicCard } from './HolographicCard'
import { NeonButton } from './NeonButton'
import { Terminal, Activity, Database, Shield, Cpu, Wifi, Cloud, Server, RefreshCw } from 'lucide-react'

const ClientServiceOrb = dynamic(
  () => import('./ClientServiceOrb'),
  { ssr: false }
)

const SERVICES = [
  { name: 'JELLYFIN', port: 8096, icon: '🎬', color: '#00ffff' },
  { name: 'PROWLARR', port: 9696, icon: '🔍', color: '#ff00ff' },
  { name: 'SONARR', port: 8989, icon: '📺', color: '#9d00ff' },
  { name: 'RADARR', port: 7878, icon: '🎥', color: '#00ff88' },
  { name: 'LIDARR', port: 8686, icon: '🎵', color: '#ffff00' },
  { name: 'QBITTORRENT', port: 8080, icon: '⬇️', color: '#ff6600' },
]

interface ServiceStatus {
  name: string
  status: 'online' | 'offline' | 'checking' | 'error'
  responseTime?: number
}

export function CyberpunkDashboard() {
  const [serviceStatuses, setServiceStatuses] = useState<ServiceStatus[]>(
    SERVICES.map(s => ({ name: s.name, status: 'checking' }))
  )
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [systemMetrics, setSystemMetrics] = useState(() => ({
    cpu: 45,
    memory: 67,
    disk: 82,
    network: 23
  }))
  const [terminalLines, setTerminalLines] = useState<string[]>([
    '> SYSTEM INITIALIZED',
    '> NEURAL NETWORK ONLINE',
    '> SECURITY PROTOCOLS ACTIVE',
    '> AWAITING COMMANDS...'
  ])

  const checkServices = async () => {
    setIsRefreshing(true)
    addTerminalLine('> SCANNING SERVICES...')
    
    const updatedStatuses = await Promise.all(
      SERVICES.map(async (service) => {
        try {
          const startTime = Date.now()
          const response = await fetch(`http://localhost:${service.port}`, {
            method: 'HEAD',
            mode: 'no-cors',
            signal: AbortSignal.timeout(3000)
          })
          const responseTime = Date.now() - startTime
          
          addTerminalLine(`> ${service.name}: ONLINE [${responseTime}ms]`)
          return {
            name: service.name,
            status: 'online' as const,
            responseTime
          }
        } catch (error) {
          addTerminalLine(`> ${service.name}: OFFLINE`)
          return {
            name: service.name,
            status: 'offline' as const
          }
        }
      })
    )
    
    setServiceStatuses(updatedStatuses)
    setIsRefreshing(false)
    addTerminalLine('> SCAN COMPLETE')
  }

  const addTerminalLine = (line: string) => {
    setTerminalLines(prev => [...prev.slice(-9), line])
  }

  useEffect(() => {
    // Simulate system metrics updates
    const metricsInterval = setInterval(() => {
      setSystemMetrics({
        cpu: Math.floor(Math.random() * 30 + 40),
        memory: Math.floor(Math.random() * 20 + 60),
        disk: Math.floor(Math.random() * 10 + 80),
        network: Math.floor(Math.random() * 50 + 10)
      })
    }, 3000)

    return () => clearInterval(metricsInterval)
  }, [])

  return (
    <CyberpunkLayout>
      <div className={styles.dashboardContainer}>
      <header className={styles.header}>
        <h1 className={styles.title}>MEDIA NEXUS CONTROL</h1>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className={styles.refreshButton}
          onClick={checkServices}
          disabled={isRefreshing}
        >
          <RefreshCw className={isRefreshing ? 'animate-spin' : ''} size={18} />
          {isRefreshing ? 'SCANNING...' : 'INITIATE SCAN'}
        </motion.button>
      </header>
      <div className="min-h-screen p-6">
        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
          {/* Services Section */}
          <motion.div
            className="lg:col-span-2"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <div className={styles.holographicCard}>
              <h2 className={styles.cardTitle}>SERVICE MATRIX</h2>
              <p className={styles.cardSubtitle}>Neural Network Status</p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                {serviceStatuses.map((service, index) => (
                  <motion.div
                    key={service.name}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    <ClientServiceOrb
                      name={service.name}
                      status={service.status}
                      responseTime={service.responseTime}
                      onClick={() => {
                        const serviceConfig = SERVICES.find(s => s.name === service.name)
                        if (serviceConfig) {
                          window.open(`http://localhost:${serviceConfig.port}`, '_blank')
                        }
                      }}
                    />
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* System Metrics */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <div className={styles.holographicCard}>
              <h2 className={styles.cardTitle}>SYSTEM CORE</h2>
              <p className={styles.cardSubtitle}>Performance Metrics</p>
              <div className="space-y-4">
                {Object.entries(systemMetrics).map(([key, value]) => (
                  <div key={key}>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-xs uppercase">
                        {key === 'cpu' && <Cpu className="inline w-3 h-3 mr-1" />}
                        {key === 'memory' && <Database className="inline w-3 h-3 mr-1" />}
                        {key === 'disk' && <Server className="inline w-3 h-3 mr-1" />}
                        {key === 'network' && <Wifi className="inline w-3 h-3 mr-1" />}
                        {key}
                      </span>
                      <span className="text-xs font-mono">
                        {value}%
                      </span>
                    </div>
                    <div className="relative h-2 bg-black/50 rounded-full overflow-hidden">
                      <motion.div
                        className="absolute inset-y-0 left-0 rounded-full"
                        style={{
                          background: `linear-gradient(90deg, #00ffff, #ff00ff)`,
                          boxShadow: `0 0 10px ${value > 80 ? '#ff0040' : '#00ffff'}`
                        }}
                        initial={{ width: 0 }}
                        animate={{ width: `${value}%` }}
                        transition={{ duration: 1, ease: 'easeOut' }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6 pt-6 border-t border-cyan-500/20">
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase">
                    <Shield className="inline w-3 h-3 mr-1" />
                    Security
                  </span>
                  <motion.span
                    className="text-xs font-mono"
                    animate={{ opacity: [1, 0.5, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    ACTIVE
                  </motion.span>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Terminal */}
          <motion.div
            className="lg:col-span-3"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <div className={styles.holographicCard}>
              <h2 className={styles.cardTitle}>NEURAL TERMINAL</h2>
              <p className={styles.cardSubtitle}>System Console</p>
              <div className={styles.terminal}>
                <div className="flex items-center gap-2 mb-2 pb-2 border-b border-green-500/20">
                  <Terminal className="w-4 h-4" />
                  <span>root@nexus</span>
                  <span className="opacity-50">~/media-server</span>
                </div>
                <div className="space-y-1 h-40 overflow-y-auto">
                  <AnimatePresence mode="popLayout">
                    {terminalLines.map((line, index) => (
                      <motion.div
                        key={`${line}-${index}`}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 10 }}
                        transition={{ duration: 0.2 }}
                      >
                        {line}
                      </motion.div>
                    ))}
                  </AnimatePresence>
                  <motion.span
                    className="inline-block w-2 h-3 bg-green-500"
                    animate={{ opacity: [1, 0, 1] }}
                    transition={{ duration: 1, repeat: Infinity }}
                  />
                </div>
              </div>

              <div className="mt-4 flex gap-2">
                <NeonButton
                  size="sm"
                  color="green"
                  onClick={() => addTerminalLine('> EXECUTING DIAGNOSTIC...')}
                >
                  DIAGNOSTIC
                </NeonButton>
                <NeonButton
                  size="sm"
                  color="purple"
                  onClick={() => addTerminalLine('> OPTIMIZING NEURAL PATHWAYS...')}
                >
                  OPTIMIZE
                </NeonButton>
                <NeonButton
                  size="sm"
                  color="yellow"
                  onClick={() => setTerminalLines(['> TERMINAL CLEARED', '> READY...'])}
                >
                  CLEAR
                </NeonButton>
              </div>
            </div>
          </motion.div>

          {/* Quick Actions */}
          <motion.div
            className="lg:col-span-3"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <HolographicCard title="QUANTUM CONTROLS" subtitle="System Operations" glowColor="#ffff00">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <NeonButton color="cyan" fullWidth>
                  <Cloud className="w-4 h-4" />
                  SYNC
                </NeonButton>
                <NeonButton color="pink" fullWidth>
                  <Activity className="w-4 h-4" />
                  MONITOR
                </NeonButton>
                <NeonButton color="green" fullWidth>
                  <Shield className="w-4 h-4" />
                  SECURE
                </NeonButton>
                <NeonButton color="purple" fullWidth>
                  <Server className="w-4 h-4" />
                  BACKUP
                </NeonButton>
              </div>
            </HolographicCard>
          </motion.div>
        </div>
      </div>
      </div>
    </CyberpunkLayout>
  )
}