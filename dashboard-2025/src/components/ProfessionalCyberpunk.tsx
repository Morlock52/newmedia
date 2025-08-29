'use client'

import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const SERVICES = [
  { id: 'jellyfin', name: 'Jellyfin', port: 8096, icon: '🎬', status: 'online', latency: 12 },
  { id: 'sonarr', name: 'Sonarr', port: 8989, icon: '📺', status: 'online', latency: 8 },
  { id: 'radarr', name: 'Radarr', port: 7878, icon: '🎥', status: 'online', latency: 15 },
  { id: 'prowlarr', name: 'Prowlarr', port: 9696, icon: '🔍', status: 'offline', latency: 0 },
  { id: 'lidarr', name: 'Lidarr', port: 8686, icon: '🎵', status: 'online', latency: 22 },
  { id: 'qbittorrent', name: 'qBittorrent', port: 8080, icon: '⬇️', status: 'online', latency: 5 },
]

export default function ProfessionalCyberpunk() {
  const [services, setServices] = useState(SERVICES)
  const [selectedService, setSelectedService] = useState<string | null>(null)
  const [metrics, setMetrics] = useState({
    cpu: { value: 45, trend: 'up' },
    memory: { value: 67, trend: 'stable' },
    network: { value: 234, trend: 'up' },
    disk: { value: 82, trend: 'down' }
  })
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    canvas.width = window.innerWidth
    canvas.height = window.innerHeight

    const particles: Array<{x: number, y: number, vx: number, vy: number, size: number}> = []
    
    for (let i = 0; i < 50; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        size: Math.random() * 2
      })
    }

    const animate = () => {
      ctx.fillStyle = 'rgba(10, 10, 20, 0.05)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      
      particles.forEach(p => {
        p.x += p.vx
        p.y += p.vy
        
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1
        
        ctx.beginPath()
        const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size)
        gradient.addColorStop(0, 'rgba(0, 255, 255, 0.8)')
        gradient.addColorStop(1, 'rgba(0, 255, 255, 0)')
        ctx.fillStyle = gradient
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
        ctx.fill()
      })
      
      requestAnimationFrame(animate)
    }
    
    animate()

    const handleResize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => ({
        cpu: { 
          value: Math.max(20, Math.min(90, prev.cpu.value + (Math.random() - 0.5) * 10)),
          trend: Math.random() > 0.5 ? 'up' : 'down'
        },
        memory: { 
          value: Math.max(40, Math.min(85, prev.memory.value + (Math.random() - 0.5) * 5)),
          trend: Math.random() > 0.5 ? 'up' : 'stable'
        },
        network: { 
          value: Math.max(50, Math.min(500, prev.network.value + (Math.random() - 0.5) * 50)),
          trend: Math.random() > 0.5 ? 'up' : 'down'
        },
        disk: { 
          value: Math.max(60, Math.min(95, prev.disk.value + (Math.random() - 0.5) * 2)),
          trend: 'stable'
        }
      }))
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-purple-950 to-gray-950 text-white overflow-hidden relative">
      <canvas 
        ref={canvasRef}
        className="fixed inset-0 pointer-events-none opacity-50"
        style={{ mixBlendMode: 'screen' }}
      />
      
      {/* Grid Background */}
      <div className="fixed inset-0 pointer-events-none opacity-10">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(cyan 1px, transparent 1px),
            linear-gradient(90deg, cyan 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px'
        }} />
      </div>

      {/* Header */}
      <motion.header 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="relative z-10 p-8 backdrop-blur-xl bg-black/20 border-b border-cyan-500/30"
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
              NEXUS CONTROL
            </h1>
            <p className="text-cyan-400/60 mt-2 font-mono text-sm">
              SYSTEM STATUS: OPERATIONAL | SECURITY: MAXIMUM | TIME: {new Date().toLocaleTimeString()}
            </p>
          </div>
          <div className="flex gap-4">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-lg font-bold shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/50 transition-shadow"
            >
              SCAN NETWORK
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg font-bold shadow-lg shadow-purple-500/25 hover:shadow-purple-500/50 transition-shadow"
            >
              OPTIMIZE
            </motion.button>
          </div>
        </div>
      </motion.header>

      <div className="relative z-10 p-8 max-w-7xl mx-auto">
        {/* Metrics Dashboard */}
        <div className="grid grid-cols-4 gap-6 mb-8">
          {Object.entries(metrics).map(([key, data], index) => (
            <motion.div
              key={key}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-gradient-to-br from-gray-900/90 to-gray-800/90 backdrop-blur-xl rounded-xl p-6 border border-cyan-500/20 hover:border-cyan-500/50 transition-all"
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-gray-400 uppercase text-xs font-bold tracking-wider">{key}</span>
                <span className={`text-xs ${data.trend === 'up' ? 'text-green-400' : data.trend === 'down' ? 'text-red-400' : 'text-yellow-400'}`}>
                  {data.trend === 'up' ? '↑' : data.trend === 'down' ? '↓' : '→'}
                </span>
              </div>
              <div className="text-3xl font-bold text-cyan-400">
                {key === 'network' ? `${data.value} MB/s` : `${data.value}%`}
              </div>
              <div className="mt-3 h-2 bg-gray-800 rounded-full overflow-hidden">
                <motion.div 
                  className="h-full bg-gradient-to-r from-cyan-500 to-purple-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${key === 'network' ? data.value / 5 : data.value}%` }}
                  transition={{ duration: 1, ease: "easeOut" }}
                />
              </div>
            </motion.div>
          ))}
        </div>

        {/* Services Grid */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="grid grid-cols-3 gap-6 mb-8"
        >
          {services.map((service, index) => (
            <motion.div
              key={service.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6 + index * 0.1 }}
              whileHover={{ scale: 1.02, y: -5 }}
              onClick={() => setSelectedService(service.id)}
              className={`
                relative bg-gradient-to-br from-gray-900/90 to-gray-800/90 backdrop-blur-xl rounded-xl p-6 
                border ${service.status === 'online' ? 'border-green-500/30' : 'border-red-500/30'}
                hover:border-cyan-500/50 transition-all cursor-pointer group
                ${selectedService === service.id ? 'ring-2 ring-cyan-500 shadow-lg shadow-cyan-500/25' : ''}
              `}
            >
              {/* Glow Effect */}
              <div className={`
                absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity
                ${service.status === 'online' ? 'bg-green-500/10' : 'bg-red-500/10'}
              `} />
              
              <div className="relative z-10">
                <div className="flex items-start justify-between mb-4">
                  <div className="text-4xl">{service.icon}</div>
                  <div className={`
                    px-2 py-1 rounded-full text-xs font-bold
                    ${service.status === 'online' 
                      ? 'bg-green-500/20 text-green-400 shadow-lg shadow-green-500/25' 
                      : 'bg-red-500/20 text-red-400 shadow-lg shadow-red-500/25'}
                  `}>
                    {service.status.toUpperCase()}
                  </div>
                </div>
                
                <h3 className="text-xl font-bold text-white mb-2">{service.name}</h3>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between text-gray-400">
                    <span>Port</span>
                    <span className="font-mono text-cyan-400">{service.port}</span>
                  </div>
                  {service.status === 'online' && (
                    <div className="flex justify-between text-gray-400">
                      <span>Latency</span>
                      <span className="font-mono text-green-400">{service.latency}ms</span>
                    </div>
                  )}
                </div>

                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="mt-4 w-full py-2 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-lg 
                           border border-cyan-500/30 hover:border-cyan-500/50 transition-all text-cyan-400 font-bold"
                  onClick={(e) => {
                    e.stopPropagation()
                    if (service.status === 'online') {
                      window.open(`http://localhost:${service.port}`, '_blank')
                    }
                  }}
                >
                  {service.status === 'online' ? 'ACCESS' : 'UNAVAILABLE'}
                </motion.button>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Terminal */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1 }}
          className="bg-black/80 backdrop-blur-xl rounded-xl border border-green-500/30 p-6"
        >
          <div className="flex items-center gap-2 mb-4">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span className="ml-4 text-green-400 font-mono text-sm">NEXUS://TERMINAL/MAIN</span>
          </div>
          
          <div className="font-mono text-sm space-y-1 text-green-400">
            <div>[SYSTEM] Neural network initialized</div>
            <div>[SECURITY] Firewall status: ACTIVE</div>
            <div>[NETWORK] All services monitored</div>
            <div>[AI] Predictive analytics: ONLINE</div>
            <div className="flex items-center">
              <span className="text-cyan-400">nexus@control:~$</span>
              <span className="ml-2 animate-pulse">_</span>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}