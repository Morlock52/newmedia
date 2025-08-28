'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Activity, Cpu, HardDrive, MemoryStick, Network, Zap } from 'lucide-react'

interface Service {
  name: string
  port: number
  type: string
  color: string
  status: string
}

interface SystemMetricsProps {
  services: Service[]
}

export function SystemMetrics({ services }: SystemMetricsProps) {
  const [metrics, setMetrics] = useState({
    cpu: 0,
    memory: 0,
    disk: 0,
    network: 0,
    uptime: '0d 0h 0m',
    activeServices: 0
  })

  useEffect(() => {
    // Simulate real-time metrics updates
    const interval = setInterval(() => {
      setMetrics({
        cpu: Math.random() * 100,
        memory: 65 + Math.random() * 20,
        disk: 45 + Math.random() * 10,
        network: Math.random() * 1000,
        uptime: '5d 12h 34m',
        activeServices: services.filter(s => s.status === 'online').length
      })
    }, 2000)

    return () => clearInterval(interval)
  }, [services])

  const MetricCard = ({ 
    icon: Icon, 
    label, 
    value, 
    unit, 
    color,
    progress 
  }: {
    icon: any
    label: string
    value: string | number
    unit?: string
    color: string
    progress?: number
  }) => (
    <motion.div
      whileHover={{ scale: 1.02, y: -2 }}
      className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-white/10"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <Icon className={`w-5 h-5 ${color}`} />
          <span className="text-sm text-gray-300">{label}</span>
        </div>
        <span className="text-lg font-bold text-white">
          {typeof value === 'number' ? value.toFixed(1) : value}
          {unit && <span className="text-sm text-gray-400 ml-1">{unit}</span>}
        </span>
      </div>
      
      {progress !== undefined && (
        <div className="w-full bg-gray-700 rounded-full h-2">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5 }}
            className={`h-2 rounded-full bg-gradient-to-r ${
              progress > 80 ? 'from-red-500 to-red-600' :
              progress > 60 ? 'from-yellow-500 to-orange-500' :
              'from-green-500 to-emerald-500'
            }`}
          />
        </div>
      )}
    </motion.div>
  )

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-2">System Metrics</h2>
        <p className="text-gray-400">Real-time server performance</p>
      </div>

      <div className="space-y-4">
        <MetricCard
          icon={Cpu}
          label="CPU Usage"
          value={metrics.cpu}
          unit="%"
          color="text-blue-400"
          progress={metrics.cpu}
        />

        <MetricCard
          icon={MemoryStick}
          label="Memory"
          value={metrics.memory}
          unit="%"
          color="text-green-400"
          progress={metrics.memory}
        />

        <MetricCard
          icon={HardDrive}
          label="Disk Usage"
          value={metrics.disk}
          unit="%"
          color="text-purple-400"
          progress={metrics.disk}
        />

        <MetricCard
          icon={Network}
          label="Network"
          value={metrics.network}
          unit="MB/s"
          color="text-orange-400"
        />

        <MetricCard
          icon={Zap}
          label="Active Services"
          value={metrics.activeServices}
          unit={`/ ${services.length}`}
          color="text-emerald-400"
        />

        <MetricCard
          icon={Activity}
          label="Uptime"
          value={metrics.uptime}
          color="text-pink-400"
        />
      </div>

      {/* Service Type Distribution */}
      <div className="mt-6">
        <h3 className="text-lg font-semibold text-white mb-3">Service Distribution</h3>
        <div className="space-y-2">
          {Object.entries(
            services.reduce((acc, service) => {
              acc[service.type] = (acc[service.type] || 0) + 1
              return acc
            }, {} as Record<string, number>)
          ).map(([type, count]) => (
            <div key={type} className="flex justify-between items-center">
              <span className="text-gray-300 capitalize">{type}</span>
              <span className="text-white font-semibold">{count}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}