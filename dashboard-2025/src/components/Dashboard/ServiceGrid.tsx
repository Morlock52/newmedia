'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Play, 
  Download, 
  Search, 
  Monitor, 
  Shield, 
  Settings,
  ExternalLink,
  Circle,
  AlertCircle,
  CheckCircle
} from 'lucide-react'

interface Service {
  name: string
  port: number
  type: string
  color: string
  status: string
}

interface ServiceGridProps {
  services: Service[]
}

const getServiceIcon = (type: string) => {
  switch (type) {
    case 'media': return Play
    case 'download': return Download
    case 'arr': return Search
    case 'monitoring': return Monitor
    case 'security': return Shield
    case 'auth': return Shield
    default: return Settings
  }
}

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'online': return CheckCircle
    case 'offline': return AlertCircle
    default: return Circle
  }
}

export function ServiceGrid({ services }: ServiceGridProps) {
  const [filter, setFilter] = useState('all')
  const [hoveredService, setHoveredService] = useState<string | null>(null)

  const serviceTypes = ['all', ...Array.from(new Set(services.map(s => s.type)))]
  
  const filteredServices = filter === 'all' 
    ? services 
    : services.filter(s => s.type === filter)

  const handleServiceClick = (service: Service) => {
    const baseUrl = window.location.hostname
    const serviceUrl = `http://${baseUrl}:${service.port}`
    window.open(serviceUrl, '_blank')
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-2">Services</h2>
        <p className="text-gray-400">Manage your media server stack</p>
      </div>

      {/* Filter Tabs */}
      <div className="flex flex-wrap gap-2 justify-center">
        {serviceTypes.map((type) => (
          <motion.button
            key={type}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setFilter(type)}
            className={`px-3 py-1 rounded-full text-sm transition-all ${
              filter === type
                ? 'bg-purple-500 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
            }`}
          >
            {type === 'all' ? 'All' : type.charAt(0).toUpperCase() + type.slice(1)}
          </motion.button>
        ))}
      </div>

      {/* Service Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 max-h-96 overflow-y-auto custom-scrollbar">
        <AnimatePresence>
          {filteredServices.map((service, index) => {
            const Icon = getServiceIcon(service.type)
            const StatusIcon = getStatusIcon(service.status)
            
            return (
              <motion.div
                key={service.name}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ delay: index * 0.05 }}
                whileHover={{ 
                  scale: 1.05, 
                  y: -2,
                  boxShadow: `0 10px 30px ${service.color}40`
                }}
                onHoverStart={() => setHoveredService(service.name)}
                onHoverEnd={() => setHoveredService(null)}
                onClick={() => handleServiceClick(service)}
                className="bg-white/5 backdrop-blur-sm rounded-xl p-3 border border-white/10 cursor-pointer group relative overflow-hidden"
              >
                {/* Background gradient on hover */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ 
                    opacity: hoveredService === service.name ? 0.1 : 0 
                  }}
                  className="absolute inset-0 rounded-xl"
                  style={{ backgroundColor: service.color }}
                />

                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <Icon 
                        className="w-4 h-4" 
                        style={{ color: service.color }} 
                      />
                      <StatusIcon 
                        className={`w-3 h-3 ${
                          service.status === 'online' ? 'text-green-400' : 'text-red-400'
                        }`} 
                      />
                    </div>
                    <ExternalLink className="w-3 h-3 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>

                  <h3 className="font-semibold text-white text-sm mb-1">
                    {service.name}
                  </h3>
                  
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-400 capitalize">
                      {service.type}
                    </span>
                    <span className="text-gray-500">
                      :{service.port}
                    </span>
                  </div>

                  {/* Connection indicator */}
                  <div className="mt-2 flex items-center space-x-2">
                    <div 
                      className={`w-2 h-2 rounded-full ${
                        service.status === 'online' ? 'bg-green-400' : 'bg-red-400'
                      }`}
                    />
                    <span className={`text-xs ${
                      service.status === 'online' ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {service.status}
                    </span>
                  </div>
                </div>

                {/* Animated border */}
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ 
                    scale: hoveredService === service.name ? 1 : 0 
                  }}
                  className="absolute inset-0 rounded-xl border-2 opacity-50"
                  style={{ borderColor: service.color }}
                />
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-3 gap-4 mt-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-green-400">
            {services.filter(s => s.status === 'online').length}
          </div>
          <div className="text-xs text-gray-400">Online</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-red-400">
            {services.filter(s => s.status === 'offline').length}
          </div>
          <div className="text-xs text-gray-400">Offline</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-400">
            {services.length}
          </div>
          <div className="text-xs text-gray-400">Total</div>
        </div>
      </div>

      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 2px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(139, 92, 246, 0.5);
          border-radius: 2px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(139, 92, 246, 0.8);
        }
      `}</style>
    </div>
  )
}