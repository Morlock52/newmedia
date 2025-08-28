'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Pause, 
  Play, 
  RotateCcw, 
  Download, 
  Search, 
  Plus,
  Zap,
  Settings,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  Activity
} from 'lucide-react'

export function QuickActions() {
  const [isLoading, setIsLoading] = useState<string | null>(null)
  const [notifications, setNotifications] = useState([
    { id: 1, type: 'success', message: 'All services online', time: '2m ago' },
    { id: 2, type: 'info', message: 'New movie downloaded', time: '5m ago' },
    { id: 3, type: 'warning', message: 'High CPU usage detected', time: '10m ago' },
  ])

  const handleAction = async (actionName: string) => {
    setIsLoading(actionName)
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000))
    
    setIsLoading(null)
    
    // Add notification
    const newNotification = {
      id: Date.now(),
      type: 'success' as const,
      message: `${actionName} completed successfully`,
      time: 'now'
    }
    
    setNotifications(prev => [newNotification, ...prev.slice(0, 4)])
  }

  const ActionButton = ({ 
    icon: Icon, 
    label, 
    color,
    action 
  }: {
    icon: any
    label: string
    color: string
    action: () => void
  }) => (
    <motion.button
      whileHover={{ scale: 1.05, y: -2 }}
      whileTap={{ scale: 0.95 }}
      onClick={action}
      disabled={isLoading !== null}
      className={`relative p-4 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10 
        hover:bg-white/10 transition-all group overflow-hidden
        ${isLoading === label ? 'animate-pulse' : ''}
      `}
    >
      {/* Background glow on hover */}
      <motion.div
        initial={{ opacity: 0 }}
        whileHover={{ opacity: 0.1 }}
        className={`absolute inset-0 rounded-xl ${color}`}
      />
      
      <div className="relative z-10 flex flex-col items-center space-y-2">
        {isLoading === label ? (
          <RefreshCw className="w-6 h-6 text-white animate-spin" />
        ) : (
          <Icon className={`w-6 h-6 ${color.replace('bg-', 'text-')}`} />
        )}
        <span className="text-sm text-white font-medium">{label}</span>
      </div>

      {/* Animated border */}
      <motion.div
        initial={{ scale: 0 }}
        whileHover={{ scale: 1 }}
        className={`absolute inset-0 rounded-xl border-2 opacity-50 ${
          color.replace('bg-', 'border-')
        }`}
      />
    </motion.button>
  )

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-2">Quick Actions</h2>
        <p className="text-gray-400">Control your media server</p>
      </div>

      {/* Action Buttons Grid */}
      <div className="grid grid-cols-2 gap-3">
        <ActionButton
          icon={Play}
          label="Start All"
          color="bg-green-500"
          action={() => handleAction('Start All')}
        />
        
        <ActionButton
          icon={Pause}
          label="Stop All"
          color="bg-red-500"
          action={() => handleAction('Stop All')}
        />
        
        <ActionButton
          icon={RotateCcw}
          label="Restart"
          color="bg-blue-500"
          action={() => handleAction('Restart')}
        />
        
        <ActionButton
          icon={RefreshCw}
          label="Update"
          color="bg-purple-500"
          action={() => handleAction('Update')}
        />
        
        <ActionButton
          icon={Search}
          label="Scan Media"
          color="bg-orange-500"
          action={() => handleAction('Scan Media')}
        />
        
        <ActionButton
          icon={Download}
          label="Check Updates"
          color="bg-cyan-500"
          action={() => handleAction('Check Updates')}
        />
      </div>

      {/* System Status */}
      <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-white/10">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-white">System Status</h3>
          <Activity className="w-5 h-5 text-green-400" />
        </div>
        
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-gray-300">Docker</span>
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-green-400" />
              <span className="text-green-400 text-sm">Running</span>
            </div>
          </div>
          
          <div className="flex justify-between items-center">
            <span className="text-gray-300">VPN</span>
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-green-400" />
              <span className="text-green-400 text-sm">Connected</span>
            </div>
          </div>
          
          <div className="flex justify-between items-center">
            <span className="text-gray-300">Storage</span>
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 text-yellow-400" />
              <span className="text-yellow-400 text-sm">75% Full</span>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Notifications */}
      <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-3">Recent Activity</h3>
        
        <div className="space-y-3 max-h-40 overflow-y-auto custom-scrollbar">
          {notifications.map((notification) => (
            <motion.div
              key={notification.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-start space-x-3"
            >
              <div className={`w-2 h-2 rounded-full mt-2 ${
                notification.type === 'success' ? 'bg-green-400' :
                notification.type === 'warning' ? 'bg-yellow-400' :
                'bg-blue-400'
              }`} />
              
              <div className="flex-1 min-w-0">
                <p className="text-sm text-gray-300">{notification.message}</p>
                <p className="text-xs text-gray-500">{notification.time}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-white/5 backdrop-blur-sm rounded-xl p-3 border border-white/10 text-center">
          <div className="text-xl font-bold text-white">2.5TB</div>
          <div className="text-xs text-gray-400">Downloaded</div>
        </div>
        
        <div className="bg-white/5 backdrop-blur-sm rounded-xl p-3 border border-white/10 text-center">
          <div className="text-xl font-bold text-white">147</div>
          <div className="text-xs text-gray-400">Movies</div>
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