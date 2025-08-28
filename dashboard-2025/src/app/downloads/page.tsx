'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Download, 
  Pause, 
  Play, 
  Trash2, 
  Filter,
  ArrowDown,
  ArrowUp,
  Clock,
  CheckCircle,
  AlertCircle,
  MoreHorizontal
} from 'lucide-react'
import { NavigationHeader } from '@/components/Layout/NavigationHeader'

interface DownloadItem {
  id: string
  name: string
  size: string
  progress: number
  speed: string
  eta: string
  status: 'downloading' | 'paused' | 'completed' | 'error' | 'queued'
  type: 'movie' | 'tv' | 'music'
  seeders?: number
  leechers?: number
}

export default function DownloadsPage() {
  const [downloads, setDownloads] = useState<DownloadItem[]>([])
  const [filter, setFilter] = useState('all')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Simulate fetching download data
    setTimeout(() => {
      setDownloads([
        {
          id: '1',
          name: 'The.Matrix.Resurrections.2021.2160p.BluRay.x265-SECTOR7',
          size: '15.2 GB',
          progress: 67,
          speed: '12.5 MB/s',
          eta: '8m 32s',
          status: 'downloading',
          type: 'movie',
          seeders: 45,
          leechers: 12
        },
        {
          id: '2',
          name: 'Stranger.Things.S04E01.2160p.NF.WEB-DL.x265-NTb',
          size: '8.7 GB',
          progress: 100,
          speed: '0 MB/s',
          eta: 'Completed',
          status: 'completed',
          type: 'tv',
          seeders: 89,
          leechers: 3
        },
        {
          id: '3',
          name: 'Dune.2021.IMAX.2160p.BluRay.x265-SECTOR7',
          size: '18.9 GB',
          progress: 23,
          speed: '0 MB/s',
          eta: 'Paused',
          status: 'paused',
          type: 'movie',
          seeders: 23,
          leechers: 7
        },
        {
          id: '4',
          name: 'Pink.Floyd.The.Dark.Side.of.the.Moon.FLAC',
          size: '342 MB',
          progress: 89,
          speed: '2.1 MB/s',
          eta: '1m 15s',
          status: 'downloading',
          type: 'music',
          seeders: 156,
          leechers: 4
        },
        {
          id: '5',
          name: 'Spider-Man.No.Way.Home.2021.2160p.BluRay.x265',
          size: '0 GB',
          progress: 0,
          speed: '0 MB/s',
          eta: 'Queued',
          status: 'queued',
          type: 'movie',
          seeders: 0,
          leechers: 0
        },
        {
          id: '6',
          name: 'The.Office.S01-S09.Complete.Series.x264-MiXED',
          size: '45.2 GB',
          progress: 12,
          speed: '0 MB/s',
          eta: 'Error',
          status: 'error',
          type: 'tv',
          seeders: 8,
          leechers: 15
        }
      ])
      setLoading(false)
    }, 1000)
  }, [])

  const filteredDownloads = downloads.filter(download => {
    if (filter === 'all') return true
    return download.status === filter
  })

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'downloading': return ArrowDown
      case 'completed': return CheckCircle
      case 'paused': return Pause
      case 'error': return AlertCircle
      case 'queued': return Clock
      default: return Download
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'downloading': return 'text-blue-400'
      case 'completed': return 'text-green-400'
      case 'paused': return 'text-yellow-400'
      case 'error': return 'text-red-400'
      case 'queued': return 'text-gray-400'
      default: return 'text-gray-400'
    }
  }

  const handleAction = (id: string, action: string) => {
    setDownloads(prev => prev.map(download => {
      if (download.id === id) {
        switch (action) {
          case 'pause':
            return { ...download, status: 'paused' as const, speed: '0 MB/s', eta: 'Paused' }
          case 'resume':
            return { ...download, status: 'downloading' as const }
          case 'delete':
            return null
          default:
            return download
        }
      }
      return download
    }).filter(Boolean) as DownloadItem[])
  }

  const DownloadCard = ({ download }: { download: DownloadItem }) => {
    const StatusIcon = getStatusIcon(download.status)
    
    return (
      <motion.div
        layout
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-6 hover:bg-white/10 transition-all"
      >
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-white text-lg mb-1 truncate">
              {download.name.replace(/\./g, ' ')}
            </h3>
            <div className="flex items-center space-x-4 text-sm text-gray-400">
              <span className="capitalize">{download.type}</span>
              <span>{download.size}</span>
              {download.seeders !== undefined && (
                <span>S: {download.seeders} L: {download.leechers}</span>
              )}
            </div>
          </div>
          
          <div className="flex items-center space-x-2 ml-4">
            <StatusIcon className={`w-5 h-5 ${getStatusColor(download.status)}`} />
            <button className="p-1 text-gray-400 hover:text-white transition-colors">
              <MoreHorizontal className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-gray-400">Progress</span>
            <span className="text-sm text-white font-medium">{download.progress}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${download.progress}%` }}
              transition={{ duration: 0.5 }}
              className={`h-2 rounded-full ${
                download.status === 'completed' ? 'bg-green-500' :
                download.status === 'error' ? 'bg-red-500' :
                download.status === 'paused' ? 'bg-yellow-500' :
                'bg-blue-500'
              }`}
            />
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <span className="text-xs text-gray-400 block">Speed</span>
            <span className="text-sm text-white font-medium">{download.speed}</span>
          </div>
          <div>
            <span className="text-xs text-gray-400 block">ETA</span>
            <span className="text-sm text-white font-medium">{download.eta}</span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center space-x-2">
          {download.status === 'downloading' && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleAction(download.id, 'pause')}
              className="px-3 py-1 bg-yellow-500/20 text-yellow-400 rounded-lg text-sm hover:bg-yellow-500/30 transition-colors"
            >
              <Pause className="w-4 h-4 inline mr-1" />
              Pause
            </motion.button>
          )}
          
          {download.status === 'paused' && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleAction(download.id, 'resume')}
              className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-lg text-sm hover:bg-blue-500/30 transition-colors"
            >
              <Play className="w-4 h-4 inline mr-1" />
              Resume
            </motion.button>
          )}
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => handleAction(download.id, 'delete')}
            className="px-3 py-1 bg-red-500/20 text-red-400 rounded-lg text-sm hover:bg-red-500/30 transition-colors"
          >
            <Trash2 className="w-4 h-4 inline mr-1" />
            Remove
          </motion.button>
        </div>
      </motion.div>
    )
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <NavigationHeader />
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-purple-500 mx-auto mb-4"></div>
            <h2 className="text-2xl font-bold text-white">Loading Downloads...</h2>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <NavigationHeader />
      
      <div className="container mx-auto px-6 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold text-white mb-2">Downloads</h1>
          <p className="text-gray-400">Monitor and manage your download queue</p>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8"
        >
          <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Active</p>
                <p className="text-2xl font-bold text-blue-400">
                  {downloads.filter(d => d.status === 'downloading').length}
                </p>
              </div>
              <ArrowDown className="w-8 h-8 text-blue-400" />
            </div>
          </div>

          <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Completed</p>
                <p className="text-2xl font-bold text-green-400">
                  {downloads.filter(d => d.status === 'completed').length}
                </p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-400" />
            </div>
          </div>

          <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Paused</p>
                <p className="text-2xl font-bold text-yellow-400">
                  {downloads.filter(d => d.status === 'paused').length}
                </p>
              </div>
              <Pause className="w-8 h-8 text-yellow-400" />
            </div>
          </div>

          <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Total Speed</p>
                <p className="text-2xl font-bold text-purple-400">
                  {downloads
                    .filter(d => d.status === 'downloading')
                    .reduce((total, d) => total + parseFloat(d.speed.replace(' MB/s', '')), 0)
                    .toFixed(1)} MB/s
                </p>
              </div>
              <ArrowUp className="w-8 h-8 text-purple-400" />
            </div>
          </div>
        </motion.div>

        {/* Filters */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10 mb-8"
        >
          <div className="flex flex-wrap gap-2">
            {['all', 'downloading', 'completed', 'paused', 'error', 'queued'].map((status) => (
              <motion.button
                key={status}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setFilter(status)}
                className={`px-4 py-2 rounded-lg text-sm transition-all ${
                  filter === status
                    ? 'bg-purple-500 text-white'
                    : 'bg-white/10 text-gray-300 hover:bg-white/20'
                }`}
              >
                {status === 'all' ? 'All' : status.charAt(0).toUpperCase() + status.slice(1)}
                <span className="ml-2 text-xs opacity-75">
                  {status === 'all' ? downloads.length : downloads.filter(d => d.status === status).length}
                </span>
              </motion.button>
            ))}
          </div>
        </motion.div>

        {/* Downloads List */}
        <AnimatePresence>
          {filteredDownloads.length > 0 ? (
            <div className="space-y-4">
              {filteredDownloads.map((download) => (
                <DownloadCard key={download.id} download={download} />
              ))}
            </div>
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="text-center py-16"
            >
              <Download className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-white mb-2">No downloads found</h2>
              <p className="text-gray-400">No downloads match the selected filter</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}