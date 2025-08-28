'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Search, 
  Plus, 
  Clock, 
  CheckCircle, 
  XCircle,
  Star,
  Calendar,
  User,
  Filter,
  Tv,
  Film,
  Music
} from 'lucide-react'
import { NavigationHeader } from '@/components/Layout/NavigationHeader'

interface MediaRequest {
  id: string
  title: string
  type: 'movie' | 'tv' | 'music'
  year: number
  rating: number
  poster: string
  status: 'pending' | 'approved' | 'downloading' | 'completed' | 'denied'
  requestedBy: string
  requestedAt: string
  approvedBy?: string
  reason?: string
}

export default function RequestsPage() {
  const [requests, setRequests] = useState<MediaRequest[]>([])
  const [filter, setFilter] = useState('all')
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [showAddRequest, setShowAddRequest] = useState(false)

  useEffect(() => {
    // Simulate fetching requests data
    setTimeout(() => {
      setRequests([
        {
          id: '1',
          title: 'Spider-Man: No Way Home',
          type: 'movie',
          year: 2021,
          rating: 8.4,
          poster: '/api/placeholder/300/450',
          status: 'approved',
          requestedBy: 'John Doe',
          requestedAt: '2024-01-15T10:30:00Z',
          approvedBy: 'Admin'
        },
        {
          id: '2',
          title: 'House of the Dragon',
          type: 'tv',
          year: 2022,
          rating: 8.5,
          poster: '/api/placeholder/300/450',
          status: 'downloading',
          requestedBy: 'Jane Smith',
          requestedAt: '2024-01-14T15:45:00Z',
          approvedBy: 'Admin'
        },
        {
          id: '3',
          title: 'The Batman',
          type: 'movie',
          year: 2022,
          rating: 7.8,
          poster: '/api/placeholder/300/450',
          status: 'pending',
          requestedBy: 'Mike Wilson',
          requestedAt: '2024-01-16T09:15:00Z'
        },
        {
          id: '4',
          title: 'Stranger Things Season 5',
          type: 'tv',
          year: 2024,
          rating: 0,
          poster: '/api/placeholder/300/450',
          status: 'denied',
          requestedBy: 'Sarah Johnson',
          requestedAt: '2024-01-13T14:20:00Z',
          reason: 'Not yet released'
        },
        {
          id: '5',
          title: 'Adele - 30',
          type: 'music',
          year: 2021,
          rating: 9.1,
          poster: '/api/placeholder/300/300',
          status: 'completed',
          requestedBy: 'Alex Brown',
          requestedAt: '2024-01-12T11:30:00Z',
          approvedBy: 'Admin'
        }
      ])
      setLoading(false)
    }, 1000)
  }, [])

  const filteredRequests = requests.filter(request => {
    const matchesSearch = request.title.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesFilter = filter === 'all' || request.status === filter
    return matchesSearch && matchesFilter
  })

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending': return Clock
      case 'approved': return CheckCircle
      case 'downloading': return CheckCircle
      case 'completed': return CheckCircle
      case 'denied': return XCircle
      default: return Clock
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'text-yellow-400'
      case 'approved': return 'text-blue-400'
      case 'downloading': return 'text-purple-400'
      case 'completed': return 'text-green-400'
      case 'denied': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'movie': return Film
      case 'tv': return Tv
      case 'music': return Music
      default: return Film
    }
  }

  const handleAction = (id: string, action: 'approve' | 'deny') => {
    setRequests(prev => prev.map(request => {
      if (request.id === id) {
        return {
          ...request,
          status: action === 'approve' ? 'approved' : 'denied',
          approvedBy: action === 'approve' ? 'Admin' : undefined,
          reason: action === 'deny' ? 'Denied by administrator' : undefined
        }
      }
      return request
    }))
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <NavigationHeader />
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-purple-500 mx-auto mb-4"></div>
            <h2 className="text-2xl font-bold text-white">Loading Requests...</h2>
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
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">Media Requests</h1>
              <p className="text-gray-400">Manage and review user media requests</p>
            </div>
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowAddRequest(true)}
              className="bg-purple-500 hover:bg-purple-600 text-white px-6 py-3 rounded-xl flex items-center space-x-2 transition-colors"
            >
              <Plus className="w-5 h-5" />
              <span>New Request</span>
            </motion.button>
          </div>
        </motion.div>

        {/* Stats Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-1 md:grid-cols-5 gap-6 mb-8"
        >
          {[
            { label: 'Pending', value: requests.filter(r => r.status === 'pending').length, color: 'text-yellow-400', icon: Clock },
            { label: 'Approved', value: requests.filter(r => r.status === 'approved').length, color: 'text-blue-400', icon: CheckCircle },
            { label: 'Downloading', value: requests.filter(r => r.status === 'downloading').length, color: 'text-purple-400', icon: CheckCircle },
            { label: 'Completed', value: requests.filter(r => r.status === 'completed').length, color: 'text-green-400', icon: CheckCircle },
            { label: 'Denied', value: requests.filter(r => r.status === 'denied').length, color: 'text-red-400', icon: XCircle }
          ].map((stat, index) => {
            const Icon = stat.icon
            return (
              <div key={stat.label} className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">{stat.label}</p>
                    <p className={`text-2xl font-bold ${stat.color}`}>{stat.value}</p>
                  </div>
                  <Icon className={`w-8 h-8 ${stat.color}`} />
                </div>
              </div>
            )
          })}
        </motion.div>

        {/* Search and Filters */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10 mb-8"
        >
          <div className="flex flex-col lg:flex-row gap-4 items-center">
            {/* Search */}
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search requests..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>

            {/* Filters */}
            <div className="flex items-center space-x-4">
              <select
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="px-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="all">All Status</option>
                <option value="pending">Pending</option>
                <option value="approved">Approved</option>
                <option value="downloading">Downloading</option>
                <option value="completed">Completed</option>
                <option value="denied">Denied</option>
              </select>
            </div>
          </div>
        </motion.div>

        {/* Requests List */}
        <AnimatePresence>
          {filteredRequests.length > 0 ? (
            <div className="space-y-4">
              {filteredRequests.map((request) => {
                const StatusIcon = getStatusIcon(request.status)
                const TypeIcon = getTypeIcon(request.type)
                
                return (
                  <motion.div
                    key={request.id}
                    layout
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-6 hover:bg-white/10 transition-all"
                  >
                    <div className="flex items-start space-x-6">
                      {/* Poster placeholder */}
                      <div className="w-16 h-24 bg-gray-700 rounded-lg flex items-center justify-center flex-shrink-0">
                        <TypeIcon className="w-8 h-8 text-gray-400" />
                      </div>

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between mb-2">
                          <div>
                            <h3 className="text-xl font-semibold text-white mb-1">{request.title}</h3>
                            <div className="flex items-center space-x-4 text-sm text-gray-400">
                              <span className="capitalize">{request.type}</span>
                              <span>{request.year}</span>
                              {request.rating > 0 && (
                                <div className="flex items-center space-x-1">
                                  <Star className="w-4 h-4 text-yellow-400 fill-current" />
                                  <span>{request.rating}</span>
                                </div>
                              )}
                            </div>
                          </div>

                          <div className="flex items-center space-x-2">
                            <StatusIcon className={`w-5 h-5 ${getStatusColor(request.status)}`} />
                            <span className={`text-sm font-medium capitalize ${getStatusColor(request.status)}`}>
                              {request.status}
                            </span>
                          </div>
                        </div>

                        {/* Request details */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                          <div className="flex items-center space-x-2 text-sm text-gray-400">
                            <User className="w-4 h-4" />
                            <span>Requested by: {request.requestedBy}</span>
                          </div>
                          <div className="flex items-center space-x-2 text-sm text-gray-400">
                            <Calendar className="w-4 h-4" />
                            <span>{new Date(request.requestedAt).toLocaleDateString()}</span>
                          </div>
                          {request.approvedBy && (
                            <div className="flex items-center space-x-2 text-sm text-gray-400">
                              <CheckCircle className="w-4 h-4" />
                              <span>Approved by: {request.approvedBy}</span>
                            </div>
                          )}
                          {request.reason && (
                            <div className="flex items-center space-x-2 text-sm text-gray-400">
                              <XCircle className="w-4 h-4" />
                              <span>Reason: {request.reason}</span>
                            </div>
                          )}
                        </div>

                        {/* Actions */}
                        {request.status === 'pending' && (
                          <div className="flex items-center space-x-3">
                            <motion.button
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                              onClick={() => handleAction(request.id, 'approve')}
                              className="px-4 py-2 bg-green-500/20 text-green-400 rounded-lg text-sm hover:bg-green-500/30 transition-colors"
                            >
                              Approve
                            </motion.button>
                            <motion.button
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                              onClick={() => handleAction(request.id, 'deny')}
                              className="px-4 py-2 bg-red-500/20 text-red-400 rounded-lg text-sm hover:bg-red-500/30 transition-colors"
                            >
                              Deny
                            </motion.button>
                          </div>
                        )}
                      </div>
                    </div>
                  </motion.div>
                )
              })}
            </div>
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="text-center py-16"
            >
              <Search className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-white mb-2">No requests found</h2>
              <p className="text-gray-400">Try adjusting your search or filters</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}