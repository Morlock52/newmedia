'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Search, 
  Filter, 
  Play, 
  Download, 
  Star, 
  Calendar,
  Monitor,
  Grid,
  List,
  Image as ImageIcon
} from 'lucide-react'
import { NavigationHeader } from '@/components/Layout/NavigationHeader'

interface MediaItem {
  id: string
  title: string
  type: 'movie' | 'tv' | 'music'
  year: number
  rating: number
  poster: string
  status: 'available' | 'downloading' | 'pending'
  progress?: number
}

export default function MediaPage() {
  const [searchTerm, setSearchTerm] = useState('')
  const [filter, setFilter] = useState('all')
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [mediaItems, setMediaItems] = useState<MediaItem[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Simulate fetching media data
    setTimeout(() => {
      setMediaItems([
        {
          id: '1',
          title: 'The Dark Knight',
          type: 'movie',
          year: 2008,
          rating: 9.0,
          poster: '/api/placeholder/300/450',
          status: 'available'
        },
        {
          id: '2',
          title: 'Breaking Bad',
          type: 'tv',
          year: 2008,
          rating: 9.5,
          poster: '/api/placeholder/300/450',
          status: 'available'
        },
        {
          id: '3',
          title: 'Inception',
          type: 'movie',
          year: 2010,
          rating: 8.8,
          poster: '/api/placeholder/300/450',
          status: 'downloading',
          progress: 67
        },
        {
          id: '4',
          title: 'Stranger Things',
          type: 'tv',
          year: 2016,
          rating: 8.7,
          poster: '/api/placeholder/300/450',
          status: 'available'
        },
        {
          id: '5',
          title: 'Dune',
          type: 'movie',
          year: 2021,
          rating: 8.0,
          poster: '/api/placeholder/300/450',
          status: 'pending'
        },
        {
          id: '6',
          title: 'The Beatles - Abbey Road',
          type: 'music',
          year: 1969,
          rating: 9.3,
          poster: '/api/placeholder/300/300',
          status: 'available'
        }
      ])
      setLoading(false)
    }, 1000)
  }, [])

  const filteredItems = mediaItems.filter(item => {
    const matchesSearch = item.title.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesFilter = filter === 'all' || item.type === filter
    return matchesSearch && matchesFilter
  })

  const MediaCard = ({ item }: { item: MediaItem }) => (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      whileHover={{ scale: 1.05, y: -5 }}
      className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 overflow-hidden group cursor-pointer"
    >
      {/* Poster */}
      <div className="relative aspect-[3/4] bg-gray-800">
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent z-10" />
        <div className="w-full h-full bg-gray-700 flex items-center justify-center">
          <ImageIcon className="w-16 h-16 text-gray-500" />
        </div>
        
        {/* Status Badge */}
        <div className={`absolute top-2 right-2 z-20 px-2 py-1 rounded-full text-xs font-medium ${
          item.status === 'available' ? 'bg-green-500/80 text-white' :
          item.status === 'downloading' ? 'bg-blue-500/80 text-white' :
          'bg-orange-500/80 text-white'
        }`}>
          {item.status === 'downloading' && item.progress ? `${item.progress}%` : item.status}
        </div>

        {/* Download Progress */}
        {item.status === 'downloading' && item.progress && (
          <div className="absolute bottom-0 left-0 right-0 z-20">
            <div className="bg-black/60 h-2">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${item.progress}%` }}
                className="h-full bg-blue-500"
              />
            </div>
          </div>
        )}

        {/* Hover Actions */}
        <motion.div
          initial={{ opacity: 0 }}
          whileHover={{ opacity: 1 }}
          className="absolute inset-0 z-30 flex items-center justify-center space-x-3"
        >
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            className="p-3 bg-purple-500/80 rounded-full backdrop-blur-sm"
          >
            <Play className="w-6 h-6 text-white" />
          </motion.button>
          
          {item.status === 'available' && (
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              className="p-3 bg-green-500/80 rounded-full backdrop-blur-sm"
            >
              <Monitor className="w-6 h-6 text-white" />
            </motion.button>
          )}
        </motion.div>
      </div>

      {/* Content */}
      <div className="p-4">
        <h3 className="font-semibold text-white text-lg mb-1 truncate">
          {item.title}
        </h3>
        
        <div className="flex items-center justify-between text-sm text-gray-400 mb-2">
          <span className="capitalize">{item.type}</span>
          <span>{item.year}</span>
        </div>
        
        <div className="flex items-center space-x-1">
          <Star className="w-4 h-4 text-yellow-400 fill-current" />
          <span className="text-white font-medium">{item.rating}</span>
        </div>
      </div>
    </motion.div>
  )

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <NavigationHeader />
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-purple-500 mx-auto mb-4"></div>
            <h2 className="text-2xl font-bold text-white">Loading Media Library...</h2>
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
          <h1 className="text-4xl font-bold text-white mb-2">Media Library</h1>
          <p className="text-gray-400">Browse and manage your media collection</p>
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
                placeholder="Search media..."
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
                <option value="all">All Types</option>
                <option value="movie">Movies</option>
                <option value="tv">TV Shows</option>
                <option value="music">Music</option>
              </select>

              {/* View Mode Toggle */}
              <div className="flex bg-white/10 rounded-xl p-1">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 rounded-lg transition-all ${
                    viewMode === 'grid' ? 'bg-purple-500 text-white' : 'text-gray-400'
                  }`}
                >
                  <Grid className="w-5 h-5" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 rounded-lg transition-all ${
                    viewMode === 'list' ? 'bg-purple-500 text-white' : 'text-gray-400'
                  }`}
                >
                  <List className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Results Count */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mb-6"
        >
          <p className="text-gray-400">
            Showing {filteredItems.length} of {mediaItems.length} items
          </p>
        </motion.div>

        {/* Media Grid */}
        <AnimatePresence mode="wait">
          {filteredItems.length > 0 ? (
            <motion.div
              key="media-grid"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className={`grid gap-6 ${
                viewMode === 'grid'
                  ? 'grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5'
                  : 'grid-cols-1 max-w-4xl mx-auto'
              }`}
            >
              {filteredItems.map((item) => (
                <MediaCard key={item.id} item={item} />
              ))}
            </motion.div>
          ) : (
            <motion.div
              key="no-results"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="text-center py-16"
            >
              <Search className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-white mb-2">No media found</h2>
              <p className="text-gray-400">Try adjusting your search or filters</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}