import React, { useState, useEffect } from 'react'
import { Box, Typography, useTheme, Container, IconButton, Fab } from '@mui/material'
import { motion, AnimatePresence } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import {
  PlayArrow,
  Movie,
  MusicNote,
  CloudDownload,
  Settings,
  Dashboard as DashboardIcon,
  Visibility,
  Speed,
  Storage,
  Security,
  TrendingUp,
  Refresh,
} from '@mui/icons-material'
import { fetchSystemStatus } from '../services/api'

// Define service data from docker-compose
const mediaServices = [
  {
    id: 'jellyfin',
    name: 'Jellyfin',
    description: 'Your personal Netflix - Stream anywhere, anytime',
    icon: '🍿',
    port: 8096,
    color: '#00a4dc',
    gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    category: 'Media Server',
    status: 'running',
    preview: '/api/placeholder/400/250',
    stats: { users: 12, movies: 1247, shows: 89, episodes: 3421 }
  },
  {
    id: 'overseerr',
    name: 'Overseerr',
    description: 'Request management made beautiful',
    icon: '🎬',
    port: 5055,
    color: '#E50914',
    gradient: 'linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)',
    category: 'Request Manager',
    status: 'running',
    preview: '/api/placeholder/400/250',
    stats: { pending: 8, approved: 156, total: 1832 }
  },
  {
    id: 'sonarr',
    name: 'Sonarr',
    description: 'Intelligent TV series management',
    icon: '📺',
    port: 8989,
    color: '#35C5F0',
    gradient: 'linear-gradient(135deg, #74b9ff 0%, #0984e3 100%)',
    category: 'TV Manager',
    status: 'running',
    preview: '/api/placeholder/400/250',
    stats: { series: 89, episodes: 2341, monitored: 67 }
  },
  {
    id: 'radarr',
    name: 'Radarr',
    description: 'Movie collection curator extraordinaire',
    icon: '🎭',
    port: 7878,
    color: '#FFC230',
    gradient: 'linear-gradient(135deg, #fdcb6e 0%, #e17055 100%)',
    category: 'Movie Manager',
    status: 'running',
    preview: '/api/placeholder/400/250',
    stats: { movies: 1247, monitored: 891, downloaded: 1180 }
  },
  {
    id: 'lidarr',
    name: 'Lidarr',
    description: 'Music library perfection',
    icon: '🎵',
    port: 8686,
    color: '#E91E63',
    gradient: 'linear-gradient(135deg, #fd79a8 0%, #e84393 100%)',
    category: 'Music Manager',
    status: 'running',
    preview: '/api/placeholder/400/250',
    stats: { artists: 234, albums: 1456, tracks: 18743 }
  },
  {
    id: 'bazarr',
    name: 'Bazarr',
    description: 'Subtitle sync specialist',
    icon: '📝',
    port: 6767,
    color: '#9C27B0',
    gradient: 'linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%)',
    category: 'Subtitles',
    status: 'running',
    preview: '/api/placeholder/400/250',
    stats: { languages: 12, downloaded: 8934, missing: 145 }
  },
  {
    id: 'prowlarr',
    name: 'Prowlarr',
    description: 'Indexer management hub',
    icon: '🔍',
    port: 9696,
    color: '#FF5722',
    gradient: 'linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%)',
    category: 'Indexer Manager',
    status: 'running',
    preview: '/api/placeholder/400/250',
    stats: { indexers: 15, searches: 2341, grabs: 567 }
  },
  {
    id: 'qbittorrent',
    name: 'qBittorrent',
    description: 'Torrent powerhouse via VPN',
    icon: '🌊',
    port: 8080,
    color: '#2E7D32',
    gradient: 'linear-gradient(135deg, #00b894 0%, #00cec9 100%)',
    category: 'Downloader',
    status: 'running',
    preview: '/api/placeholder/400/250',
    stats: { active: 12, downloading: 8, ratio: '2.34' }
  },
  {
    id: 'sabnzbd',
    name: 'SABnzbd',
    description: 'Usenet download master',
    icon: '📦',
    port: 8081,
    color: '#FFA726',
    gradient: 'linear-gradient(135deg, #fab1a0 0%, #fd79a8 100%)',
    category: 'Usenet Client',
    status: 'running',
    preview: '/api/placeholder/400/250',
    stats: { queue: 5, speed: '45 MB/s', completed: 8934 }
  },
  {
    id: 'tautulli',
    name: 'Tautulli',
    description: 'Media analytics & insights',
    icon: '📊',
    port: 8181,
    color: '#607D8B',
    gradient: 'linear-gradient(135deg, #74b9ff 0%, #55a3ff 100%)',
    category: 'Analytics',
    status: 'running',
    preview: '/api/placeholder/400/250',
    stats: { plays: 18934, users: 12, bandwidth: '156 GB' }
  },
  {
    id: 'grafana',
    name: 'Grafana',
    description: 'System performance monitoring',
    icon: '📈',
    port: 3000,
    color: '#F46800',
    gradient: 'linear-gradient(135deg, #ff7675 0%, #e17055 100%)',
    category: 'Monitoring',
    status: 'running',
    preview: '/api/placeholder/400/250',
    stats: { dashboards: 8, alerts: 2, uptime: '99.98%' }
  },
  {
    id: 'portainer',
    name: 'Portainer',
    description: 'Container management center',
    icon: '🐳',
    port: 9000,
    color: '#13BEF9',
    gradient: 'linear-gradient(135deg, #74b9ff 0%, #0984e3 100%)',
    category: 'Management',
    status: 'running',
    preview: '/api/placeholder/400/250',
    stats: { containers: 18, images: 24, volumes: 12 }
  }
]

const Dashboard: React.FC = () => {
  const theme = useTheme()
  const [currentTime, setCurrentTime] = useState(new Date())
  const [hoveredService, setHoveredService] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  
  const { data: systemStatus, isLoading, refetch } = useQuery({
    queryKey: ['systemStatus'],
    queryFn: fetchSystemStatus,
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  const handleRefresh = async () => {
    setRefreshing(true)
    await refetch()
    setTimeout(() => setRefreshing(false), 1000)
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      },
    },
  }

  const itemVariants = {
    hidden: { y: 40, opacity: 0, scale: 0.95 },
    visible: {
      y: 0,
      opacity: 1,
      scale: 1,
      transition: {
        type: 'spring',
        stiffness: 100,
        damping: 15,
      },
    },
  }

  const serviceCardVariants = {
    hidden: { y: 30, opacity: 0, rotateX: -15 },
    visible: {
      y: 0,
      opacity: 1,
      rotateX: 0,
      transition: {
        type: 'spring',
        stiffness: 120,
        damping: 20,
      },
    },
    hover: {
      y: -10,
      scale: 1.02,
      rotateX: 5,
      transition: {
        type: 'spring',
        stiffness: 400,
        damping: 25,
      },
    },
  }

  return (
    <Box sx={{ 
      minHeight: '100vh',
      background: `
        radial-gradient(ellipse at top, rgba(0, 255, 136, 0.1) 0%, transparent 50%),
        radial-gradient(ellipse at bottom, rgba(0, 136, 255, 0.1) 0%, transparent 50%),
        linear-gradient(135deg, #0a0a0a 0%, #1a0a1a 50%, #0a1a1a 100%)
      `,
      position: 'relative',
      overflow: 'hidden',
      '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundImage: `
          radial-gradient(circle at 20% 80%, rgba(0, 255, 136, 0.05) 0%, transparent 50%),
          radial-gradient(circle at 80% 20%, rgba(0, 136, 255, 0.05) 0%, transparent 50%),
          radial-gradient(circle at 40% 40%, rgba(255, 0, 136, 0.03) 0%, transparent 50%)
        `,
        animation: 'float 20s ease-in-out infinite',
        zIndex: -1,
      },
      '@keyframes float': {
        '0%, 100%': { transform: 'translateY(0px) rotate(0deg)' },
        '50%': { transform: 'translateY(-20px) rotate(180deg)' },
      },
    }}>
      <Container maxWidth="xl" sx={{ py: 4, position: 'relative', zIndex: 1 }}>
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Header Section */}
          <motion.div variants={itemVariants}>
            <Box sx={{ 
              mb: 6, 
              textAlign: 'center',
              position: 'relative',
            }}>
              <Typography
                variant="h1"
                component="h1"
                sx={{
                  fontSize: { xs: '2.5rem', md: '4rem', lg: '5rem' },
                  fontWeight: 800,
                  background: `linear-gradient(135deg, 
                    ${theme.palette.primary.main} 0%, 
                    ${theme.palette.secondary.main} 50%, 
                    ${theme.palette.error.main} 100%
                  )`,
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                  mb: 2,
                  textShadow: '0 0 30px rgba(0, 255, 136, 0.5)',
                  position: 'relative',
                  '&::after': {
                    content: '""',
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    width: '120%',
                    height: '120%',
                    background: `linear-gradient(45deg, 
                      rgba(0, 255, 136, 0.1), 
                      rgba(0, 136, 255, 0.1)
                    )`,
                    borderRadius: '50%',
                    filter: 'blur(50px)',
                    zIndex: -1,
                    animation: 'pulse 3s ease-in-out infinite',
                  },
                }}
              >
                🚀 ULTIMATE MEDIA HUB
              </Typography>
              <Typography 
                variant="h6" 
                sx={{ 
                  color: 'rgba(255, 255, 255, 0.8)',
                  fontSize: { xs: '1rem', md: '1.25rem' },
                  fontWeight: 300,
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  mb: 1,
                }}
              >
                Your Digital Entertainment Empire
              </Typography>
              <Typography 
                variant="body1" 
                sx={{ 
                  color: 'rgba(255, 255, 255, 0.6)',
                  fontSize: '1.1rem',
                  maxWidth: '600px',
                  mx: 'auto',
                  lineHeight: 1.6,
                }}
              >
                {currentTime.toLocaleString()} • {mediaServices.filter(s => s.status === 'running').length} Services Running • All Systems Optimal
              </Typography>
            </Box>
          </motion.div>

          {/* Stats Overview */}
          <motion.div variants={itemVariants}>
            <Box sx={{ 
              mb: 6,
              display: 'grid',
              gridTemplateColumns: { xs: '1fr 1fr', md: 'repeat(4, 1fr)' },
              gap: 3,
            }}>
              {[
                { icon: <Movie />, label: 'Movies', value: '1,247', color: '#E50914' },
                { icon: <PlayArrow />, label: 'TV Shows', value: '89', color: '#00a4dc' },
                { icon: <MusicNote />, label: 'Albums', value: '1,456', color: '#E91E63' },
                { icon: <CloudDownload />, label: 'Downloads', value: '20', color: '#2E7D32' },
              ].map((stat, index) => (
                <motion.div
                  key={stat.label}
                  whileHover={{ scale: 1.05, y: -5 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Box sx={{
                    background: 'rgba(255, 255, 255, 0.03)',
                    backdropFilter: 'blur(20px)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: 3,
                    p: 3,
                    textAlign: 'center',
                    position: 'relative',
                    overflow: 'hidden',
                    '&::before': {
                      content: '""',
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      height: '3px',
                      background: stat.color,
                      borderRadius: '3px 3px 0 0',
                    },
                    '&:hover': {
                      border: `1px solid ${stat.color}30`,
                      boxShadow: `0 10px 40px ${stat.color}20`,
                    },
                  }}>
                    <Box sx={{ color: stat.color, mb: 1 }}>
                      {stat.icon}
                    </Box>
                    <Typography variant="h4" sx={{ fontWeight: 700, mb: 0.5 }}>
                      {stat.value}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {stat.label}
                    </Typography>
                  </Box>
                </motion.div>
              ))}
            </Box>
          </motion.div>

          {/* Service Grid */}
          <motion.div variants={itemVariants}>
            <Box sx={{ mb: 4 }}>
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                mb: 3,
              }}>
                <Typography 
                  variant="h4" 
                  sx={{ 
                    fontWeight: 700,
                    color: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 2,
                  }}
                >
                  <DashboardIcon sx={{ color: theme.palette.primary.main }} />
                  Media Services
                </Typography>
                <IconButton 
                  onClick={handleRefresh}
                  sx={{ 
                    color: theme.palette.primary.main,
                    '&:hover': {
                      backgroundColor: 'rgba(0, 255, 136, 0.1)',
                      transform: 'rotate(180deg)',
                    },
                    transition: 'all 0.3s ease',
                  }}
                >
                  <Refresh sx={{ 
                    animation: refreshing ? 'spin 1s linear infinite' : 'none',
                    '@keyframes spin': {
                      '0%': { transform: 'rotate(0deg)' },
                      '100%': { transform: 'rotate(360deg)' },
                    },
                  }} />
                </IconButton>
              </Box>
              
              <Box sx={{
                display: 'grid',
                gridTemplateColumns: {
                  xs: '1fr',
                  sm: 'repeat(2, 1fr)',
                  md: 'repeat(3, 1fr)',
                  lg: 'repeat(4, 1fr)',
                },
                gap: 3,
              }}>
                <AnimatePresence>
                  {mediaServices.map((service, index) => (
                    <motion.div
                      key={service.id}
                      variants={serviceCardVariants}
                      initial="hidden"
                      animate="visible"
                      whileHover="hover"
                      custom={index}
                      onMouseEnter={() => setHoveredService(service.id)}
                      onMouseLeave={() => setHoveredService(null)}
                      style={{ transformStyle: 'preserve-3d' }}
                    >
                      <Box
                        component="a"
                        href={`http://localhost:${service.port}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        sx={{
                          display: 'block',
                          textDecoration: 'none',
                          color: 'inherit',
                          height: '100%',
                        }}
                      >
                        <Box sx={{
                          background: hoveredService === service.id 
                            ? `linear-gradient(135deg, ${service.color}20 0%, rgba(255, 255, 255, 0.05) 100%)`
                            : 'rgba(255, 255, 255, 0.03)',
                          backdropFilter: 'blur(20px)',
                          border: hoveredService === service.id 
                            ? `1px solid ${service.color}50`
                            : '1px solid rgba(255, 255, 255, 0.1)',
                          borderRadius: 4,
                          overflow: 'hidden',
                          position: 'relative',
                          height: '280px',
                          cursor: 'pointer',
                          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                          '&:hover': {
                            boxShadow: `
                              0 25px 50px ${service.color}20,
                              0 0 0 1px ${service.color}30,
                              inset 0 0 50px rgba(255, 255, 255, 0.05)
                            `,
                          },
                        }}>
                          {/* Service Header */}
                          <Box sx={{
                            p: 2.5,
                            background: service.gradient,
                            position: 'relative',
                            height: '120px',
                            display: 'flex',
                            flexDirection: 'column',
                            justifyContent: 'center',
                            alignItems: 'center',
                            '&::before': {
                              content: '""',
                              position: 'absolute',
                              top: 0,
                              left: 0,
                              right: 0,
                              bottom: 0,
                              background: 'rgba(0, 0, 0, 0.2)',
                              backdropFilter: 'blur(10px)',
                            },
                          }}>
                            <Typography 
                              sx={{ 
                                fontSize: '3rem', 
                                mb: 1,
                                position: 'relative',
                                zIndex: 1,
                                filter: 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3))',
                              }}
                            >
                              {service.icon}
                            </Typography>
                            <Box sx={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: 1,
                              position: 'relative',
                              zIndex: 1,
                            }}>
                              <Box sx={{
                                width: 8,
                                height: 8,
                                borderRadius: '50%',
                                backgroundColor: service.status === 'running' ? '#4ade80' : '#f87171',
                                boxShadow: service.status === 'running' 
                                  ? '0 0 10px #4ade80' 
                                  : '0 0 10px #f87171',
                                animation: service.status === 'running' ? 'pulse 2s infinite' : 'none',
                              }} />
                              <Typography 
                                variant="caption" 
                                sx={{ 
                                  color: 'white',
                                  textTransform: 'uppercase',
                                  fontWeight: 600,
                                  letterSpacing: '0.05em',
                                  textShadow: '0 1px 2px rgba(0, 0, 0, 0.5)',
                                }}
                              >
                                {service.status}
                              </Typography>
                            </Box>
                          </Box>

                          {/* Service Content */}
                          <Box sx={{ p: 2.5, height: 'calc(100% - 120px)', display: 'flex', flexDirection: 'column' }}>
                            <Typography 
                              variant="h6" 
                              sx={{ 
                                fontWeight: 700, 
                                mb: 1,
                                color: 'white',
                                fontSize: '1.1rem',
                              }}
                            >
                              {service.name}
                            </Typography>
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                color: 'rgba(255, 255, 255, 0.7)',
                                mb: 2,
                                lineHeight: 1.4,
                                fontSize: '0.875rem',
                                flex: 1,
                              }}
                            >
                              {service.description}
                            </Typography>
                            
                            {/* Stats */}
                            <Box sx={{ 
                              display: 'flex', 
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              pt: 1.5,
                              borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                            }}>
                              <Typography 
                                variant="caption" 
                                sx={{ 
                                  color: service.color,
                                  fontWeight: 600,
                                  textTransform: 'uppercase',
                                  letterSpacing: '0.05em',
                                }}
                              >
                                :{service.port}
                              </Typography>
                              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                                <Visibility sx={{ fontSize: 16, color: 'rgba(255, 255, 255, 0.5)' }} />
                                <Typography 
                                  variant="caption" 
                                  sx={{ 
                                    color: 'rgba(255, 255, 255, 0.7)',
                                    fontWeight: 500,
                                  }}
                                >
                                  Open
                                </Typography>
                              </Box>
                            </Box>
                          </Box>
                        </Box>
                      </Box>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </Box>
            </Box>
          </motion.div>
        </motion.div>
      </Container>
      
      {/* Floating Action Button */}
      <Fab
        sx={{
          position: 'fixed',
          bottom: 32,
          right: 32,
          background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
          color: 'black',
          width: 64,
          height: 64,
          '&:hover': {
            transform: 'scale(1.1) rotate(90deg)',
            boxShadow: `0 0 30px ${theme.palette.primary.main}80`,
          },
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        }}
        onClick={() => window.location.reload()}
      >
        <Settings sx={{ fontSize: 28 }} />
      </Fab>
    </Box>
  )
}

export default Dashboard