import React from 'react'
import {
  Card,
  CardContent,
  Grid,
  Typography,
  Box,
  LinearProgress,
  Chip,
  Skeleton,
  Tooltip,
  IconButton,
} from '@mui/material'
import {
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  Speed,
  Storage,
  Memory,
  NetworkCheck,
  Refresh,
} from '@mui/icons-material'
import { motion } from 'framer-motion'
import { format } from 'date-fns'

interface SystemStatusProps {
  status?: {
    overall: 'healthy' | 'warning' | 'error'
    uptime: string
    cpu: {
      usage: number
      temperature: number
      cores: number
    }
    memory: {
      used: number
      total: number
      percentage: number
    }
    storage: {
      used: number
      total: number
      percentage: number
    }
    network: {
      download: number
      upload: number
      latency: number
    }
    services: {
      running: number
      total: number
      errors: number
    }
    lastUpdated: string
  }
  isLoading: boolean
}

const SystemStatus: React.FC<SystemStatusProps> = ({ status, isLoading }) => {
  const getStatusIcon = (overallStatus: string) => {
    switch (overallStatus) {
      case 'healthy':
        return <CheckCircle sx={{ fontSize: 40, color: '#00ff88' }} />
      case 'warning':
        return <Warning sx={{ fontSize: 40, color: '#ffaa00' }} />
      case 'error':
        return <ErrorIcon sx={{ fontSize: 40, color: '#ff3366' }} />
      default:
        return null
    }
  }

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatSpeed = (bytesPerSecond: number) => {
    return formatBytes(bytesPerSecond) + '/s'
  }

  if (isLoading) {
    return (
      <Card>
        <CardContent>
          <Grid container spacing={3}>
            {[1, 2, 3, 4].map((i) => (
              <Grid item xs={12} sm={6} md={3} key={i}>
                <Skeleton variant="rectangular" height={120} sx={{ borderRadius: 2 }} />
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
    )
  }

  const systemData = status || {
    overall: 'healthy',
    uptime: '5d 12h 34m',
    cpu: { usage: 45, temperature: 62, cores: 8 },
    memory: { used: 8.5, total: 16, percentage: 53 },
    storage: { used: 750, total: 2000, percentage: 37.5 },
    network: { download: 125000, upload: 45000, latency: 12 },
    services: { running: 15, total: 18, errors: 0 },
    lastUpdated: new Date().toISOString(),
  }

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {getStatusIcon(systemData.overall)}
            <Box>
              <Typography variant="h5">System Status</Typography>
              <Typography variant="body2" color="text.secondary">
                Uptime: {systemData.uptime}
              </Typography>
            </Box>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip
              label={`${systemData.services.running}/${systemData.services.total} Services`}
              color={systemData.services.errors > 0 ? 'error' : 'success'}
              size="small"
            />
            <Tooltip title="Refresh">
              <IconButton size="small">
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        <Grid container spacing={3}>
          {/* CPU Status */}
          <Grid item xs={12} sm={6} md={3}>
            <motion.div whileHover={{ scale: 1.02 }} transition={{ duration: 0.2 }}>
              <Box
                sx={{
                  p: 2,
                  borderRadius: 2,
                  background: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  height: '100%',
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <Speed color="primary" />
                  <Typography variant="subtitle1">CPU</Typography>
                </Box>
                <Typography variant="h4" sx={{ mb: 1 }}>
                  {systemData.cpu.usage}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={systemData.cpu.usage}
                  sx={{
                    height: 6,
                    borderRadius: 3,
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 3,
                      background: systemData.cpu.usage > 80
                        ? 'linear-gradient(90deg, #ff3366 0%, #cc1144 100%)'
                        : systemData.cpu.usage > 60
                        ? 'linear-gradient(90deg, #ffaa00 0%, #cc8800 100%)'
                        : 'linear-gradient(90deg, #00ff88 0%, #00cc66 100%)',
                    },
                  }}
                />
                <Box sx={{ mt: 1, display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="caption" color="text.secondary">
                    {systemData.cpu.cores} cores
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {systemData.cpu.temperature}°C
                  </Typography>
                </Box>
              </Box>
            </motion.div>
          </Grid>

          {/* Memory Status */}
          <Grid item xs={12} sm={6} md={3}>
            <motion.div whileHover={{ scale: 1.02 }} transition={{ duration: 0.2 }}>
              <Box
                sx={{
                  p: 2,
                  borderRadius: 2,
                  background: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  height: '100%',
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <Memory color="secondary" />
                  <Typography variant="subtitle1">Memory</Typography>
                </Box>
                <Typography variant="h4" sx={{ mb: 1 }}>
                  {systemData.memory.percentage}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={systemData.memory.percentage}
                  sx={{
                    height: 6,
                    borderRadius: 3,
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 3,
                      background: 'linear-gradient(90deg, #0088ff 0%, #0066cc 100%)',
                    },
                  }}
                />
                <Box sx={{ mt: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    {systemData.memory.used}GB / {systemData.memory.total}GB
                  </Typography>
                </Box>
              </Box>
            </motion.div>
          </Grid>

          {/* Storage Status */}
          <Grid item xs={12} sm={6} md={3}>
            <motion.div whileHover={{ scale: 1.02 }} transition={{ duration: 0.2 }}>
              <Box
                sx={{
                  p: 2,
                  borderRadius: 2,
                  background: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  height: '100%',
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <Storage color="warning" />
                  <Typography variant="subtitle1">Storage</Typography>
                </Box>
                <Typography variant="h4" sx={{ mb: 1 }}>
                  {systemData.storage.percentage.toFixed(1)}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={systemData.storage.percentage}
                  sx={{
                    height: 6,
                    borderRadius: 3,
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 3,
                      background: 'linear-gradient(90deg, #ffaa00 0%, #cc8800 100%)',
                    },
                  }}
                />
                <Box sx={{ mt: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    {formatBytes(systemData.storage.used * 1024 * 1024 * 1024)} / {formatBytes(systemData.storage.total * 1024 * 1024 * 1024)}
                  </Typography>
                </Box>
              </Box>
            </motion.div>
          </Grid>

          {/* Network Status */}
          <Grid item xs={12} sm={6} md={3}>
            <motion.div whileHover={{ scale: 1.02 }} transition={{ duration: 0.2 }}>
              <Box
                sx={{
                  p: 2,
                  borderRadius: 2,
                  background: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  height: '100%',
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <NetworkCheck color="info" />
                  <Typography variant="subtitle1">Network</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1, mb: 1 }}>
                  <Typography variant="h4">
                    {systemData.network.latency}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    ms
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                  <Box>
                    <Typography variant="caption" color="text.secondary" display="block">
                      Download
                    </Typography>
                    <Typography variant="body2">
                      {formatSpeed(systemData.network.download)}
                    </Typography>
                  </Box>
                  <Box sx={{ textAlign: 'right' }}>
                    <Typography variant="caption" color="text.secondary" display="block">
                      Upload
                    </Typography>
                    <Typography variant="body2">
                      {formatSpeed(systemData.network.upload)}
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </motion.div>
          </Grid>
        </Grid>

        <Box sx={{ mt: 2, textAlign: 'right' }}>
          <Typography variant="caption" color="text.secondary">
            Last updated: {format(new Date(systemData.lastUpdated), 'HH:mm:ss')}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  )
}

export default SystemStatus