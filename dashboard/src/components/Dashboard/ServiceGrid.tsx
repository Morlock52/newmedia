import React, { useState } from 'react'
import {
  Card,
  CardContent,
  Grid,
  Typography,
  IconButton,
  Chip,
  Box,
  Tooltip,
  LinearProgress,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Switch,
} from '@mui/material'
import {
  PlayArrow,
  Stop,
  Refresh,
  MoreVert,
  Settings,
  Terminal,
  Delete,
  Info,
  CheckCircle,
  Error,
  Warning,
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import { fetchServices, controlService } from '../../services/api'

interface Service {
  id: string
  name: string
  displayName: string
  status: 'running' | 'stopped' | 'error' | 'starting' | 'stopping'
  health: 'healthy' | 'unhealthy' | 'unknown'
  port: number
  cpu: number
  memory: number
  uptime: string
  version: string
  autoStart: boolean
}

const ServiceGrid: React.FC = () => {
  const queryClient = useQueryClient()
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  const [selectedService, setSelectedService] = useState<Service | null>(null)

  const { data: services = [], isLoading } = useQuery<Service[]>({
    queryKey: ['services'],
    queryFn: fetchServices,
    refetchInterval: 3000,
  })

  const controlMutation = useMutation({
    mutationFn: ({ serviceId, action }: { serviceId: string; action: string }) =>
      controlService(serviceId, action),
    onSuccess: (_, { serviceId, action }) => {
      toast.success(`Service ${action} initiated`)
      queryClient.invalidateQueries({ queryKey: ['services'] })
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to control service')
    },
  })

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, service: Service) => {
    setAnchorEl(event.currentTarget)
    setSelectedService(service)
  }

  const handleMenuClose = () => {
    setAnchorEl(null)
    setSelectedService(null)
  }

  const handleServiceControl = (action: string) => {
    if (selectedService) {
      controlMutation.mutate({ serviceId: selectedService.id, action })
      handleMenuClose()
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'success'
      case 'stopped':
        return 'default'
      case 'error':
        return 'error'
      case 'starting':
      case 'stopping':
        return 'warning'
      default:
        return 'default'
    }
  }

  const getHealthIcon = (health: string) => {
    switch (health) {
      case 'healthy':
        return <CheckCircle color="success" />
      case 'unhealthy':
        return <Error color="error" />
      default:
        return <Warning color="warning" />
    }
  }

  const cardVariants = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: { opacity: 1, scale: 1 },
    hover: { scale: 1.02 },
  }

  if (isLoading) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Services
          </Typography>
          <LinearProgress />
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h5">
            Services
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {services.filter(s => s.status === 'running').length} of {services.length} running
          </Typography>
        </Box>

        <Grid container spacing={2}>
          <AnimatePresence>
            {services.map((service) => (
              <Grid item xs={12} sm={6} md={4} key={service.id}>
                <motion.div
                  variants={cardVariants}
                  initial="hidden"
                  animate="visible"
                  exit="hidden"
                  whileHover="hover"
                  layout
                >
                  <Card
                    sx={{
                      height: '100%',
                      position: 'relative',
                      overflow: 'visible',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        height: '4px',
                        background: service.status === 'running' 
                          ? 'linear-gradient(90deg, #00ff88 0%, #00cc66 100%)'
                          : service.status === 'error'
                          ? 'linear-gradient(90deg, #ff3366 0%, #cc1144 100%)'
                          : 'rgba(255, 255, 255, 0.1)',
                      },
                    }}
                  >
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                        <Box>
                          <Typography variant="h6" component="div">
                            {service.displayName}
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                            <Chip
                              label={service.status}
                              size="small"
                              color={getStatusColor(service.status) as any}
                            />
                            <Tooltip title={`Health: ${service.health}`}>
                              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                {getHealthIcon(service.health)}
                              </Box>
                            </Tooltip>
                          </Box>
                        </Box>
                        <IconButton
                          size="small"
                          onClick={(e) => handleMenuOpen(e, service)}
                        >
                          <MoreVert />
                        </IconButton>
                      </Box>

                      <Box sx={{ mb: 2 }}>
                        <Typography variant="body2" color="text.secondary">
                          Port: {service.port}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Version: {service.version}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Uptime: {service.uptime}
                        </Typography>
                      </Box>

                      <Box sx={{ mb: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                          <Typography variant="caption" color="text.secondary">
                            CPU
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {service.cpu}%
                          </Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={service.cpu}
                          sx={{
                            height: 4,
                            borderRadius: 2,
                            backgroundColor: 'rgba(255, 255, 255, 0.1)',
                            '& .MuiLinearProgress-bar': {
                              borderRadius: 2,
                              background: service.cpu > 80
                                ? 'linear-gradient(90deg, #ff3366 0%, #cc1144 100%)'
                                : service.cpu > 60
                                ? 'linear-gradient(90deg, #ffaa00 0%, #cc8800 100%)'
                                : 'linear-gradient(90deg, #00ff88 0%, #00cc66 100%)',
                            },
                          }}
                        />
                      </Box>

                      <Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                          <Typography variant="caption" color="text.secondary">
                            Memory
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {service.memory}%
                          </Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={service.memory}
                          sx={{
                            height: 4,
                            borderRadius: 2,
                            backgroundColor: 'rgba(255, 255, 255, 0.1)',
                            '& .MuiLinearProgress-bar': {
                              borderRadius: 2,
                              background: service.memory > 80
                                ? 'linear-gradient(90deg, #ff3366 0%, #cc1144 100%)'
                                : service.memory > 60
                                ? 'linear-gradient(90deg, #ffaa00 0%, #cc8800 100%)'
                                : 'linear-gradient(90deg, #00ff88 0%, #00cc66 100%)',
                            },
                          }}
                        />
                      </Box>
                    </CardContent>
                  </Card>
                </motion.div>
              </Grid>
            ))}
          </AnimatePresence>
        </Grid>

        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={handleMenuClose}
          PaperProps={{
            sx: {
              minWidth: 200,
            },
          }}
        >
          {selectedService?.status === 'running' ? (
            <MenuItem onClick={() => handleServiceControl('stop')}>
              <ListItemIcon>
                <Stop fontSize="small" />
              </ListItemIcon>
              <ListItemText>Stop</ListItemText>
            </MenuItem>
          ) : (
            <MenuItem onClick={() => handleServiceControl('start')}>
              <ListItemIcon>
                <PlayArrow fontSize="small" />
              </ListItemIcon>
              <ListItemText>Start</ListItemText>
            </MenuItem>
          )}
          
          <MenuItem onClick={() => handleServiceControl('restart')}>
            <ListItemIcon>
              <Refresh fontSize="small" />
            </ListItemIcon>
            <ListItemText>Restart</ListItemText>
          </MenuItem>

          <MenuItem onClick={handleMenuClose}>
            <ListItemIcon>
              <Settings fontSize="small" />
            </ListItemIcon>
            <ListItemText>Configure</ListItemText>
          </MenuItem>

          <MenuItem onClick={handleMenuClose}>
            <ListItemIcon>
              <Terminal fontSize="small" />
            </ListItemIcon>
            <ListItemText>Logs</ListItemText>
          </MenuItem>

          <MenuItem onClick={handleMenuClose}>
            <ListItemIcon>
              <Info fontSize="small" />
            </ListItemIcon>
            <ListItemText>Details</ListItemText>
          </MenuItem>

          <MenuItem
            onClick={() => {
              if (selectedService) {
                const newAutoStart = !selectedService.autoStart
                // Update auto-start setting
                toast.success(`Auto-start ${newAutoStart ? 'enabled' : 'disabled'}`)
                handleMenuClose()
              }
            }}
          >
            <ListItemIcon>
              <Switch
                checked={selectedService?.autoStart || false}
                size="small"
              />
            </ListItemIcon>
            <ListItemText>Auto-start</ListItemText>
          </MenuItem>
        </Menu>
      </CardContent>
    </Card>
  )
}

export default ServiceGrid