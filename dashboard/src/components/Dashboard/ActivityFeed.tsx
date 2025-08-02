import React from 'react'
import {
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Avatar,
  Chip,
  Box,
  IconButton,
  Tooltip,
} from '@mui/material'
import {
  PlayCircle,
  StopCircle,
  Download,
  Upload,
  Error,
  CheckCircle,
  Info,
  Warning,
  FilterList,
  Refresh,
} from '@mui/icons-material'
import { format } from 'date-fns'
import { motion, AnimatePresence } from 'framer-motion'

interface ActivityItem {
  id: string
  type: 'service' | 'media' | 'system' | 'user'
  action: string
  description: string
  timestamp: Date
  severity: 'info' | 'success' | 'warning' | 'error'
  service?: string
}

const mockActivities: ActivityItem[] = [
  {
    id: '1',
    type: 'service',
    action: 'Started',
    description: 'Jellyfin service started successfully',
    timestamp: new Date(Date.now() - 2 * 60 * 1000),
    severity: 'success',
    service: 'Jellyfin',
  },
  {
    id: '2',
    type: 'media',
    action: 'Added',
    description: 'New movie added: Interstellar (2014)',
    timestamp: new Date(Date.now() - 15 * 60 * 1000),
    severity: 'info',
  },
  {
    id: '3',
    type: 'system',
    action: 'Optimized',
    description: 'Database optimization completed',
    timestamp: new Date(Date.now() - 30 * 60 * 1000),
    severity: 'success',
  },
  {
    id: '4',
    type: 'user',
    action: 'Login',
    description: 'Admin logged in from 192.168.1.100',
    timestamp: new Date(Date.now() - 45 * 60 * 1000),
    severity: 'info',
  },
  {
    id: '5',
    type: 'service',
    action: 'Error',
    description: 'Radarr failed to connect to download client',
    timestamp: new Date(Date.now() - 60 * 60 * 1000),
    severity: 'error',
    service: 'Radarr',
  },
  {
    id: '6',
    type: 'media',
    action: 'Downloaded',
    description: '3 episodes of "The Last of Us" completed',
    timestamp: new Date(Date.now() - 90 * 60 * 1000),
    severity: 'success',
  },
]

const ActivityFeed: React.FC = () => {
  const getIcon = (activity: ActivityItem) => {
    switch (activity.severity) {
      case 'success':
        return <CheckCircle />
      case 'error':
        return <Error />
      case 'warning':
        return <Warning />
      default:
        return <Info />
    }
  }

  const getColor = (severity: string) => {
    switch (severity) {
      case 'success':
        return 'success'
      case 'error':
        return 'error'
      case 'warning':
        return 'warning'
      default:
        return 'info'
    }
  }

  const getActionIcon = (action: string) => {
    switch (action.toLowerCase()) {
      case 'started':
        return <PlayCircle />
      case 'stopped':
        return <StopCircle />
      case 'downloaded':
        return <Download />
      case 'uploaded':
        return <Upload />
      default:
        return null
    }
  }

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Activity Feed
          </Typography>
          <Box>
            <Tooltip title="Filter">
              <IconButton size="small">
                <FilterList />
              </IconButton>
            </Tooltip>
            <Tooltip title="Refresh">
              <IconButton size="small">
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        <List sx={{ flex: 1, overflow: 'auto', maxHeight: 400 }}>
          <AnimatePresence>
            {mockActivities.map((activity, index) => (
              <motion.div
                key={activity.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ delay: index * 0.05 }}
              >
                <ListItem
                  alignItems="flex-start"
                  sx={{
                    borderRadius: 2,
                    mb: 1,
                    backgroundColor: 'rgba(255, 255, 255, 0.02)',
                    '&:hover': {
                      backgroundColor: 'rgba(255, 255, 255, 0.05)',
                    },
                  }}
                >
                  <ListItemAvatar>
                    <Avatar
                      sx={{
                        bgcolor: `${getColor(activity.severity)}.main`,
                        width: 36,
                        height: 36,
                      }}
                    >
                      {getActionIcon(activity.action) || getIcon(activity)}
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="body2" component="span">
                          {activity.description}
                        </Typography>
                        {activity.service && (
                          <Chip
                            label={activity.service}
                            size="small"
                            variant="outlined"
                            sx={{ height: 20, fontSize: '0.7rem' }}
                          />
                        )}
                      </Box>
                    }
                    secondary={
                      <Typography variant="caption" color="text.secondary">
                        {format(activity.timestamp, 'HH:mm:ss')} • {format(activity.timestamp, 'MMM d')}
                      </Typography>
                    }
                  />
                </ListItem>
              </motion.div>
            ))}
          </AnimatePresence>
        </List>
      </CardContent>
    </Card>
  )
}

export default ActivityFeed