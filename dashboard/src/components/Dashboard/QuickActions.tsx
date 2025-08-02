import React from 'react'
import {
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Stack,
} from '@mui/material'
import {
  PlayArrow,
  Stop,
  Refresh,
  CloudUpload,
  Scanner,
  Build,
  Speed,
  BackupTable,
} from '@mui/icons-material'
import { motion } from 'framer-motion'
import toast from 'react-hot-toast'

const QuickActions: React.FC = () => {
  const actions = [
    {
      label: 'Start All Services',
      icon: <PlayArrow />,
      color: 'success',
      action: () => toast.success('Starting all services...'),
    },
    {
      label: 'Stop All Services',
      icon: <Stop />,
      color: 'error',
      action: () => toast.error('Stopping all services...'),
    },
    {
      label: 'Restart Stack',
      icon: <Refresh />,
      color: 'warning',
      action: () => toast.info('Restarting stack...'),
    },
    {
      label: 'Scan Media',
      icon: <Scanner />,
      color: 'info',
      action: () => toast.success('Scanning media library...'),
    },
    {
      label: 'Optimize Performance',
      icon: <Speed />,
      color: 'primary',
      action: () => toast.success('Optimizing performance...'),
    },
    {
      label: 'Backup Now',
      icon: <BackupTable />,
      color: 'secondary',
      action: () => toast.success('Starting backup...'),
    },
  ]

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Quick Actions
        </Typography>
        <Stack spacing={1}>
          {actions.map((action, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Button
                fullWidth
                variant="outlined"
                color={action.color as any}
                startIcon={action.icon}
                onClick={action.action}
                sx={{
                  justifyContent: 'flex-start',
                  textTransform: 'none',
                  borderRadius: 2,
                  py: 1,
                  '&:hover': {
                    transform: 'translateX(4px)',
                  },
                }}
              >
                {action.label}
              </Button>
            </motion.div>
          ))}
        </Stack>
      </CardContent>
    </Card>
  )
}

export default QuickActions