import React from 'react'
import { Typography, Box } from '@mui/material'
import ServiceGrid from '../components/Dashboard/ServiceGrid'

const ServiceControl: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3 }}>Service Control</Typography>
      <ServiceGrid />
    </Box>
  )
}

export default ServiceControl