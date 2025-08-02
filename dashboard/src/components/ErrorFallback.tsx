import React from 'react'
import { Box, Typography, Button, Paper } from '@mui/material'
import { Error as ErrorIcon, Refresh } from '@mui/icons-material'

interface ErrorFallbackProps {
  error: Error
  resetErrorBoundary: () => void
}

const ErrorFallback: React.FC<ErrorFallbackProps> = ({ error, resetErrorBoundary }) => {
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        backgroundColor: 'background.default',
        p: 3,
      }}
    >
      <Paper
        sx={{
          p: 4,
          maxWidth: 600,
          textAlign: 'center',
          backgroundColor: 'rgba(255, 255, 255, 0.05)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <ErrorIcon sx={{ fontSize: 80, color: 'error.main', mb: 2 }} />
        <Typography variant="h4" gutterBottom>
          Oops! Something went wrong
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          We encountered an unexpected error. Don't worry, your data is safe.
        </Typography>
        <Paper
          sx={{
            p: 2,
            mt: 2,
            mb: 3,
            backgroundColor: 'rgba(255, 51, 102, 0.1)',
            border: '1px solid rgba(255, 51, 102, 0.3)',
          }}
        >
          <Typography variant="body2" sx={{ fontFamily: 'monospace', textAlign: 'left' }}>
            {error.message}
          </Typography>
        </Paper>
        <Button
          variant="contained"
          color="primary"
          startIcon={<Refresh />}
          onClick={resetErrorBoundary}
          size="large"
        >
          Try Again
        </Button>
      </Paper>
    </Box>
  )
}

export default ErrorFallback