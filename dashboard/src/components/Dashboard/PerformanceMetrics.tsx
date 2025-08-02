import React from 'react'
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
} from '@mui/material'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { useTheme } from '@mui/material/styles'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

const PerformanceMetrics: React.FC = () => {
  const theme = useTheme()

  const generateTimeLabels = () => {
    const labels = []
    for (let i = 59; i >= 0; i--) {
      labels.push(`${i}m`)
    }
    return labels
  }

  const generateRandomData = (min: number, max: number, count: number) => {
    return Array.from({ length: count }, () => 
      Math.floor(Math.random() * (max - min + 1)) + min
    )
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        grid: {
          display: false,
        },
        ticks: {
          color: theme.palette.text.secondary,
          maxTicksLimit: 10,
        },
      },
      y: {
        grid: {
          color: 'rgba(255, 255, 255, 0.05)',
        },
        ticks: {
          color: theme.palette.text.secondary,
        },
      },
    },
  }

  const cpuData = {
    labels: generateTimeLabels(),
    datasets: [
      {
        label: 'CPU Usage',
        data: generateRandomData(30, 70, 60),
        borderColor: theme.palette.primary.main,
        backgroundColor: theme.palette.primary.main + '20',
        fill: true,
        tension: 0.4,
      },
    ],
  }

  const memoryData = {
    labels: generateTimeLabels(),
    datasets: [
      {
        label: 'Memory Usage',
        data: generateRandomData(40, 60, 60),
        borderColor: theme.palette.secondary.main,
        backgroundColor: theme.palette.secondary.main + '20',
        fill: true,
        tension: 0.4,
      },
    ],
  }

  const networkData = {
    labels: generateTimeLabels(),
    datasets: [
      {
        label: 'Download',
        data: generateRandomData(50, 150, 60),
        borderColor: theme.palette.info.main,
        backgroundColor: theme.palette.info.main + '20',
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Upload',
        data: generateRandomData(10, 50, 60),
        borderColor: theme.palette.warning.main,
        backgroundColor: theme.palette.warning.main + '20',
        fill: true,
        tension: 0.4,
      },
    ],
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Performance Metrics
        </Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                CPU Usage
              </Typography>
              <Box sx={{ height: 200 }}>
                <Line data={cpuData} options={chartOptions} />
              </Box>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Memory Usage
              </Typography>
              <Box sx={{ height: 200 }}>
                <Line data={memoryData} options={chartOptions} />
              </Box>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Network I/O
              </Typography>
              <Box sx={{ height: 200 }}>
                <Line data={networkData} options={{
                  ...chartOptions,
                  plugins: {
                    ...chartOptions.plugins,
                    legend: {
                      display: true,
                      position: 'bottom' as const,
                      labels: {
                        color: theme.palette.text.secondary,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        padding: 20,
                      },
                    },
                  },
                }} />
              </Box>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  )
}

export default PerformanceMetrics