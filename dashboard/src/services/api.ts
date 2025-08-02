import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for auth
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('auth_token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// System Status
export const fetchSystemStatus = async () => {
  const { data } = await api.get('/system/status')
  return data
}

// Services
export const fetchServices = async () => {
  const { data } = await api.get('/services')
  return data
}

export const controlService = async (serviceId: string, action: 'start' | 'stop' | 'restart') => {
  const { data } = await api.post(`/services/${serviceId}/control`, { action })
  return data
}

export const fetchServiceLogs = async (serviceId: string, lines = 100) => {
  const { data } = await api.get(`/services/${serviceId}/logs`, { params: { lines } })
  return data
}

// Environment Variables
export const fetchEnvVariables = async () => {
  const { data } = await api.get('/env')
  return data
}

export const updateEnvVariables = async (variables: any[]) => {
  const { data } = await api.put('/env', { variables })
  return data
}

export const validateEnvConfig = async (variables: any[]) => {
  const { data } = await api.post('/env/validate', { variables })
  return data
}

// Media Library
export const fetchMediaStats = async () => {
  const { data } = await api.get('/media/stats')
  return data
}

export const fetchRecentMedia = async (type?: string) => {
  const { data } = await api.get('/media/recent', { params: { type } })
  return data
}

export const searchMedia = async (query: string, type?: string) => {
  const { data } = await api.get('/media/search', { params: { q: query, type } })
  return data
}

// Analytics
export const fetchAnalytics = async (period = '7d') => {
  const { data } = await api.get('/analytics', { params: { period } })
  return data
}

export const fetchPerformanceMetrics = async () => {
  const { data } = await api.get('/analytics/performance')
  return data
}

// Settings
export const fetchSettings = async () => {
  const { data } = await api.get('/settings')
  return data
}

export const updateSettings = async (settings: any) => {
  const { data } = await api.put('/settings', settings)
  return data
}

// Notifications
export const fetchNotifications = async () => {
  const { data } = await api.get('/notifications')
  return data
}

export const markNotificationRead = async (id: string) => {
  const { data } = await api.put(`/notifications/${id}/read`)
  return data
}

// WebSocket connection for real-time updates
export const createWebSocketConnection = (onMessage: (data: any) => void) => {
  const wsUrl = API_BASE_URL.replace(/^http/, 'ws') + '/ws'
  const ws = new WebSocket(wsUrl)

  ws.onopen = () => {
    console.log('WebSocket connected')
  }

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error)
    }
  }

  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
  }

  ws.onclose = () => {
    console.log('WebSocket disconnected')
    // Attempt to reconnect after 5 seconds
    setTimeout(() => createWebSocketConnection(onMessage), 5000)
  }

  return ws
}

export default api