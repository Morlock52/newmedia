import { createSlice, PayloadAction } from '@reduxjs/toolkit'

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

interface ServicesState {
  services: Service[]
  loading: boolean
  error: string | null
}

const initialState: ServicesState = {
  services: [],
  loading: false,
  error: null,
}

const servicesSlice = createSlice({
  name: 'services',
  initialState,
  reducers: {
    setServices: (state, action: PayloadAction<Service[]>) => {
      state.services = action.payload
      state.loading = false
      state.error = null
    },
    updateService: (state, action: PayloadAction<Service>) => {
      const index = state.services.findIndex(s => s.id === action.payload.id)
      if (index !== -1) {
        state.services[index] = action.payload
      }
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload
      state.loading = false
    },
  },
})

export const { setServices, updateService, setLoading, setError } = servicesSlice.actions
export default servicesSlice.reducer