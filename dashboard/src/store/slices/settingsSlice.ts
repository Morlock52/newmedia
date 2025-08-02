import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface Settings {
  general: {
    language: string
    timezone: string
    dateFormat: string
    theme: 'light' | 'dark' | 'auto'
  }
  notifications: {
    enabled: boolean
    email: boolean
    push: boolean
    types: {
      serviceStatus: boolean
      mediaAdded: boolean
      downloadComplete: boolean
      errors: boolean
    }
  }
  performance: {
    enableAnimations: boolean
    cacheSize: number
    refreshInterval: number
    logRetention: number
  }
  security: {
    twoFactorEnabled: boolean
    sessionTimeout: number
    apiAccess: boolean
  }
}

interface SettingsState {
  settings: Settings
  loading: boolean
  error: string | null
  unsavedChanges: boolean
}

const initialState: SettingsState = {
  settings: {
    general: {
      language: 'en',
      timezone: 'America/New_York',
      dateFormat: 'MM/DD/YYYY',
      theme: 'dark',
    },
    notifications: {
      enabled: true,
      email: true,
      push: false,
      types: {
        serviceStatus: true,
        mediaAdded: true,
        downloadComplete: true,
        errors: true,
      },
    },
    performance: {
      enableAnimations: true,
      cacheSize: 100,
      refreshInterval: 5000,
      logRetention: 7,
    },
    security: {
      twoFactorEnabled: false,
      sessionTimeout: 30,
      apiAccess: true,
    },
  },
  loading: false,
  error: null,
  unsavedChanges: false,
}

const settingsSlice = createSlice({
  name: 'settings',
  initialState,
  reducers: {
    setSettings: (state, action: PayloadAction<Settings>) => {
      state.settings = action.payload
      state.loading = false
      state.error = null
      state.unsavedChanges = false
    },
    updateSettings: (state, action: PayloadAction<Partial<Settings>>) => {
      state.settings = { ...state.settings, ...action.payload }
      state.unsavedChanges = true
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload
      state.loading = false
    },
    resetChanges: (state) => {
      state.unsavedChanges = false
    },
  },
})

export const { setSettings, updateSettings, setLoading, setError, resetChanges } = settingsSlice.actions
export default settingsSlice.reducer