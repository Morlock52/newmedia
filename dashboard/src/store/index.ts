import { configureStore } from '@reduxjs/toolkit'
import servicesReducer from './slices/servicesSlice'
import mediaReducer from './slices/mediaSlice'
import settingsReducer from './slices/settingsSlice'
import notificationsReducer from './slices/notificationsSlice'

export const store = configureStore({
  reducer: {
    services: servicesReducer,
    media: mediaReducer,
    settings: settingsReducer,
    notifications: notificationsReducer,
  },
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch