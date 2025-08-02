import React, { Suspense, lazy } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { ThemeProvider, CssBaseline } from '@mui/material'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { Provider } from 'react-redux'
import { Toaster } from 'react-hot-toast'
import { ErrorBoundary } from 'react-error-boundary'
import { store } from './store'
import { theme } from './theme'
import Layout from './components/Layout'
import LoadingScreen from './components/LoadingScreen'
import ErrorFallback from './components/ErrorFallback'
import './styles/global.css'

// Lazy load pages for better performance
const Dashboard = lazy(() => import('./pages/Dashboard'))
const MediaLibrary = lazy(() => import('./pages/MediaLibrary'))
const ServiceControl = lazy(() => import('./pages/ServiceControl'))
const EnvManager = lazy(() => import('./pages/EnvManager'))
const Analytics = lazy(() => import('./pages/Analytics'))
const Settings = lazy(() => import('./pages/Settings'))
const Tutorial = lazy(() => import('./pages/Tutorial'))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes
      retry: 3,
      refetchOnWindowFocus: false,
    },
  },
})

function App() {
  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <Provider store={store}>
        <QueryClientProvider client={queryClient}>
          <ThemeProvider theme={theme}>
            <CssBaseline />
            <Router>
              <Layout>
                <Suspense fallback={<LoadingScreen />}>
                  <Routes>
                    <Route path="/" element={<Dashboard />} />
                    <Route path="/media" element={<MediaLibrary />} />
                    <Route path="/services" element={<ServiceControl />} />
                    <Route path="/environment" element={<EnvManager />} />
                    <Route path="/analytics" element={<Analytics />} />
                    <Route path="/settings" element={<Settings />} />
                    <Route path="/tutorial" element={<Tutorial />} />
                  </Routes>
                </Suspense>
              </Layout>
            </Router>
            <Toaster
              position="bottom-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#1a1a1a',
                  color: '#fff',
                  border: '1px solid #333',
                },
                success: {
                  iconTheme: {
                    primary: '#00ff88',
                    secondary: '#000',
                  },
                },
                error: {
                  iconTheme: {
                    primary: '#ff3366',
                    secondary: '#000',
                  },
                },
              }}
            />
            <ReactQueryDevtools initialIsOpen={false} />
          </ThemeProvider>
        </QueryClientProvider>
      </Provider>
    </ErrorBoundary>
  )
}

export default App