import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from './contexts/ThemeContext';
import { AuthProvider } from './contexts/AuthContext';
import { NotificationProvider } from './contexts/NotificationContext';

// Pages
import Dashboard from './pages/Dashboard';
import Setup from './pages/Setup';
import Media from './pages/Media';
import Apps from './pages/Apps';
import Settings from './pages/Settings';
import AI from './pages/AI';
import Users from './pages/Users';
import Stats from './pages/Stats';

// Components
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';
import LoadingScreen from './components/LoadingScreen';
import ErrorBoundary from './components/ErrorBoundary';

// Styles
import './styles/globals.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

function App() {
  const [isInitialized, setIsInitialized] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    checkInitialization();
  }, []);

  const checkInitialization = async () => {
    try {
      const response = await fetch('/api/system/status');
      const data = await response.json();
      setIsInitialized(data.initialized);
    } catch (error) {
      console.error('Failed to check initialization:', error);
      setIsInitialized(false);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return <LoadingScreen />;
  }

  if (!isInitialized) {
    return (
      <ThemeProvider>
        <Setup onComplete={() => setIsInitialized(true)} />
      </ThemeProvider>
    );
  }

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider>
          <AuthProvider>
            <NotificationProvider>
              <Router>
                <Routes>
                  <Route path="/" element={<Layout />}>
                    <Route index element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
                    <Route path="media/*" element={<ProtectedRoute><Media /></ProtectedRoute>} />
                    <Route path="apps/*" element={<ProtectedRoute><Apps /></ProtectedRoute>} />
                    <Route path="ai/*" element={<ProtectedRoute><AI /></ProtectedRoute>} />
                    <Route path="users/*" element={<ProtectedRoute requireAdmin><Users /></ProtectedRoute>} />
                    <Route path="stats/*" element={<ProtectedRoute><Stats /></ProtectedRoute>} />
                    <Route path="settings/*" element={<ProtectedRoute requireAdmin><Settings /></ProtectedRoute>} />
                  </Route>
                </Routes>
              </Router>
            </NotificationProvider>
          </AuthProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;