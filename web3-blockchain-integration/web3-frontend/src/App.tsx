import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { QueryClient, QueryClientProvider } from 'react-query';

// Web3 Imports
import { EthereumClient, w3mConnectors, w3mProvider } from '@web3modal/ethereum';
import { Web3Modal } from '@web3modal/react';
import { configureChains, createConfig, WagmiConfig } from 'wagmi';
import { arbitrum, mainnet, polygon, optimism } from 'wagmi/chains';

// Components
import Header from './components/Layout/Header';
import Sidebar from './components/Layout/Sidebar';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import MyContent from './pages/MyContent';
import Marketplace from './pages/Marketplace';
import DAO from './pages/DAO';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';

// Hooks & Utils
import { Web3Provider } from './contexts/Web3Context';
import { IPFSProvider } from './contexts/IPFSContext';
import { NotificationProvider } from './contexts/NotificationContext';

// Create theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#3b82f6',
      light: '#60a5fa',
      dark: '#1d4ed8',
    },
    secondary: {
      main: '#8b5cf6',
      light: '#a78bfa',
      dark: '#7c3aed',
    },
    background: {
      default: '#0f172a',
      paper: '#1e293b',
    },
    text: {
      primary: '#f8fafc',
      secondary: '#cbd5e1',
    },
  },
  typography: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
          fontWeight: 600,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          border: '1px solid rgba(255, 255, 255, 0.1)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
        },
      },
    },
  },
});

// Web3 Configuration
const chains = [mainnet, polygon, arbitrum, optimism];
const projectId = process.env.REACT_APP_WALLETCONNECT_PROJECT_ID || 'default-project-id';

const { publicClient } = configureChains(chains, [w3mProvider({ projectId })]);
const wagmiConfig = createConfig({
  autoConnect: true,
  connectors: w3mConnectors({ projectId, chains }),
  publicClient,
});

const ethereumClient = new EthereumClient(wagmiConfig, chains);

// React Query Client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 1000 * 60 * 5, // 5 minutes
    },
  },
});

function App() {
  const [sidebarOpen, setSidebarOpen] = React.useState(false);

  const handleSidebarToggle = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <QueryClientProvider client={queryClient}>
        <WagmiConfig config={wagmiConfig}>
          <NotificationProvider>
            <Web3Provider>
              <IPFSProvider>
                <Router>
                  <Box sx={{ display: 'flex', minHeight: '100vh' }}>
                    <Header onMenuClick={handleSidebarToggle} />
                    <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
                    
                    <Box
                      component="main"
                      sx={{
                        flexGrow: 1,
                        p: 3,
                        mt: 8, // Account for header height
                        ml: sidebarOpen ? { sm: '240px' } : 0,
                        transition: 'margin-left 0.3s ease',
                      }}
                    >
                      <Routes>
                        <Route path="/" element={<Dashboard />} />
                        <Route path="/upload" element={<Upload />} />
                        <Route path="/my-content" element={<MyContent />} />
                        <Route path="/marketplace" element={<Marketplace />} />
                        <Route path="/dao" element={<DAO />} />
                        <Route path="/analytics" element={<Analytics />} />
                        <Route path="/settings" element={<Settings />} />
                      </Routes>
                    </Box>
                  </Box>
                </Router>
              </IPFSProvider>
            </Web3Provider>
          </NotificationProvider>
          
          <Web3Modal
            projectId={projectId}
            ethereumClient={ethereumClient}
            themeMode="dark"
            themeVariables={{
              '--w3m-font-family': theme.typography.fontFamily,
              '--w3m-accent-color': theme.palette.primary.main,
            }}
          />
        </WagmiConfig>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

export default App;