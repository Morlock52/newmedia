import { createTheme } from '@mui/material/styles'

export const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ff88',
      light: '#33ffaa',
      dark: '#00cc66',
      contrastText: '#000',
    },
    secondary: {
      main: '#0088ff',
      light: '#33aaff',
      dark: '#0066cc',
      contrastText: '#fff',
    },
    error: {
      main: '#ff3366',
      light: '#ff5588',
      dark: '#cc1144',
    },
    warning: {
      main: '#ffaa00',
      light: '#ffbb33',
      dark: '#cc8800',
    },
    info: {
      main: '#00ffff',
      light: '#33ffff',
      dark: '#00cccc',
    },
    success: {
      main: '#00ff88',
      light: '#33ffaa',
      dark: '#00cc66',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0b0b0',
    },
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"SF Pro Display"',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {
      fontSize: '3rem',
      fontWeight: 700,
      lineHeight: 1.2,
    },
    h2: {
      fontSize: '2.5rem',
      fontWeight: 600,
      lineHeight: 1.3,
    },
    h3: {
      fontSize: '2rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
      lineHeight: 1.5,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 500,
      lineHeight: 1.6,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 500,
      lineHeight: 1.7,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 16,
  },
  shadows: [
    'none',
    '0 2px 4px rgba(0, 0, 0, 0.1)',
    '0 4px 8px rgba(0, 0, 0, 0.15)',
    '0 8px 16px rgba(0, 0, 0, 0.2)',
    '0 12px 24px rgba(0, 0, 0, 0.25)',
    '0 16px 32px rgba(0, 0, 0, 0.3)',
    '0 20px 40px rgba(0, 0, 0, 0.35)',
    '0 24px 48px rgba(0, 0, 0, 0.4)',
    '0 32px 64px rgba(0, 0, 0, 0.45)',
    '0 40px 80px rgba(0, 0, 0, 0.5)',
    '0 48px 96px rgba(0, 0, 0, 0.55)',
    '0 56px 112px rgba(0, 0, 0, 0.6)',
    '0 64px 128px rgba(0, 0, 0, 0.65)',
    '0 72px 144px rgba(0, 0, 0, 0.7)',
    '0 80px 160px rgba(0, 0, 0, 0.75)',
    '0 88px 176px rgba(0, 0, 0, 0.8)',
    '0 96px 192px rgba(0, 0, 0, 0.85)',
    '0 104px 208px rgba(0, 0, 0, 0.9)',
    '0 112px 224px rgba(0, 0, 0, 0.95)',
    '0 120px 240px rgba(0, 0, 0, 1)',
    '0 128px 256px rgba(0, 0, 0, 1)',
    '0 136px 272px rgba(0, 0, 0, 1)',
    '0 144px 288px rgba(0, 0, 0, 1)',
    '0 152px 304px rgba(0, 0, 0, 1)',
    '0 160px 320px rgba(0, 0, 0, 1)',
  ],
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          padding: '12px 24px',
          fontWeight: 600,
          textTransform: 'none',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: '-100%',
            width: '100%',
            height: '100%',
            background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent)',
            transition: 'left 0.5s ease',
          },
          '&:hover': {
            transform: 'translateY(-3px) scale(1.05)',
            boxShadow: '0 15px 35px rgba(0, 0, 0, 0.4)',
            '&::before': {
              left: '100%',
            },
          },
        },
        contained: {
          background: 'linear-gradient(135deg, #00ff88, #0088ff)',
          color: '#000',
          boxShadow: '0 4px 15px rgba(0, 255, 136, 0.3)',
          '&:hover': {
            background: 'linear-gradient(135deg, #33ffaa, #33aaff)',
            boxShadow: '0 15px 35px rgba(0, 255, 136, 0.4)',
          },
        },
        outlined: {
          border: '2px solid rgba(0, 255, 136, 0.5)',
          color: '#00ff88',
          '&:hover': {
            border: '2px solid #00ff88',
            backgroundColor: 'rgba(0, 255, 136, 0.1)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          background: 'rgba(255, 255, 255, 0.03)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: 16,
          transition: 'all 0.4s cubic-bezier(0.23, 1, 0.320, 1)',
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: -2,
            left: -2,
            right: -2,
            bottom: -2,
            background: 'linear-gradient(45deg, transparent, rgba(0, 255, 136, 0.4), transparent)',
            borderRadius: 'inherit',
            opacity: 0,
            transition: 'opacity 0.3s ease',
            zIndex: -1,
          },
          '&:hover': {
            transform: 'translateY(-8px) scale(1.02)',
            boxShadow: `
              0 25px 50px rgba(0, 0, 0, 0.4),
              0 0 0 1px rgba(0, 255, 136, 0.3),
              inset 0 1px 0 rgba(255, 255, 255, 0.1)
            `,
            '&::before': {
              opacity: 1,
            },
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: 'rgba(255, 255, 255, 0.05)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            '& fieldset': {
              borderColor: 'rgba(255, 255, 255, 0.2)',
            },
            '&:hover fieldset': {
              borderColor: 'rgba(255, 255, 255, 0.3)',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#00ff88',
            },
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(255, 255, 255, 0.1)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.15)',
          },
        },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: 'rgba(0, 0, 0, 0.9)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          fontSize: '0.875rem',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: 'rgba(10, 10, 10, 0.95)',
          backdropFilter: 'blur(20px)',
          borderRight: '1px solid rgba(255, 255, 255, 0.1)',
        },
      },
    },
  },
})