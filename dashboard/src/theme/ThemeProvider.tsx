import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { cyberpunkTheme, CyberpunkTheme, ThemeContextType } from './CyberpunkTheme';
import './CyberpunkTheme.css';

// Create Theme Context
const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// Theme Provider Component
interface ThemeProviderProps {
  children: ReactNode;
  customTheme?: Partial<CyberpunkTheme>;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ 
  children, 
  customTheme 
}) => {
  const [theme, setTheme] = useState<CyberpunkTheme>(() => ({
    ...cyberpunkTheme,
    ...customTheme
  }));
  
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [accentColor, setAccentColor] = useState(cyberpunkTheme.colors.primary.neon);
  const [animations, setAnimations] = useState(true);
  const [soundEffects, setSoundEffects] = useState(true);
  const [glitchEffects, setGlitchEffects] = useState(true);

  // Apply theme to document root
  useEffect(() => {
    applyThemeToDOM();
    loadUserPreferences();
    initializeAudioContext();
  }, []);

  // Apply theme changes
  useEffect(() => {
    updateCSSVariables();
  }, [theme, accentColor, isDarkMode]);

  const applyThemeToDOM = () => {
    // Add cyberpunk class to body
    document.body.classList.add('cyberpunk-theme');
    
    // Set meta theme color for mobile browsers
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
      metaThemeColor.setAttribute('content', theme.colors.background.dark);
    } else {
      const meta = document.createElement('meta');
      meta.name = 'theme-color';
      meta.content = theme.colors.background.dark;
      document.head.appendChild(meta);
    }
  };

  const updateCSSVariables = () => {
    const root = document.documentElement;
    
    // Update accent color
    root.style.setProperty('--accent-color', accentColor);
    
    // Update dark mode
    if (!isDarkMode) {
      // Light mode adjustments (inverted cyberpunk)
      root.style.setProperty('--color-bg-dark', '#f0f0ff');
      root.style.setProperty('--color-bg-darker', '#e0e0f0');
      root.style.setProperty('--color-text-primary', '#000000');
    } else {
      // Reset to dark mode
      root.style.setProperty('--color-bg-dark', theme.colors.background.dark);
      root.style.setProperty('--color-bg-darker', theme.colors.background.darker);
      root.style.setProperty('--color-text-primary', theme.colors.text.primary);
    }
    
    // Toggle animations
    root.style.setProperty('--enable-animations', animations ? '1' : '0');
  };

  const loadUserPreferences = () => {
    try {
      const savedPrefs = localStorage.getItem('cyberpunk-theme-prefs');
      if (savedPrefs) {
        const prefs = JSON.parse(savedPrefs);
        setIsDarkMode(prefs.isDarkMode ?? true);
        setAccentColor(prefs.accentColor ?? cyberpunkTheme.colors.primary.neon);
        setAnimations(prefs.animations ?? true);
        setSoundEffects(prefs.soundEffects ?? true);
        setGlitchEffects(prefs.glitchEffects ?? true);
      }
    } catch (error) {
      console.error('Failed to load theme preferences:', error);
    }
  };

  const saveUserPreferences = () => {
    try {
      const prefs = {
        isDarkMode,
        accentColor,
        animations,
        soundEffects,
        glitchEffects
      };
      localStorage.setItem('cyberpunk-theme-prefs', JSON.stringify(prefs));
    } catch (error) {
      console.error('Failed to save theme preferences:', error);
    }
  };

  const initializeAudioContext = () => {
    if (!soundEffects) return;
    
    // Create audio context for UI sound effects
    const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
    if (AudioContext) {
      const audioContext = new AudioContext();
      (window as any).cyberAudioContext = audioContext;
    }
  };

  const playSound = (type: 'click' | 'hover' | 'success' | 'error' | 'notification') => {
    if (!soundEffects || !(window as any).cyberAudioContext) return;
    
    const audioContext = (window as any).cyberAudioContext;
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    // Different sound patterns for different actions
    switch (type) {
      case 'click':
        oscillator.frequency.value = 800;
        oscillator.type = 'square';
        gainNode.gain.value = 0.1;
        break;
      case 'hover':
        oscillator.frequency.value = 600;
        oscillator.type = 'sine';
        gainNode.gain.value = 0.05;
        break;
      case 'success':
        oscillator.frequency.value = 1000;
        oscillator.type = 'sine';
        gainNode.gain.value = 0.15;
        break;
      case 'error':
        oscillator.frequency.value = 200;
        oscillator.type = 'sawtooth';
        gainNode.gain.value = 0.2;
        break;
      case 'notification':
        oscillator.frequency.value = 1200;
        oscillator.type = 'triangle';
        gainNode.gain.value = 0.1;
        break;
    }
    
    oscillator.start();
    gainNode.gain.exponentialRampToValueAtTime(0.00001, audioContext.currentTime + 0.1);
    oscillator.stop(audioContext.currentTime + 0.1);
  };

  const toggleDarkMode = () => {
    setIsDarkMode(prev => {
      const newValue = !prev;
      saveUserPreferences();
      return newValue;
    });
  };

  const updateAccentColor = (color: string) => {
    setAccentColor(color);
    saveUserPreferences();
  };

  const contextValue: ThemeContextType = {
    theme,
    isDarkMode,
    toggleDarkMode,
    accentColor,
    setAccentColor: updateAccentColor
  };

  // Enhanced context with additional features
  const enhancedContext = {
    ...contextValue,
    animations,
    setAnimations,
    soundEffects,
    setSoundEffects,
    glitchEffects,
    setGlitchEffects,
    playSound
  };

  return (
    <ThemeContext.Provider value={enhancedContext as any}>
      <div className={`theme-wrapper ${isDarkMode ? 'dark-mode' : 'light-mode'}`}>
        {glitchEffects && <div className="glitch-overlay" />}
        {children}
      </div>
    </ThemeContext.Provider>
  );
};

// Custom hook to use theme
export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

// Additional theme utilities
export const useThemeColors = () => {
  const { theme } = useTheme();
  return theme.colors;
};

export const useThemeAnimations = () => {
  const { theme } = useTheme();
  return theme.animations;
};

export const useThemeEffects = () => {
  const { theme } = useTheme();
  return theme.effects;
};

// Styled component helper
export const styled = (Component: React.ComponentType<any>) => {
  return React.forwardRef((props: any, ref) => {
    const theme = useTheme();
    return <Component {...props} theme={theme} ref={ref} />;
  });
};

export default ThemeProvider;