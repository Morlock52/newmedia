import React, { useState, useEffect, createContext, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface ThemeColors {
  primary: string;
  secondary: string;
  accent: string;
  background: string;
  surface: string;
  text: string;
  success: string;
  warning: string;
  error: string;
  info: string;
}

interface CyberpunkThemeConfig {
  colors: ThemeColors;
  animations: {
    glitch: boolean;
    neon: boolean;
    matrix: boolean;
    scanlines: boolean;
  };
  effects: {
    particles: boolean;
    bloom: boolean;
    chromatic: boolean;
    noise: boolean;
  };
  sounds: {
    enabled: boolean;
    volume: number;
    hover: string;
    click: string;
    notification: string;
  };
}

interface CyberpunkContextType {
  theme: CyberpunkThemeConfig;
  setTheme: (theme: Partial<CyberpunkThemeConfig>) => void;
  presets: Record<string, CyberpunkThemeConfig>;
  applyPreset: (presetName: string) => void;
  playSound: (soundType: keyof CyberpunkThemeConfig['sounds']) => void;
}

const CyberpunkContext = createContext<CyberpunkContextType | null>(null);

const defaultTheme: CyberpunkThemeConfig = {
  colors: {
    primary: '#00FFFF',
    secondary: '#FF00FF',
    accent: '#FFFF00',
    background: '#0a0a0a',
    surface: '#1a1a2e',
    text: '#00FFFF',
    success: '#00FF00',
    warning: '#FFFF00',
    error: '#FF0040',
    info: '#00FFFF'
  },
  animations: {
    glitch: true,
    neon: true,
    matrix: false,
    scanlines: true
  },
  effects: {
    particles: true,
    bloom: true,
    chromatic: false,
    noise: true
  },
  sounds: {
    enabled: true,
    volume: 0.3,
    hover: '/sounds/hover.mp3',
    click: '/sounds/click.mp3',
    notification: '/sounds/notification.mp3'
  }
};

const themePresets: Record<string, CyberpunkThemeConfig> = {
  classic: {
    ...defaultTheme,
    colors: {
      ...defaultTheme.colors,
      primary: '#00FFFF',
      secondary: '#FF00FF',
      accent: '#FFFF00'
    }
  },
  neon: {
    ...defaultTheme,
    colors: {
      ...defaultTheme.colors,
      primary: '#FF6B35',
      secondary: '#F7931E',
      accent: '#FFD23F'
    },
    effects: {
      ...defaultTheme.effects,
      bloom: true,
      chromatic: true
    }
  },
  matrix: {
    ...defaultTheme,
    colors: {
      ...defaultTheme.colors,
      primary: '#00FF00',
      secondary: '#008F11',
      accent: '#00FF41',
      background: '#000000',
      text: '#00FF00'
    },
    animations: {
      ...defaultTheme.animations,
      matrix: true,
      glitch: false
    }
  },
  retro: {
    ...defaultTheme,
    colors: {
      ...defaultTheme.colors,
      primary: '#FF0080',
      secondary: '#8000FF',
      accent: '#00FFFF',
      background: '#1a0033',
      surface: '#330066'
    },
    effects: {
      ...defaultTheme.effects,
      bloom: true,
      particles: true
    }
  },
  minimal: {
    ...defaultTheme,
    colors: {
      ...defaultTheme.colors,
      primary: '#FFFFFF',
      secondary: '#CCCCCC',
      accent: '#FF0080',
      background: '#000000',
      text: '#FFFFFF'
    },
    animations: {
      glitch: false,
      neon: false,
      matrix: false,
      scanlines: false
    },
    effects: {
      particles: false,
      bloom: false,
      chromatic: false,
      noise: false
    }
  }
};

const CyberpunkTheme: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setThemeState] = useState<CyberpunkThemeConfig>(defaultTheme);
  const [audioContext, setAudioContext] = useState<AudioContext | null>(null);
  const [soundBuffers, setSoundBuffers] = useState<Record<string, AudioBuffer>>({});

  useEffect(() => {
    // Initialize audio context
    if (theme.sounds.enabled && !audioContext) {
      const context = new (window.AudioContext || (window as any).webkitAudioContext)();
      setAudioContext(context);
    }
    
    // Load sound files
    if (audioContext && theme.sounds.enabled) {
      loadSounds();
    }
    
    // Apply theme to document
    applyThemeToDocument();
    
    return () => {
      if (audioContext) {
        audioContext.close();
      }
    };
  }, [theme]);

  const loadSounds = async () => {
    if (!audioContext) return;
    
    const soundFiles = {
      hover: theme.sounds.hover,
      click: theme.sounds.click,
      notification: theme.sounds.notification
    };
    
    const buffers: Record<string, AudioBuffer> = {};
    
    for (const [key, url] of Object.entries(soundFiles)) {
      try {
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        buffers[key] = audioBuffer;
      } catch (error) {
        console.warn(`Failed to load sound: ${url}`);
      }
    }
    
    setSoundBuffers(buffers);
  };

  const applyThemeToDocument = () => {
    const root = document.documentElement;
    
    // Set CSS variables
    Object.entries(theme.colors).forEach(([key, value]) => {
      root.style.setProperty(`--color-${key}`, value);
    });
    
    // Apply body styles
    document.body.style.background = `linear-gradient(135deg, ${theme.colors.background} 0%, ${theme.colors.surface} 100%)`;
    document.body.style.color = theme.colors.text;
    document.body.style.fontFamily = 'Orbitron, monospace';
    
    // Add cyberpunk effects
    addCyberpunkEffects();
  };

  const addCyberpunkEffects = () => {
    // Remove existing effects
    const existingEffects = document.querySelectorAll('.cyberpunk-effect');
    existingEffects.forEach(effect => effect.remove());
    
    if (theme.effects.scanlines) {
      addScanlines();
    }
    
    if (theme.effects.particles) {
      addParticles();
    }
    
    if (theme.effects.noise) {
      addNoise();
    }
  };

  const addScanlines = () => {
    const scanlines = document.createElement('div');
    scanlines.className = 'cyberpunk-effect scanlines';
    scanlines.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 9999;
      background: linear-gradient(
        transparent 50%,
        rgba(0, 255, 255, 0.03) 50%
      );
      background-size: 100% 4px;
      animation: scanlines 0.1s linear infinite;
    `;
    
    const style = document.createElement('style');
    style.textContent = `
      @keyframes scanlines {
        0% { transform: translateY(0); }
        100% { transform: translateY(4px); }
      }
    `;
    
    document.head.appendChild(style);
    document.body.appendChild(scanlines);
  };

  const addParticles = () => {
    const particleContainer = document.createElement('div');
    particleContainer.className = 'cyberpunk-effect particles';
    particleContainer.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: -1;
      overflow: hidden;
    `;
    
    // Create floating particles
    for (let i = 0; i < 50; i++) {
      const particle = document.createElement('div');
      particle.style.cssText = `
        position: absolute;
        width: 2px;
        height: 2px;
        background: ${theme.colors.primary};
        border-radius: 50%;
        left: ${Math.random() * 100}%;
        top: ${Math.random() * 100}%;
        animation: float ${3 + Math.random() * 4}s linear infinite;
        opacity: ${0.3 + Math.random() * 0.7};
        box-shadow: 0 0 6px ${theme.colors.primary};
      `;
      particleContainer.appendChild(particle);
    }
    
    const style = document.createElement('style');
    style.textContent = `
      @keyframes float {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        33% { transform: translate(30px, -30px) rotate(120deg); }
        66% { transform: translate(-20px, 20px) rotate(240deg); }
      }
    `;
    
    document.head.appendChild(style);
    document.body.appendChild(particleContainer);
  };

  const addNoise = () => {
    const noise = document.createElement('div');
    noise.className = 'cyberpunk-effect noise';
    noise.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 9998;
      opacity: 0.02;
      background-image: 
        radial-gradient(circle, ${theme.colors.primary} 1px, transparent 1px);
      background-size: 15px 15px;
      animation: noise 0.2s infinite;
    `;
    
    const style = document.createElement('style');
    style.textContent = `
      @keyframes noise {
        0%, 100% { transform: translate(0, 0); }
        10% { transform: translate(-1px, 1px); }
        20% { transform: translate(1px, -1px); }
        30% { transform: translate(-1px, -1px); }
        40% { transform: translate(1px, 1px); }
        50% { transform: translate(-1px, 0); }
        60% { transform: translate(1px, 0); }
        70% { transform: translate(0, -1px); }
        80% { transform: translate(0, 1px); }
        90% { transform: translate(-1px, -1px); }
      }
    `;
    
    document.head.appendChild(style);
    document.body.appendChild(noise);
  };

  const setTheme = (newTheme: Partial<CyberpunkThemeConfig>) => {
    setThemeState(prev => ({ ...prev, ...newTheme }));
  };

  const applyPreset = (presetName: string) => {
    if (themePresets[presetName]) {
      setThemeState(themePresets[presetName]);
    }
  };

  const playSound = (soundType: keyof CyberpunkThemeConfig['sounds']) => {
    if (!theme.sounds.enabled || !audioContext || !soundBuffers[soundType]) return;
    
    try {
      const source = audioContext.createBufferSource();
      const gainNode = audioContext.createGain();
      
      source.buffer = soundBuffers[soundType];
      source.connect(gainNode);
      gainNode.connect(audioContext.destination);
      
      gainNode.gain.value = theme.sounds.volume;
      source.start();
    } catch (error) {
      console.warn('Failed to play sound:', error);
    }
  };

  const contextValue: CyberpunkContextType = {
    theme,
    setTheme,
    presets: themePresets,
    applyPreset,
    playSound
  };

  return (
    <CyberpunkContext.Provider value={contextValue}>
      {children}
      <CyberpunkStyleProvider />
    </CyberpunkContext.Provider>
  );
};

const CyberpunkStyleProvider: React.FC = () => {
  const context = useContext(CyberpunkContext);
  if (!context) return null;
  
  const { theme } = context;

  return (
    <>
      <style jsx global>{`
        :root {
          --color-primary: ${theme.colors.primary};
          --color-secondary: ${theme.colors.secondary};
          --color-accent: ${theme.colors.accent};
          --color-background: ${theme.colors.background};
          --color-surface: ${theme.colors.surface};
          --color-text: ${theme.colors.text};
          --color-success: ${theme.colors.success};
          --color-warning: ${theme.colors.warning};
          --color-error: ${theme.colors.error};
          --color-info: ${theme.colors.info};
        }
        
        * {
          box-sizing: border-box;
        }
        
        body {
          margin: 0;
          padding: 0;
          font-family: 'Orbitron', monospace;
          background: linear-gradient(135deg, var(--color-background) 0%, var(--color-surface) 100%);
          color: var(--color-text);
          overflow-x: hidden;
        }
        
        .cyberpunk-glow {
          box-shadow: 0 0 20px var(--color-primary);
          border: 2px solid var(--color-primary);
        }
        
        .cyberpunk-text-glow {
          text-shadow: 0 0 10px var(--color-primary);
        }
        
        .cyberpunk-border {
          border: 2px solid var(--color-primary);
          border-radius: 8px;
        }
        
        .cyberpunk-bg {
          background: rgba(0, 0, 0, 0.8);
          backdrop-filter: blur(10px);
        }
        
        ${theme.animations.glitch ? `
          .cyberpunk-glitch {
            animation: glitch 2s infinite;
          }
          
          @keyframes glitch {
            0%, 100% { 
              transform: translate(0);
              filter: hue-rotate(0deg);
            }
            10% { 
              transform: translate(-2px, 1px);
              filter: hue-rotate(90deg);
            }
            20% { 
              transform: translate(2px, -1px);
              filter: hue-rotate(180deg);
            }
            30% { 
              transform: translate(-1px, 2px);
              filter: hue-rotate(270deg);
            }
            40% { 
              transform: translate(1px, -2px);
              filter: hue-rotate(0deg);
            }
            50% { 
              transform: translate(-2px, -1px);
              filter: hue-rotate(90deg);
            }
            60% { 
              transform: translate(2px, 1px);
              filter: hue-rotate(180deg);
            }
            70% { 
              transform: translate(-1px, -2px);
              filter: hue-rotate(270deg);
            }
            80% { 
              transform: translate(1px, 2px);
              filter: hue-rotate(0deg);
            }
            90% { 
              transform: translate(-2px, 1px);
              filter: hue-rotate(90deg);
            }
          }
        ` : ''}
        
        ${theme.animations.neon ? `
          .cyberpunk-neon {
            animation: neonPulse 2s ease-in-out infinite alternate;
          }
          
          @keyframes neonPulse {
            from {
              box-shadow: 
                0 0 5px var(--color-primary),
                0 0 10px var(--color-primary),
                0 0 20px var(--color-primary);
            }
            to {
              box-shadow: 
                0 0 10px var(--color-primary),
                0 0 20px var(--color-primary),
                0 0 40px var(--color-primary),
                0 0 60px var(--color-primary);
            }
          }
        ` : ''}
        
        ${theme.animations.matrix ? `
          .cyberpunk-matrix {
            animation: matrixRain 10s linear infinite;
          }
          
          @keyframes matrixRain {
            0% { transform: translateY(-100vh); }
            100% { transform: translateY(100vh); }
          }
        ` : ''}
        
        .cyberpunk-button {
          background: linear-gradient(45deg, var(--color-primary), var(--color-secondary));
          border: none;
          padding: 12px 24px;
          color: #000;
          font-family: inherit;
          font-weight: bold;
          cursor: pointer;
          border-radius: 8px;
          transition: all 0.3s ease;
          text-transform: uppercase;
        }
        
        .cyberpunk-button:hover {
          transform: translateY(-2px);
          box-shadow: 0 10px 20px rgba(0, 255, 255, 0.3);
        }
        
        .cyberpunk-input {
          background: rgba(0, 0, 0, 0.8);
          border: 2px solid var(--color-primary);
          border-radius: 8px;
          padding: 12px;
          color: var(--color-text);
          font-family: inherit;
          font-size: 1rem;
        }
        
        .cyberpunk-input:focus {
          outline: none;
          box-shadow: 0 0 20px var(--color-primary);
        }
        
        .cyberpunk-card {
          background: rgba(0, 0, 0, 0.8);
          border: 2px solid var(--color-primary);
          border-radius: 12px;
          padding: 20px;
          backdrop-filter: blur(10px);
        }
        
        .cyberpunk-grid {
          background-image: 
            linear-gradient(var(--color-primary) 1px, transparent 1px),
            linear-gradient(90deg, var(--color-primary) 1px, transparent 1px);
          background-size: 20px 20px;
          opacity: 0.1;
        }
        
        .cyberpunk-loader {
          width: 40px;
          height: 40px;
          border: 4px solid transparent;
          border-top: 4px solid var(--color-primary);
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .cyberpunk-progress {
          width: 100%;
          height: 8px;
          background: rgba(0, 0, 0, 0.5);
          border-radius: 4px;
          overflow: hidden;
        }
        
        .cyberpunk-progress-bar {
          height: 100%;
          background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));
          transition: width 0.3s ease;
          box-shadow: 0 0 10px var(--color-primary);
        }
      `}</style>
    </>
  );
};

// Theme Customizer Component
const CyberpunkThemeCustomizer: React.FC = () => {
  const context = useContext(CyberpunkContext);
  if (!context) return null;
  
  const { theme, setTheme, presets, applyPreset } = context;
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          position: 'fixed',
          top: '20px',
          right: '20px',
          zIndex: 10000,
          background: 'linear-gradient(45deg, #00FFFF, #FF00FF)',
          border: 'none',
          borderRadius: '50%',
          width: '50px',
          height: '50px',
          cursor: 'pointer',
          fontSize: '1.5rem'
        }}
      >
        🎨
      </button>
      
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ x: 400, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 400, opacity: 0 }}
            style={{
              position: 'fixed',
              top: '80px',
              right: '20px',
              width: '350px',
              maxHeight: '70vh',
              overflowY: 'auto',
              background: 'rgba(0,0,0,0.95)',
              border: '2px solid #00FFFF',
              borderRadius: '12px',
              padding: '20px',
              zIndex: 9999,
              backdropFilter: 'blur(15px)'
            }}
          >
            <h3 style={{ color: '#00FFFF', marginBottom: '20px' }}>THEME CUSTOMIZER</h3>
            
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ color: '#FFFF00', marginBottom: '10px' }}>Presets</h4>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                {Object.keys(presets).map(presetName => (
                  <button
                    key={presetName}
                    onClick={() => applyPreset(presetName)}
                    className="cyberpunk-button"
                    style={{
                      padding: '8px 12px',
                      fontSize: '0.9rem',
                      textTransform: 'capitalize'
                    }}
                  >
                    {presetName}
                  </button>
                ))}
              </div>
            </div>
            
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ color: '#FFFF00', marginBottom: '10px' }}>Colors</h4>
              {Object.entries(theme.colors).map(([key, value]) => (
                <div key={key} style={{ marginBottom: '10px' }}>
                  <label style={{ display: 'block', marginBottom: '5px', textTransform: 'capitalize' }}>
                    {key}
                  </label>
                  <input
                    type="color"
                    value={value}
                    onChange={(e) => setTheme({
                      colors: { ...theme.colors, [key]: e.target.value }
                    })}
                    style={{
                      width: '100%',
                      height: '40px',
                      border: '2px solid #FF00FF',
                      borderRadius: '8px',
                      cursor: 'pointer'
                    }}
                  />
                </div>
              ))}
            </div>
            
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ color: '#FFFF00', marginBottom: '10px' }}>Effects</h4>
              {Object.entries(theme.effects).map(([key, value]) => (
                <label key={key} style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                  <input
                    type="checkbox"
                    checked={value}
                    onChange={(e) => setTheme({
                      effects: { ...theme.effects, [key]: e.target.checked }
                    })}
                    style={{ marginRight: '10px' }}
                  />
                  <span style={{ textTransform: 'capitalize' }}>{key}</span>
                </label>
              ))}
            </div>
            
            <div>
              <h4 style={{ color: '#FFFF00', marginBottom: '10px' }}>Sound</h4>
              <label style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                <input
                  type="checkbox"
                  checked={theme.sounds.enabled}
                  onChange={(e) => setTheme({
                    sounds: { ...theme.sounds, enabled: e.target.checked }
                  })}
                  style={{ marginRight: '10px' }}
                />
                <span>Enable Sounds</span>
              </label>
              
              {theme.sounds.enabled && (
                <div>
                  <label style={{ display: 'block', marginBottom: '5px' }}>Volume</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={theme.sounds.volume}
                    onChange={(e) => setTheme({
                      sounds: { ...theme.sounds, volume: parseFloat(e.target.value) }
                    })}
                    style={{ width: '100%' }}
                  />
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

// Hook to use cyberpunk theme
export const useCyberpunkTheme = () => {
  const context = useContext(CyberpunkContext);
  if (!context) {
    throw new Error('useCyberpunkTheme must be used within CyberpunkTheme provider');
  }
  return context;
};

export { CyberpunkThemeCustomizer };
export default CyberpunkTheme;