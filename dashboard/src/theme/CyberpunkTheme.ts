// Cyberpunk Theme System - Ultimate Media Server 2025
// Centralized theme configuration for consistent cyberpunk aesthetics

export interface CyberpunkTheme {
  colors: ColorPalette;
  typography: Typography;
  animations: Animations;
  effects: Effects;
  spacing: Spacing;
  breakpoints: Breakpoints;
  components: ComponentStyles;
}

interface ColorPalette {
  primary: {
    neon: string;
    cyan: string;
    magenta: string;
    yellow: string;
    purple: string;
  };
  secondary: {
    electric: string;
    plasma: string;
    laser: string;
    hologram: string;
    matrix: string;
  };
  status: {
    online: string;
    offline: string;
    degraded: string;
    maintenance: string;
    loading: string;
  };
  background: {
    dark: string;
    darker: string;
    darkest: string;
    gradient: string;
    glass: string;
  };
  text: {
    primary: string;
    secondary: string;
    accent: string;
    muted: string;
    inverse: string;
  };
  ui: {
    border: string;
    borderHover: string;
    shadow: string;
    glow: string;
    overlay: string;
  };
}

interface Typography {
  fonts: {
    primary: string;
    secondary: string;
    mono: string;
    display: string;
  };
  sizes: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
    xxl: string;
    display: string;
  };
  weights: {
    light: number;
    regular: number;
    medium: number;
    bold: number;
    black: number;
  };
  lineHeights: {
    tight: number;
    normal: number;
    relaxed: number;
    loose: number;
  };
  letterSpacing: {
    tight: string;
    normal: string;
    wide: string;
    wider: string;
    widest: string;
  };
}

interface Animations {
  durations: {
    instant: string;
    fast: string;
    normal: string;
    slow: string;
    slower: string;
  };
  easings: {
    sharp: string;
    smooth: string;
    elastic: string;
    bounce: string;
    cyber: string;
  };
  keyframes: {
    pulse: string;
    glow: string;
    glitch: string;
    scan: string;
    matrix: string;
    hologram: string;
    fadeIn: string;
    slideUp: string;
    rotate: string;
    float: string;
  };
}

interface Effects {
  glows: {
    sm: string;
    md: string;
    lg: string;
    xl: string;
    neon: string;
  };
  blurs: {
    none: string;
    sm: string;
    md: string;
    lg: string;
    glass: string;
  };
  gradients: {
    cyberpunk: string;
    neon: string;
    holographic: string;
    matrix: string;
    plasma: string;
  };
  filters: {
    glitch: string;
    noise: string;
    scanlines: string;
    chromatic: string;
    vhs: string;
  };
  borders: {
    thin: string;
    medium: string;
    thick: string;
    neon: string;
    animated: string;
  };
}

interface Spacing {
  xs: string;
  sm: string;
  md: string;
  lg: string;
  xl: string;
  xxl: string;
  xxxl: string;
}

interface Breakpoints {
  mobile: string;
  tablet: string;
  laptop: string;
  desktop: string;
  wide: string;
  ultrawide: string;
}

interface ComponentStyles {
  button: {
    variants: string[];
    sizes: string[];
    states: string[];
  };
  card: {
    variants: string[];
    elevations: string[];
  };
  input: {
    variants: string[];
    sizes: string[];
  };
  modal: {
    variants: string[];
    sizes: string[];
  };
  notification: {
    variants: string[];
    positions: string[];
  };
}

// Cyberpunk Theme Configuration
export const cyberpunkTheme: CyberpunkTheme = {
  colors: {
    primary: {
      neon: '#00ffff',
      cyan: '#00e5ff',
      magenta: '#ff00ff',
      yellow: '#ffff00',
      purple: '#8b00ff'
    },
    secondary: {
      electric: '#00bfff',
      plasma: '#ff1493',
      laser: '#ff0080',
      hologram: '#00ffa5',
      matrix: '#00ff00'
    },
    status: {
      online: '#00ff00',
      offline: '#ff0000',
      degraded: '#ffff00',
      maintenance: '#ff00ff',
      loading: '#00ffff'
    },
    background: {
      dark: '#1a1a2a',
      darker: '#0a0a1a',
      darkest: '#000008',
      gradient: 'linear-gradient(135deg, #0a0a0a 0%, #1a0a2a 100%)',
      glass: 'rgba(0, 0, 0, 0.8)'
    },
    text: {
      primary: '#ffffff',
      secondary: '#00ffff',
      accent: '#ff00ff',
      muted: '#888888',
      inverse: '#000000'
    },
    ui: {
      border: '#333333',
      borderHover: '#00ffff',
      shadow: 'rgba(0, 255, 255, 0.3)',
      glow: 'rgba(0, 255, 255, 0.5)',
      overlay: 'rgba(0, 0, 0, 0.7)'
    }
  },
  typography: {
    fonts: {
      primary: "'Orbitron', sans-serif",
      secondary: "'Exo 2', sans-serif",
      mono: "'Fira Code', 'Courier New', monospace",
      display: "'Audiowide', cursive"
    },
    sizes: {
      xs: '0.75rem',
      sm: '0.875rem',
      md: '1rem',
      lg: '1.25rem',
      xl: '1.5rem',
      xxl: '2rem',
      display: '3rem'
    },
    weights: {
      light: 300,
      regular: 400,
      medium: 500,
      bold: 700,
      black: 900
    },
    lineHeights: {
      tight: 1.2,
      normal: 1.5,
      relaxed: 1.75,
      loose: 2
    },
    letterSpacing: {
      tight: '-0.05em',
      normal: '0',
      wide: '0.05em',
      wider: '0.1em',
      widest: '0.2em'
    }
  },
  animations: {
    durations: {
      instant: '0ms',
      fast: '150ms',
      normal: '300ms',
      slow: '500ms',
      slower: '1000ms'
    },
    easings: {
      sharp: 'cubic-bezier(0.4, 0, 0.6, 1)',
      smooth: 'cubic-bezier(0.4, 0, 0.2, 1)',
      elastic: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
      bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
      cyber: 'cubic-bezier(0.23, 1, 0.32, 1)'
    },
    keyframes: {
      pulse: 'pulse',
      glow: 'glow',
      glitch: 'glitch',
      scan: 'scan',
      matrix: 'matrix',
      hologram: 'hologram',
      fadeIn: 'fadeIn',
      slideUp: 'slideUp',
      rotate: 'rotate',
      float: 'float'
    }
  },
  effects: {
    glows: {
      sm: '0 0 10px',
      md: '0 0 20px',
      lg: '0 0 30px',
      xl: '0 0 40px',
      neon: '0 0 20px currentColor, 0 0 40px currentColor'
    },
    blurs: {
      none: 'blur(0)',
      sm: 'blur(4px)',
      md: 'blur(8px)',
      lg: 'blur(12px)',
      glass: 'blur(20px)'
    },
    gradients: {
      cyberpunk: 'linear-gradient(135deg, #00ffff, #ff00ff)',
      neon: 'linear-gradient(90deg, #00ffff, #ff00ff, #ffff00)',
      holographic: 'linear-gradient(45deg, #00ffff, #00ff00, #ff00ff, #ffff00)',
      matrix: 'linear-gradient(180deg, #00ff00, #008800, #000000)',
      plasma: 'radial-gradient(circle, #ff00ff, #00ffff, #ff00ff)'
    },
    filters: {
      glitch: 'hue-rotate(90deg) saturate(2)',
      noise: 'contrast(1.2) brightness(1.1)',
      scanlines: 'contrast(1.1) brightness(0.95)',
      chromatic: 'hue-rotate(180deg) saturate(1.5)',
      vhs: 'contrast(1.3) brightness(0.9) sepia(0.1)'
    },
    borders: {
      thin: '1px solid',
      medium: '2px solid',
      thick: '3px solid',
      neon: '2px solid currentColor',
      animated: '2px solid transparent'
    }
  },
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    xxl: '3rem',
    xxxl: '4rem'
  },
  breakpoints: {
    mobile: '320px',
    tablet: '768px',
    laptop: '1024px',
    desktop: '1440px',
    wide: '1920px',
    ultrawide: '2560px'
  },
  components: {
    button: {
      variants: ['primary', 'secondary', 'ghost', 'neon', 'holographic'],
      sizes: ['xs', 'sm', 'md', 'lg', 'xl'],
      states: ['default', 'hover', 'active', 'disabled', 'loading']
    },
    card: {
      variants: ['default', 'glass', 'neon', 'holographic', 'matrix'],
      elevations: ['flat', 'raised', 'floating', 'hovering']
    },
    input: {
      variants: ['default', 'neon', 'glass', 'minimal', 'cyber'],
      sizes: ['sm', 'md', 'lg']
    },
    modal: {
      variants: ['default', 'glass', 'holographic', 'fullscreen'],
      sizes: ['sm', 'md', 'lg', 'xl', 'full']
    },
    notification: {
      variants: ['info', 'success', 'warning', 'error', 'cyber'],
      positions: ['top', 'bottom', 'topRight', 'topLeft', 'center']
    }
  }
};

// Theme utility functions
export const getColor = (path: string): string => {
  const keys = path.split('.');
  let value: any = cyberpunkTheme.colors;
  for (const key of keys) {
    value = value[key];
  }
  return value;
};

export const getSpacing = (size: keyof typeof cyberpunkTheme.spacing): string => {
  return cyberpunkTheme.spacing[size];
};

export const getAnimation = (name: keyof typeof cyberpunkTheme.animations.keyframes): string => {
  return cyberpunkTheme.animations.keyframes[name];
};

export const getGradient = (type: keyof typeof cyberpunkTheme.effects.gradients): string => {
  return cyberpunkTheme.effects.gradients[type];
};

export const getBreakpoint = (size: keyof typeof cyberpunkTheme.breakpoints): string => {
  return cyberpunkTheme.breakpoints[size];
};

// CSS-in-JS helper
export const generateCSSVariables = (): string => {
  const cssVars: string[] = [];
  
  // Colors
  Object.entries(cyberpunkTheme.colors).forEach(([category, values]) => {
    Object.entries(values).forEach(([key, value]) => {
      cssVars.push(`--color-${category}-${key}: ${value};`);
    });
  });
  
  // Typography
  Object.entries(cyberpunkTheme.typography.fonts).forEach(([key, value]) => {
    cssVars.push(`--font-${key}: ${value};`);
  });
  
  Object.entries(cyberpunkTheme.typography.sizes).forEach(([key, value]) => {
    cssVars.push(`--text-${key}: ${value};`);
  });
  
  // Spacing
  Object.entries(cyberpunkTheme.spacing).forEach(([key, value]) => {
    cssVars.push(`--spacing-${key}: ${value};`);
  });
  
  // Animations
  Object.entries(cyberpunkTheme.animations.durations).forEach(([key, value]) => {
    cssVars.push(`--duration-${key}: ${value};`);
  });
  
  return cssVars.join('\n  ');
};

// Theme provider types
export interface ThemeContextType {
  theme: CyberpunkTheme;
  isDarkMode: boolean;
  toggleDarkMode: () => void;
  accentColor: string;
  setAccentColor: (color: string) => void;
}

export default cyberpunkTheme;