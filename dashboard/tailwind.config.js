/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        // Enhanced holographic colors
        neon: {
          blue: "#00f5ff",
          purple: "#8a2be2",
          pink: "#ff1493",
          green: "#39ff14",
          yellow: "#ffff00",
          orange: "#ff6600",
        },
        glass: {
          light: "rgba(255, 255, 255, 0.1)",
          medium: "rgba(255, 255, 255, 0.2)",
          dark: "rgba(0, 0, 0, 0.1)",
          blur: "rgba(255, 255, 255, 0.05)",
        },
        // Media server specific colors
        jellyfin: "#00A4DC",
        plex: "#E5A00D",
        sonarr: "#35C5F0",
        radarr: "#FFC230",
        prowlarr: "#FF6900",
        qbittorrent: "#3DAEE9",
        // Enhanced status colors
        success: "#22C55E",
        warning: "#F59E0B",
        error: "#EF4444",
        info: "#3B82F6",
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      backdropBlur: {
        xs: "2px",
        '4xl': "72px",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "slide-in-right": {
          "0%": { transform: "translateX(100%)", opacity: "0" },
          "100%": { transform: "translateX(0)", opacity: "1" },
        },
        "pulse-subtle": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.8" },
        },
        "bounce-subtle": {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-2px)" },
        },
        "glow": {
          "0%, 100%": { boxShadow: "0 0 5px hsl(var(--primary))" },
          "50%": { boxShadow: "0 0 20px hsl(var(--primary))" },
        },
        "holographic": {
          "0%": { 
            background: "linear-gradient(45deg, #00f5ff, #8a2be2)",
            transform: "rotateY(0deg)",
          },
          "25%": { 
            background: "linear-gradient(90deg, #8a2be2, #ff1493)",
            transform: "rotateY(5deg)",
          },
          "50%": { 
            background: "linear-gradient(135deg, #ff1493, #39ff14)",
            transform: "rotateY(0deg)",
          },
          "75%": { 
            background: "linear-gradient(180deg, #39ff14, #ff6600)",
            transform: "rotateY(-5deg)",
          },
          "100%": { 
            background: "linear-gradient(225deg, #ff6600, #00f5ff)",
            transform: "rotateY(0deg)",
          },
        },
        "float": {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-10px)" },
        },
        "rotate-3d": {
          "0%": { transform: "rotateX(0deg) rotateY(0deg)" },
          "25%": { transform: "rotateX(5deg) rotateY(15deg)" },
          "50%": { transform: "rotateX(0deg) rotateY(30deg)" },
          "75%": { transform: "rotateX(-5deg) rotateY(15deg)" },
          "100%": { transform: "rotateX(0deg) rotateY(0deg)" },
        },
        "shimmer": {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
        "matrix-rain": {
          "0%": { transform: "translateY(-100vh)", opacity: "0" },
          "10%": { opacity: "1" },
          "90%": { opacity: "1" },
          "100%": { transform: "translateY(100vh)", opacity: "0" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "fade-in": "fade-in 0.3s ease-out",
        "slide-in-right": "slide-in-right 0.3s ease-out",
        "pulse-subtle": "pulse-subtle 2s infinite",
        "bounce-subtle": "bounce-subtle 1s infinite",
        "glow": "glow 2s infinite",
        "holographic": "holographic 4s ease-in-out infinite",
        "float": "float 3s ease-in-out infinite",
        "rotate-3d": "rotate-3d 8s ease-in-out infinite",
        "shimmer": "shimmer 2s ease-in-out infinite",
        "matrix-rain": "matrix-rain 3s linear infinite",
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic": "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
        "holographic-gradient": "linear-gradient(45deg, #00f5ff 0%, #8a2be2 25%, #ff1493 50%, #39ff14 75%, #ff6600 100%)",
        "glass-gradient": "linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%)",
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        display: ['Orbitron', 'monospace'],
        cyber: ['Audiowide', 'cursive'],
      },
      boxShadow: {
        'neon-blue': '0 0 20px #00f5ff, 0 0 40px #00f5ff, 0 0 80px #00f5ff',
        'neon-purple': '0 0 20px #8a2be2, 0 0 40px #8a2be2, 0 0 80px #8a2be2',
        'neon-pink': '0 0 20px #ff1493, 0 0 40px #ff1493, 0 0 80px #ff1493',
        'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
        'glass-inset': 'inset 0 1px 0 0 rgba(255, 255, 255, 0.2)',
      },
    },
  },
  plugins: [
    require("tailwindcss-animate"),
    require("@tailwindcss/typography"),
  ],
}