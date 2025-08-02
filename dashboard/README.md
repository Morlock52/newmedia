# Ultimate Media Dashboard 2025

A modern, React-based dashboard for managing your media server with real-time monitoring, AI assistance, and beautiful UI.

## Features

### 🎯 Core Features
- **Real-time Service Monitoring** - Monitor all services with live status updates
- **Environment Variable Management** - Secure .env editor with validation
- **AI Assistant** - Intelligent helper for troubleshooting and optimization
- **3D Media Visualization** - Interactive 3D sphere showing media connections
- **Interactive Tutorial** - Step-by-step guide for beginners and advanced users

### 🚀 Technical Features
- **React 18** with TypeScript
- **Material-UI v5** for modern components
- **Three.js** for 3D visualizations
- **Redux Toolkit** for state management
- **React Query** for server state
- **Framer Motion** for animations
- **PWA Support** with offline capabilities
- **WebSocket** real-time updates
- **Responsive Design** mobile-first approach

## Installation

```bash
# Navigate to dashboard directory
cd dashboard

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Environment Setup

Create a `.env` file in the dashboard directory:

```env
VITE_API_URL=http://localhost:3002/api
VITE_WS_URL=ws://localhost:3002/ws
```

## Architecture

### Component Structure
```
src/
├── components/       # Reusable components
│   ├── Dashboard/   # Dashboard-specific components
│   ├── Layout.tsx   # Main layout wrapper
│   └── ...
├── pages/           # Route pages
├── services/        # API and external services
├── store/           # Redux store and slices
├── hooks/           # Custom React hooks
├── utils/           # Utility functions
└── styles/          # Global styles
```

### Key Technologies

#### Frontend Framework
- **React 18** - Latest React features including Suspense
- **TypeScript** - Type safety and better DX
- **Vite** - Lightning fast build tool

#### UI/UX
- **Material-UI v5** - Comprehensive component library
- **Framer Motion** - Smooth animations
- **Three.js + React Three Fiber** - 3D graphics
- **Chart.js & Recharts** - Data visualization

#### State Management
- **Redux Toolkit** - Simplified Redux with best practices
- **React Query** - Server state management
- **Zustand** - Lightweight state for UI

#### Development Tools
- **ESLint** - Code linting
- **Prettier** - Code formatting
- **Vitest** - Unit testing
- **MSW** - API mocking

## Features in Detail

### Service Control Panel
- Start/stop/restart services
- Real-time status monitoring
- Resource usage tracking
- Health checks
- Auto-start configuration

### Environment Manager
- Category-based organization
- Secure value handling
- Validation and error checking
- Import/export functionality
- Random value generation for secrets

### AI Assistant
- Natural language queries
- Voice input support
- Contextual suggestions
- System optimization recommendations
- Troubleshooting help

### Media Visualization
- 3D interactive sphere
- Media type connections
- Real-time updates
- Gesture controls

### Analytics Dashboard
- Performance metrics
- Usage statistics
- Trend analysis
- Custom date ranges

### Tutorial System
- Interactive walkthroughs
- Code examples with syntax highlighting
- Progress tracking
- Quick action suggestions

## Performance Optimizations

- **Code Splitting** - Lazy loading for routes
- **Bundle Optimization** - Separate vendor chunks
- **Image Optimization** - WebP with fallbacks
- **Service Worker** - Offline support and caching
- **Virtual Scrolling** - For large lists
- **Memoization** - Prevent unnecessary re-renders

## Security Features

- **Secure Environment Variables** - Password field masking
- **API Authentication** - JWT token management
- **Input Validation** - Form validation with Yup
- **XSS Protection** - Sanitized user inputs
- **HTTPS Only** - Secure connections enforced

## Browser Support

- Chrome/Edge (latest 2 versions)
- Firefox (latest 2 versions)
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Android)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - feel free to use in your own projects!