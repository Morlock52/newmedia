/**
 * Enhanced API Server with WebSocket Support
 * Provides unified API endpoints for all services with real-time capabilities
 */

import express from 'express'
import { createServer } from 'http'
import cors from 'cors'
import helmet from 'helmet'
import rateLimit from 'express-rate-limit'
import swaggerJSDoc from 'swagger-jsdoc'
import swaggerUi from 'swagger-ui-express'

import { HealthChecker } from '../lib/health/HealthChecker'
import { ServiceDiscovery } from '../lib/discovery/ServiceDiscovery'
import { AuthSystem } from '../lib/auth/AuthSystem'
import { RealTimeManager } from '../lib/websocket/RealTimeManager'
import { RecommendationEngine } from '../lib/ai/RecommendationEngine'
import { ServiceManager } from '../lib/services/ServiceConnectors'

const app = express()
const server = createServer(app)

// Initialize services
const serviceConfigs = {
  jellyfin: { url: process.env.JELLYFIN_URL || 'http://localhost:8096', auth: false },
  plex: { url: process.env.PLEX_URL || 'http://localhost:32400', auth: true },
  sonarr: { url: process.env.SONARR_URL || 'http://localhost:8989', auth: true, apiKey: process.env.SONARR_API_KEY },
  radarr: { url: process.env.RADARR_URL || 'http://localhost:7878', auth: true, apiKey: process.env.RADARR_API_KEY },
  qbittorrent: { url: process.env.QBITTORRENT_URL || 'http://localhost:8080', auth: true, username: process.env.QB_USERNAME, password: process.env.QB_PASSWORD },
  prowlarr: { url: process.env.PROWLARR_URL || 'http://localhost:9696', auth: true, apiKey: process.env.PROWLARR_API_KEY }
}

const healthChecker = new HealthChecker(serviceConfigs)
const serviceDiscovery = new ServiceDiscovery()
const authSystem = new AuthSystem(
  process.env.JWT_SECRET || 'development-secret',
  {
    google: {
      clientId: process.env.GOOGLE_CLIENT_ID || '',
      clientSecret: process.env.GOOGLE_CLIENT_SECRET || '',
      redirectUri: process.env.GOOGLE_REDIRECT_URI || ''
    },
    github: {
      clientId: process.env.GITHUB_CLIENT_ID || '',
      clientSecret: process.env.GITHUB_CLIENT_SECRET || '',
      redirectUri: process.env.GITHUB_REDIRECT_URI || ''
    }
  },
  process.env.REDIS_URL
)

const realTimeManager = new RealTimeManager(server)
const recommendationEngine = new RecommendationEngine()
const serviceManager = new ServiceManager(serviceConfigs)

// Auto-register services with discovery
serviceDiscovery.autoRegisterServices(serviceConfigs)

// Middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "ws:", "wss:"]
    }
  }
}))

app.use(cors({
  origin: process.env.NODE_ENV === 'production' 
    ? process.env.FRONTEND_URL 
    : ['http://localhost:3000', 'http://127.0.0.1:3000'],
  credentials: true
}))

app.use(express.json({ limit: '50mb' }))
app.use(express.urlencoded({ extended: true }))

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: { error: 'Too many requests, please try again later' }
})

app.use('/api/', limiter as any)

// Swagger configuration
const swaggerOptions = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'Ultimate Media Server API',
      version: '2025.1.0',
      description: 'Comprehensive API for managing media services, downloads, and user interactions',
      contact: {
        name: 'API Support',
        email: 'support@ultimatemediaserver.com'
      }
    },
    servers: [
      {
        url: process.env.API_BASE_URL || 'http://localhost:3001',
        description: 'Development server'
      }
    ],
    components: {
      securitySchemes: {
        bearerAuth: {
          type: 'http',
          scheme: 'bearer',
          bearerFormat: 'JWT'
        }
      }
    }
  },
  apis: ['./src/server/routes/*.ts', './src/server/api.ts']
}

const specs = swaggerJSDoc(swaggerOptions)
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(specs))

// Authentication middleware
const authenticateToken = async (req: any, res: any, next: any) => {
  const authHeader = req.headers['authorization']
  const token = authHeader && authHeader.split(' ')[1]

  if (!token) {
    return res.status(401).json({ error: 'Access token required' })
  }

  try {
    const user = await authSystem.verifyToken(token)
    if (!user) {
      return res.status(403).json({ error: 'Invalid or expired token' })
    }
    req.user = user
    next()
  } catch (error) {
    return res.status(403).json({ error: 'Invalid token' })
  }
}

/**
 * @swagger
 * components:
 *   schemas:
 *     HealthStatus:
 *       type: object
 *       properties:
 *         service:
 *           type: string
 *         status:
 *           type: string
 *           enum: [healthy, unhealthy, degraded, unknown]
 *         responseTime:
 *           type: number
 *         lastCheck:
 *           type: string
 *         uptime:
 *           type: number
 *         errorCount:
 *           type: number
 *         circuitState:
 *           type: string
 *           enum: [closed, open, half-open]
 */

/**
 * @swagger
 * /api/health:
 *   get:
 *     summary: Get health status of all services
 *     tags: [Health]
 *     responses:
 *       200:
 *         description: Health status of all services
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 timestamp:
 *                   type: string
 *                 services:
 *                   type: array
 *                   items:
 *                     $ref: '#/components/schemas/HealthStatus'
 *                 summary:
 *                   type: object
 */
app.get('/api/health', async (req, res) => {
  try {
    const healthStatuses = await healthChecker.checkAllServices()
    
    // Update service discovery with health info
    healthStatuses.forEach(status => {
      serviceDiscovery.updateServiceHealth(status.service, `${status.service}-primary`, status.status === 'healthy' ? 'healthy' : 'unhealthy')
    })
    
    // Broadcast health updates via WebSocket
    realTimeManager.broadcastMessage('services:health', healthStatuses)
    
    res.json({
      timestamp: new Date().toISOString(),
      services: healthStatuses,
      summary: {
        total: healthStatuses.length,
        healthy: healthStatuses.filter(s => s.status === 'healthy').length,
        unhealthy: healthStatuses.filter(s => s.status === 'unhealthy').length,
        degraded: healthStatuses.filter(s => s.status === 'degraded').length
      }
    })
  } catch (error: any) {
    res.status(500).json({ error: 'Health check failed', message: error?.message || 'Unknown error' })
  }
})

/**
 * @swagger
 * /api/health/{service}:
 *   get:
 *     summary: Get health status of specific service
 *     tags: [Health]
 *     parameters:
 *       - name: service
 *         in: path
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Health status of the service
 */
app.get('/api/health/:service', async (req, res) => {
  try {
    const { service } = req.params
    const config = serviceConfigs[service as keyof typeof serviceConfigs]
    
    if (!config) {
      return res.status(404).json({ error: 'Service not found' })
    }
    
    const healthStatus = await healthChecker.checkService(service, config)
    res.json(healthStatus)
  } catch (error: any) {
    res.status(500).json({ error: 'Health check failed', message: error?.message || 'Unknown error' })
  }
})

/**
 * @swagger
 * /api/auth/register:
 *   post:
 *     summary: Register a new user
 *     tags: [Authentication]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - email
 *               - username
 *               - password
 *               - confirmPassword
 *             properties:
 *               email:
 *                 type: string
 *               username:
 *                 type: string
 *               password:
 *                 type: string
 *               confirmPassword:
 *                 type: string
 *     responses:
 *       201:
 *         description: User registered successfully
 */
app.post('/api/auth/register', async (req, res) => {
  try {
    const result = await authSystem.register(req.body)
    res.status(201).json(result)
  } catch (error: any) {
    res.status(400).json({ error: error?.message || 'Registration failed' })
  }
})

/**
 * @swagger
 * /api/auth/login:
 *   post:
 *     summary: Login with email and password
 *     tags: [Authentication]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - email
 *               - password
 *             properties:
 *               email:
 *                 type: string
 *               password:
 *                 type: string
 *               rememberMe:
 *                 type: boolean
 *     responses:
 *       200:
 *         description: Login successful
 */
app.post('/api/auth/login', async (req, res) => {
  try {
    const result = await authSystem.login(req.body)
    res.json(result)
  } catch (error: any) {
    res.status(401).json({ error: error?.message || 'Authentication failed' })
  }
})

/**
 * @swagger
 * /api/auth/google:
 *   post:
 *     summary: Authenticate with Google OAuth2
 *     tags: [Authentication]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - code
 *             properties:
 *               code:
 *                 type: string
 *     responses:
 *       200:
 *         description: Google authentication successful
 */
app.post('/api/auth/google', async (req, res) => {
  try {
    const { code } = req.body
    const result = await authSystem.authenticateWithGoogle(code)
    res.json(result)
  } catch (error: any) {
    res.status(401).json({ error: error?.message || 'Authentication failed' })
  }
})

/**
 * @swagger
 * /api/auth/github:
 *   post:
 *     summary: Authenticate with GitHub OAuth2
 *     tags: [Authentication]
 */
app.post('/api/auth/github', async (req, res) => {
  try {
    const { code } = req.body
    const result = await authSystem.authenticateWithGitHub(code)
    res.json(result)
  } catch (error: any) {
    res.status(401).json({ error: error?.message || 'Authentication failed' })
  }
})

/**
 * @swagger
 * /api/auth/refresh:
 *   post:
 *     summary: Refresh access token
 *     tags: [Authentication]
 */
app.post('/api/auth/refresh', async (req, res) => {
  try {
    const { refreshToken } = req.body
    const result = await authSystem.refreshToken(refreshToken)
    res.json(result)
  } catch (error: any) {
    res.status(401).json({ error: error?.message || 'Authentication failed' })
  }
})

/**
 * @swagger
 * /api/auth/logout:
 *   post:
 *     summary: Logout and invalidate tokens
 *     tags: [Authentication]
 *     security:
 *       - bearerAuth: []
 */
app.post('/api/auth/logout', authenticateToken, async (req, res) => {
  try {
    const { refreshToken } = req.body
    const accessToken = req.headers.authorization?.split(' ')[1]
    
    if (accessToken && refreshToken) {
      await authSystem.logout(accessToken, refreshToken)
    }
    
    res.json({ message: 'Logged out successfully' })
  } catch (error: any) {
    res.status(500).json({ error: error?.message || 'Internal server error' })
  }
})

/**
 * @swagger
 * /api/services/discovery:
 *   get:
 *     summary: Get service discovery topology
 *     tags: [Services]
 *     security:
 *       - bearerAuth: []
 */
app.get('/api/services/discovery', authenticateToken, (req, res) => {
  try {
    const topology = serviceDiscovery.getServiceTopology()
    const stats = serviceDiscovery.getLoadBalancingStats()
    
    res.json({
      topology,
      loadBalancing: stats,
      timestamp: new Date().toISOString()
    })
  } catch (error: any) {
    res.status(500).json({ error: error?.message || 'Internal server error' })
  }
})

/**
 * @swagger
 * /api/media/movies:
 *   get:
 *     summary: Get all movies from connected services
 *     tags: [Media]
 *     security:
 *       - bearerAuth: []
 */
app.get('/api/media/movies', authenticateToken, async (req, res) => {
  try {
    const movies = await serviceManager.getAllMovies()
    res.json(movies)
  } catch (error: any) {
    res.status(500).json({ error: error?.message || 'Internal server error' })
  }
})

/**
 * @swagger
 * /api/media/tv:
 *   get:
 *     summary: Get all TV shows from connected services
 *     tags: [Media]
 *     security:
 *       - bearerAuth: []
 */
app.get('/api/media/tv', authenticateToken, async (req, res) => {
  try {
    const shows = await serviceManager.getAllTVShows()
    res.json(shows)
  } catch (error: any) {
    res.status(500).json({ error: error?.message || 'Internal server error' })
  }
})

/**
 * @swagger
 * /api/downloads:
 *   get:
 *     summary: Get all active downloads
 *     tags: [Downloads]
 *     security:
 *       - bearerAuth: []
 */
app.get('/api/downloads', authenticateToken, async (req, res) => {
  try {
    const downloads = await serviceManager.getAllDownloads()
    res.json(downloads)
  } catch (error: any) {
    res.status(500).json({ error: error?.message || 'Internal server error' })
  }
})

/**
 * @swagger
 * /api/search:
 *   post:
 *     summary: Search for content across all services
 *     tags: [Search]
 *     security:
 *       - bearerAuth: []
 */
app.post('/api/search', authenticateToken, async (req, res) => {
  try {
    const { query } = req.body
    const results = await serviceManager.searchContent(query)
    res.json(results)
  } catch (error: any) {
    res.status(500).json({ error: error?.message || 'Internal server error' })
  }
})

/**
 * @swagger
 * /api/recommendations:
 *   get:
 *     summary: Get personalized content recommendations
 *     tags: [AI]
 *     security:
 *       - bearerAuth: []
 */
app.get('/api/recommendations', authenticateToken, async (req, res) => {
  try {
    const { user } = req
    const limit = parseInt(req.query.limit as string) || 20
    
    const recommendations = recommendationEngine.getRecommendations(user.id, limit)
    res.json(recommendations)
  } catch (error: any) {
    res.status(500).json({ error: error?.message || 'Internal server error' })
  }
})

/**
 * @swagger
 * /api/analytics/viewing:
 *   get:
 *     summary: Get viewing pattern analysis
 *     tags: [AI]
 *     security:
 *       - bearerAuth: []
 */
app.get('/api/analytics/viewing', authenticateToken, async (req, res) => {
  try {
    const { user } = req
    const analysis = recommendationEngine.analyzeViewingPatterns(user.id)
    res.json(analysis)
  } catch (error: any) {
    res.status(500).json({ error: error?.message || 'Internal server error' })
  }
})

/**
 * @swagger
 * /api/analytics/viewing:
 *   post:
 *     summary: Record viewing pattern
 *     tags: [AI]
 *     security:
 *       - bearerAuth: []
 */
app.post('/api/analytics/viewing', authenticateToken, async (req, res) => {
  try {
    const { user } = req
    const pattern = { ...req.body, userId: user.id }
    
    recommendationEngine.recordViewingPattern(pattern)
    res.json({ message: 'Viewing pattern recorded' })
  } catch (error: any) {
    res.status(500).json({ error: error?.message || 'Internal server error' })
  }
})

/**
 * @swagger
 * /api/downloads/schedule:
 *   post:
 *     summary: Generate smart download schedule
 *     tags: [AI]
 *     security:
 *       - bearerAuth: []
 */
app.post('/api/downloads/schedule', authenticateToken, async (req, res) => {
  try {
    const { user } = req
    const { contentIds } = req.body
    
    const schedule = recommendationEngine.generateDownloadSchedule(contentIds, user.id)
    res.json(schedule)
  } catch (error: any) {
    res.status(500).json({ error: error?.message || 'Internal server error' })
  }
})

/**
 * @swagger
 * /api/cache/predictions:
 *   get:
 *     summary: Get predictive caching recommendations
 *     tags: [AI]
 *     security:
 *       - bearerAuth: []
 */
app.get('/api/cache/predictions', authenticateToken, async (req, res) => {
  try {
    const { user } = req
    const predictions = recommendationEngine.generateCachePredictions(user.id)
    res.json(predictions)
  } catch (error: any) {
    res.status(500).json({ error: error?.message || 'Internal server error' })
  }
})

// Circuit breaker management endpoints
app.post('/api/health/:service/circuit-breaker/reset', authenticateToken, async (req, res) => {
  try {
    const { service } = req.params
    await healthChecker.forceCircuitBreakerReset(service)
    res.json({ message: `Circuit breaker reset for ${service}` })
  } catch (error: any) {
    res.status(500).json({ error: error?.message || 'Internal server error' })
  }
})

app.get('/api/health/:service/history', authenticateToken, (req, res) => {
  try {
    const { service } = req.params
    const history = healthChecker.getServiceHistory(service)
    res.json(history)
  } catch (error: any) {
    res.status(500).json({ error: error?.message || 'Internal server error' })
  }
})

// WebSocket status endpoint
app.get('/api/websocket/status', authenticateToken, (req, res) => {
  try {
    const stats = {
      connectedUsers: realTimeManager.getConnectedUsers(),
      activeDownloads: realTimeManager.getActiveDownloads().length,
      totalDownloads: realTimeManager.getAllDownloads().length,
      onlineServices: realTimeManager.getServiceStatuses().filter(s => s.status === 'online').length
    }
    
    res.json(stats)
  } catch (error: any) {
    res.status(500).json({ error: error?.message || 'Internal server error' })
  }
})

// Error handling
app.use((error: any, req: any, res: any, next: any) => {
  console.error('Unhandled error:', error)
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
  })
})

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Endpoint not found' })
})

// Health monitoring loop
setInterval(async () => {
  try {
    const healthStatuses = await healthChecker.checkAllServices()
    
    // Update service discovery
    healthStatuses.forEach(status => {
      serviceDiscovery.updateServiceHealth(
        status.service, 
        `${status.service}-primary`, 
        status.status === 'healthy' ? 'healthy' : 'unhealthy'
      )
      
      // Update real-time manager
      realTimeManager.updateServiceStatus({
        name: status.service,
        status: status.status === 'healthy' ? 'online' : 
                status.status === 'degraded' ? 'degraded' : 'offline',
        responseTime: status.responseTime,
        lastUpdate: new Date()
      })
    })
    
    // Cleanup stale services
    serviceDiscovery.cleanupStaleServices()
  } catch (error) {
    console.error('Health monitoring error:', error)
  }
}, 30000) // Check every 30 seconds

const PORT = process.env.PORT || 3001

server.listen(PORT, () => {
  console.log(`🚀 Ultimate Media Server API running on port ${PORT}`)
  console.log(`📊 API Documentation available at http://localhost:${PORT}/api-docs`)
  console.log(`🔌 WebSocket server enabled`)
  console.log(`🔒 Authentication system ready`)
  console.log(`🤖 AI recommendation engine initialized`)
  console.log(`🎯 Service discovery active`)
})

export default app