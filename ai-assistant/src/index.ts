/**
 * Ultimate Media Server 2025 - AI Assistant
 * Intelligent automation and assistance for media server operations
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import compression from 'compression';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import dotenv from 'dotenv';

import { Logger } from './utils/logger';
import { Database } from './database/database';
import { OllamaService } from './services/ollama.service';
import { MediaServerService } from './services/media-server.service';
import { RecommendationService } from './services/recommendation.service';
import { AutomationService } from './services/automation.service';
import { WebSocketService } from './services/websocket.service';
import { HealthService } from './services/health.service';

// Import routes
import { aiRoutes } from './routes/ai.routes';
import { mediaRoutes } from './routes/media.routes';
import { automationRoutes } from './routes/automation.routes';
import { healthRoutes } from './routes/health.routes';

dotenv.config();

class AIAssistantServer {
    private app: express.Application;
    private server: any;
    private wss: WebSocketServer;
    private logger: Logger;
    private database: Database;
    
    // Services
    private ollamaService: OllamaService;
    private mediaServerService: MediaServerService;
    private recommendationService: RecommendationService;
    private automationService: AutomationService;
    private websocketService: WebSocketService;
    private healthService: HealthService;

    constructor() {
        this.app = express();
        this.logger = new Logger('AIAssistantServer');
        this.database = new Database();
        
        this.setupMiddleware();
        this.initializeServices();
        this.setupRoutes();
        this.setupWebSocket();
        this.setupErrorHandling();
    }

    private setupMiddleware(): void {
        // Security middleware
        this.app.use(helmet({
            contentSecurityPolicy: {
                directives: {
                    defaultSrc: ["'self'"],
                    styleSrc: ["'self'", "'unsafe-inline'"],
                    scriptSrc: ["'self'"],
                    imgSrc: ["'self'", "data:", "https:"],
                    connectSrc: ["'self'", "ws:", "wss:"],
                },
            },
        }));

        // CORS configuration
        this.app.use(cors({
            origin: process.env.CORS_ORIGINS?.split(',') || ['http://localhost:3000'],
            credentials: true,
            methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            allowedHeaders: ['Content-Type', 'Authorization', 'x-api-key'],
        }));

        // Compression and parsing
        this.app.use(compression());
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

        // Logging
        this.app.use(morgan('combined', {
            stream: { write: (message) => this.logger.info(message.trim()) }
        }));

        // Static files
        this.app.use('/static', express.static('public'));
    }

    private async initializeServices(): Promise<void> {
        try {
            // Initialize database
            await this.database.initialize();
            this.logger.info('Database initialized');

            // Initialize core services
            this.ollamaService = new OllamaService();
            this.mediaServerService = new MediaServerService();
            this.recommendationService = new RecommendationService();
            this.automationService = new AutomationService();
            this.healthService = new HealthService();

            // Start services
            await Promise.all([
                this.ollamaService.initialize(),
                this.mediaServerService.initialize(),
                this.recommendationService.initialize(),
                this.automationService.initialize(),
                this.healthService.initialize(),
            ]);

            this.logger.info('All services initialized successfully');

        } catch (error) {
            this.logger.error('Failed to initialize services:', error);
            throw error;
        }
    }

    private setupRoutes(): void {
        // API versioning
        const apiV1 = express.Router();

        // Mount route modules
        apiV1.use('/ai', aiRoutes);
        apiV1.use('/media', mediaRoutes);
        apiV1.use('/automation', automationRoutes);
        apiV1.use('/health', healthRoutes);

        // Mount versioned API
        this.app.use('/api/v1', apiV1);

        // Root endpoints
        this.app.get('/', (req, res) => {
            res.json({
                name: 'Ultimate Media Server AI Assistant',
                version: '2025.1.0',
                status: 'running',
                timestamp: new Date().toISOString(),
                endpoints: {
                    health: '/api/v1/health',
                    ai: '/api/v1/ai',
                    media: '/api/v1/media',
                    automation: '/api/v1/automation',
                    websocket: '/ws'
                }
            });
        });

        this.app.get('/health', (req, res) => {
            res.json({ status: 'healthy', timestamp: new Date().toISOString() });
        });

        // 404 handler
        this.app.use('*', (req, res) => {
            res.status(404).json({
                error: 'Not Found',
                message: `Route ${req.originalUrl} not found`,
                timestamp: new Date().toISOString()
            });
        });
    }

    private setupWebSocket(): void {
        this.server = createServer(this.app);
        
        this.wss = new WebSocketServer({
            server: this.server,
            path: '/ws',
            clientTracking: true,
        });

        this.websocketService = new WebSocketService(this.wss);
        
        this.wss.on('connection', (ws, req) => {
            this.logger.info(`WebSocket connection established from ${req.socket.remoteAddress}`);
            
            ws.on('message', async (data) => {
                try {
                    const message = JSON.parse(data.toString());
                    await this.websocketService.handleMessage(ws, message);
                } catch (error) {
                    this.logger.error('WebSocket message error:', error);
                    ws.send(JSON.stringify({
                        type: 'error',
                        message: 'Invalid message format'
                    }));
                }
            });

            ws.on('close', () => {
                this.logger.info('WebSocket connection closed');
            });

            ws.on('error', (error) => {
                this.logger.error('WebSocket error:', error);
            });

            // Send welcome message
            ws.send(JSON.stringify({
                type: 'welcome',
                message: 'Connected to Ultimate Media Server AI Assistant',
                timestamp: new Date().toISOString()
            }));
        });

        this.logger.info('WebSocket server initialized');
    }

    private setupErrorHandling(): void {
        // Global error handler
        this.app.use((error: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
            this.logger.error('Unhandled error:', error);

            const statusCode = error.statusCode || 500;
            const message = error.message || 'Internal Server Error';

            res.status(statusCode).json({
                error: {
                    message,
                    statusCode,
                    timestamp: new Date().toISOString(),
                    path: req.path
                }
            });
        });

        // Graceful shutdown
        const gracefulShutdown = async (signal: string) => {
            this.logger.info(`Received ${signal}. Starting graceful shutdown...`);

            try {
                // Stop accepting new connections
                this.server.close(() => {
                    this.logger.info('HTTP server closed');
                });

                // Close WebSocket connections
                this.wss.clients.forEach(client => {
                    client.close(1001, 'Server shutting down');
                });

                // Stop services
                await Promise.all([
                    this.automationService.shutdown(),
                    this.ollamaService.shutdown(),
                    this.database.close()
                ]);

                this.logger.info('Graceful shutdown completed');
                process.exit(0);

            } catch (error) {
                this.logger.error('Error during shutdown:', error);
                process.exit(1);
            }
        };

        process.on('SIGINT', () => gracefulShutdown('SIGINT'));
        process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));

        process.on('uncaughtException', (error) => {
            this.logger.error('Uncaught Exception:', error);
            process.exit(1);
        });

        process.on('unhandledRejection', (reason, promise) => {
            this.logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
            process.exit(1);
        });
    }

    public async start(): Promise<void> {
        const port = process.env.PORT || 8090;
        
        return new Promise((resolve) => {
            this.server.listen(port, () => {
                this.logger.info(`🤖 AI Assistant Server started on port ${port}`);
                this.logger.info(`🔗 WebSocket endpoint: ws://localhost:${port}/ws`);
                this.logger.info(`📊 Health check: http://localhost:${port}/health`);
                this.logger.info(`📚 API Documentation: http://localhost:${port}/api/v1`);
                resolve();
            });
        });
    }
}

// Start the server
async function main() {
    try {
        const server = new AIAssistantServer();
        await server.start();
    } catch (error) {
        console.error('Failed to start AI Assistant Server:', error);
        process.exit(1);
    }
}

// Only start if this file is run directly
if (require.main === module) {
    main();
}

export { AIAssistantServer };