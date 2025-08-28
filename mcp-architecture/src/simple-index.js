#!/usr/bin/env node

/**
 * Ultimate Media Server 2025 - Simple Index
 * Main entry point for single container deployment
 * Starts the MCP server and dashboard for all 30 services
 */

require('dotenv').config();
const express = require('express');
const http = require('http');
const path = require('path');
const SimpleUnifiedMCP = require('./simple-unified-mcp');

const app = express();
const port = process.env.PORT || 8090;

// Create MCP server instance
const mcpServer = new SimpleUnifiedMCP();

// Middleware
app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

// CORS middleware
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
    
    if (req.method === 'OPTIONS') {
        res.sendStatus(200);
    } else {
        next();
    }
});

// Health check endpoint
app.get('/health', async (req, res) => {
    try {
        const health = await mcpServer.getAllServices({ include_health: true });
        const summary = {
            status: 'healthy',
            timestamp: new Date().toISOString(),
            total_services: 30,
            online_services: Object.values(health.services).filter(s => s.health?.status === 'healthy').length,
            uptime: process.uptime(),
            memory: process.memoryUsage(),
            version: '2.0.0'
        };
        res.json(summary);
    } catch (error) {
        res.status(500).json({
            status: 'error',
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

// Services endpoint
app.get('/api/services', async (req, res) => {
    try {
        const services = await mcpServer.getAllServices({ include_health: true });
        res.json(services);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// System overview endpoint
app.get('/api/system', async (req, res) => {
    try {
        const overview = await mcpServer.getSystemOverview({ include_performance: true });
        res.json(overview);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Download status endpoint
app.get('/api/downloads', async (req, res) => {
    try {
        const downloads = await mcpServer.getDownloadStatus();
        res.json(downloads);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Library stats endpoint
app.get('/api/library', async (req, res) => {
    try {
        const stats = await mcpServer.getLibraryStats({ server: 'all', include_recent: true });
        res.json(stats);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Requests overview endpoint
app.get('/api/requests', async (req, res) => {
    try {
        const requests = await mcpServer.getRequestsOverview();
        res.json(requests);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Search endpoint
app.post('/api/search', async (req, res) => {
    try {
        const { query, media_type, services } = req.body;
        const results = await mcpServer.searchAcrossServices({ query, media_type, services });
        res.json(results);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Service management endpoint
app.post('/api/service/:service/:action', async (req, res) => {
    try {
        const { service, action } = req.params;
        const result = await mcpServer.manageService({ service, action });
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Chat/Assistant endpoint
app.post('/api/chat', async (req, res) => {
    try {
        const { message, context } = req.body;
        
        // Simple AI assistant response using MCP data
        const systemOverview = await mcpServer.getSystemOverview();
        const response = {
            message: `I understand you're asking about: "${message}". Based on your Ultimate Media Server 2025 system with ${systemOverview.system.total_services} services, I can help you manage your media ecosystem. What specific assistance do you need?`,
            timestamp: new Date().toISOString(),
            context: systemOverview
        };
        
        res.json(response);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// MCP proxy endpoints
app.all('/mcp/*', async (req, res) => {
    try {
        // Proxy MCP requests to the unified MCP server
        const mcpPath = req.path.replace('/mcp', '');
        res.redirect(`http://localhost:3001${mcpPath}`);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Main dashboard route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/ultimate-dashboard.html'));
});

// Service proxy routes (redirect to service URLs)
const serviceRoutes = {
    '/jellyfin': 'http://localhost:8096',
    '/plex': 'http://localhost:32400',
    '/emby': 'http://localhost:8097',
    '/sonarr': 'http://localhost:8989',
    '/radarr': 'http://localhost:7878',
    '/lidarr': 'http://localhost:8686',
    '/readarr': 'http://localhost:8787',
    '/bazarr': 'http://localhost:6767',
    '/prowlarr': 'http://localhost:9696',
    '/jackett': 'http://localhost:9117',
    '/flaresolverr': 'http://localhost:8191',
    '/qbittorrent': 'http://localhost:8080',
    '/transmission': 'http://localhost:9091',
    '/deluge': 'http://localhost:8112',
    '/nzbget': 'http://localhost:6789',
    '/sabnzbd': 'http://localhost:8085',
    '/overseerr': 'http://localhost:5055',
    '/requestrr': 'http://localhost:4545',
    '/ombi': 'http://localhost:3579',
    '/tautulli': 'http://localhost:8181',
    '/netdata': 'http://localhost:19999',
    '/homepage': 'http://localhost:3000',
    '/heimdall': 'http://localhost:7575',
    '/organizr': 'http://localhost:8081',
    '/homarr': 'http://localhost:7576',
    '/npm': 'http://localhost:81',
    '/portainer': 'http://localhost:9000'
};

// Setup service redirects
Object.entries(serviceRoutes).forEach(([route, target]) => {
    app.get(route, (req, res) => {
        res.redirect(target);
    });
});

// 404 handler
app.use('*', (req, res) => {
    res.status(404).json({
        error: 'Not found',
        message: 'The requested endpoint does not exist',
        available_endpoints: {
            dashboard: '/',
            health: '/health',
            services: '/api/services',
            system: '/api/system',
            downloads: '/api/downloads',
            library: '/api/library',
            requests: '/api/requests',
            search: '/api/search (POST)',
            chat: '/api/chat (POST)',
            mcp: '/mcp/*'
        }
    });
});

// Start the main server
const server = http.createServer(app);

server.listen(port, () => {
    console.log(`🚀 Ultimate Media Server 2025 Dashboard running on port ${port}`);
    console.log(`📊 Managing 30 services across 8 categories`);
    console.log(`🌐 Dashboard: http://localhost:${port}`);
    console.log(`🔧 Health check: http://localhost:${port}/health`);
    console.log(`📋 Services API: http://localhost:${port}/api/services`);
    
    // Start the MCP server on port 3001
    mcpServer.start(3001);
    
    console.log(`📡 MCP Server: http://localhost:3001`);
    console.log(`✨ All systems ready!`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('🛑 Received SIGTERM, shutting down gracefully...');
    server.close(() => {
        console.log('🔚 Server closed');
        process.exit(0);
    });
});

process.on('SIGINT', () => {
    console.log('🛑 Received SIGINT, shutting down gracefully...');
    server.close(() => {
        console.log('🔚 Server closed');
        process.exit(0);
    });
});

module.exports = app;