// Simple demo server for testing the holographic dashboard
// This creates a basic HTTP server and WebSocket server for local testing

const http = require('http');
const fs = require('fs');
const path = require('path');
const WebSocket = require('ws');

const PORT = 9999;
const WS_PORT = 9998;

// MIME types
const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon'
};

// Create HTTP server
const server = http.createServer((req, res) => {
    console.log(`${req.method} ${req.url}`);
    
    // Parse URL
    let filePath = '.' + req.url;
    if (filePath === './') {
        filePath = './index.html';
    }
    
    const extname = String(path.extname(filePath)).toLowerCase();
    const contentType = mimeTypes[extname] || 'application/octet-stream';
    
    fs.readFile(filePath, (error, content) => {
        if (error) {
            if (error.code === 'ENOENT') {
                res.writeHead(404, { 'Content-Type': 'text/html' });
                res.end('<h1>404 - File Not Found</h1>', 'utf-8');
            } else {
                res.writeHead(500);
                res.end(`Server Error: ${error.code}`, 'utf-8');
            }
        } else {
            res.writeHead(200, { 
                'Content-Type': contentType,
                'Access-Control-Allow-Origin': '*'
            });
            res.end(content, 'utf-8');
        }
    });
});

// Create WebSocket server
const wss = new WebSocket.Server({ port: WS_PORT });

console.log(`HTTP Server running at http://localhost:${PORT}/`);
console.log(`WebSocket Server running at ws://localhost:${WS_PORT}/`);

// WebSocket connection handling
wss.on('connection', (ws) => {
    console.log('New WebSocket client connected');
    
    // Send welcome message
    ws.send(JSON.stringify({
        type: 'connected',
        data: {
            message: 'Welcome to Holographic Media Server',
            serverTime: new Date().toISOString()
        }
    }));
    
    // Handle messages from client
    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            console.log('Received:', data.type);
            
            // Handle different message types
            switch (data.type) {
                case 'handshake':
                    ws.send(JSON.stringify({
                        type: 'handshake-ack',
                        data: {
                            serverVersion: '2.0.0',
                            features: ['realtime-stats', 'media-streaming', 'transcoding']
                        }
                    }));
                    break;
                    
                case 'request-stats':
                    sendStats(ws);
                    break;
                    
                case 'request-media-list':
                    sendMediaList(ws);
                    break;
                    
                default:
                    console.log('Unknown message type:', data.type);
            }
        } catch (error) {
            console.error('Error handling message:', error);
        }
    });
    
    ws.on('close', () => {
        console.log('Client disconnected');
    });
    
    // Send periodic updates
    const statsInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
            sendStats(ws);
        } else {
            clearInterval(statsInterval);
        }
    }, 5000);
    
    const activityInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
            sendRandomActivity(ws);
        } else {
            clearInterval(activityInterval);
        }
    }, 10000);
});

// Helper functions
function sendStats(ws) {
    ws.send(JSON.stringify({
        type: 'stats-update',
        data: {
            totalMedia: 2847 + Math.floor(Math.random() * 20),
            storageUsed: 47.3 + (Math.random() - 0.5),
            activeUsers: 12 + Math.floor(Math.random() * 5),
            bandwidth: 450 + Math.floor(Math.random() * 100),
            activeStreams: 8 + Math.floor(Math.random() * 4),
            gpuUsage: 65 + Math.floor(Math.random() * 20),
            cpuUsage: 45 + Math.floor(Math.random() * 20),
            memoryUsage: 72 + Math.floor(Math.random() * 10)
        }
    }));
}

function sendMediaList(ws) {
    const mediaTypes = ['movie', 'series', 'music', 'documentary'];
    const titles = [
        'Blade Runner 2049', 'The Matrix Resurrections', 'Dune Part Two',
        'Interstellar', 'Inception', 'The Mandalorian', 'Westworld',
        'Black Mirror', 'Stranger Things', 'The Expanse'
    ];
    
    const mediaList = [];
    for (let i = 0; i < 20; i++) {
        mediaList.push({
            id: `media-${i}`,
            title: titles[i % titles.length],
            type: mediaTypes[Math.floor(Math.random() * mediaTypes.length)],
            year: 2020 + Math.floor(Math.random() * 5),
            quality: Math.random() > 0.5 ? '4K HDR' : '1080p',
            duration: 90 + Math.floor(Math.random() * 90)
        });
    }
    
    ws.send(JSON.stringify({
        type: 'media-list',
        data: {
            items: mediaList,
            total: mediaList.length
        }
    }));
}

function sendRandomActivity(ws) {
    const activities = [
        { icon: '🎬', title: 'New movie added', description: 'Avatar: The Way of Water' },
        { icon: '👤', title: 'User activity', description: 'Sarah started watching The Crown' },
        { icon: '📡', title: 'Live stream', description: 'ESPN HD stream started' },
        { icon: '🔄', title: 'Transcoding', description: 'Completed 4K conversion' },
        { icon: '⬇️', title: 'Download', description: 'The Last of Us S01E05 completed' }
    ];
    
    const activity = activities[Math.floor(Math.random() * activities.length)];
    
    ws.send(JSON.stringify({
        type: 'activity',
        data: {
            ...activity,
            timestamp: Date.now()
        }
    }));
}

server.listen(PORT);