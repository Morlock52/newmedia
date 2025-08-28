#!/usr/bin/env node

/**
 * Start Fixed Dashboard Server
 * This script starts the comprehensive dashboard server with Socket.IO and API fixes
 */

const path = require('path');
const { spawn } = require('child_process');

console.log('🚀 Starting Ultimate Media Server 2025 - Fixed Dashboard');
console.log('📡 Includes Socket.IO, real-time updates, and complete API endpoints');
console.log('');

// Start the fixed dashboard server
const dashboardPath = path.join(__dirname, 'mcp-architecture', 'src', 'dashboard-server.js');

const dashboard = spawn('node', [dashboardPath], {
  stdio: 'inherit',
  env: {
    ...process.env,
    PORT: '8090',
    NODE_ENV: 'production'
  }
});

dashboard.on('close', (code) => {
  if (code !== 0) {
    console.log(`❌ Dashboard server exited with code ${code}`);
  } else {
    console.log('✅ Dashboard server stopped gracefully');
  }
  process.exit(code);
});

dashboard.on('error', (error) => {
  console.error('❌ Failed to start dashboard server:', error);
  process.exit(1);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down dashboard server...');
  dashboard.kill('SIGINT');
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Received SIGTERM, shutting down dashboard server...');
  dashboard.kill('SIGTERM');
});