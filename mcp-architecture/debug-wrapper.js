#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const readline = require('readline');

// Get the actual MCP server to wrap
const serverName = process.argv[2] || 'unknown';
const logFile = path.join('/Users/morlock/fun/newmedia', `mcp-debug-${serverName}-${Date.now()}.log`);

// Create log function
function log(message) {
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] ${message}\n`;
  fs.appendFileSync(logFile, logMessage);
  
  // Also log to stderr (won't interfere with stdout JSON-RPC)
  process.stderr.write(logMessage);
}

log(`Starting MCP debug wrapper for ${serverName}`);
log(`Process ID: ${process.pid}`);
log(`Node version: ${process.version}`);
log(`Working directory: ${process.cwd()}`);
log(`Environment PATH: ${process.env.PATH}`);

// Load the actual MCP server
const serverPath = process.argv[3];
if (!serverPath) {
  log('ERROR: No server path provided');
  process.exit(1);
}

log(`Loading server from: ${serverPath}`);

try {
  // Check if file exists
  if (!fs.existsSync(serverPath)) {
    log(`ERROR: Server file not found: ${serverPath}`);
    process.exit(1);
  }

  // Import and run the server
  require(serverPath);
  log('Server loaded successfully');
} catch (error) {
  log(`ERROR loading server: ${error.message}`);
  log(`Stack trace: ${error.stack}`);
  process.exit(1);
}

// Monitor stdin/stdout
process.stdin.on('data', (data) => {
  log(`STDIN received: ${data.toString().trim()}`);
});

process.stdout.write = (function(write) {
  return function(data) {
    log(`STDOUT sending: ${data.toString().trim()}`);
    return write.apply(process.stdout, arguments);
  };
})(process.stdout.write);

process.on('exit', (code) => {
  log(`Process exiting with code: ${code}`);
});

process.on('uncaughtException', (error) => {
  log(`Uncaught exception: ${error.message}`);
  log(`Stack: ${error.stack}`);
});