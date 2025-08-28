#!/usr/bin/env node

/**
 * Claude Desktop MCP Configuration Verification Script
 * Verifies that the MCP media server is properly configured and working
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

const CONFIG_PATH = path.join(os.homedir(), '.claude', 'claude_desktop_config.json');
const MCP_SERVER_PATH = path.join(__dirname, 'mcp-media-server.js');

function checkFileExists(filePath, description) {
  if (fs.existsSync(filePath)) {
    console.log(`✅ ${description}: ${filePath}`);
    return true;
  } else {
    console.log(`❌ ${description} not found: ${filePath}`);
    return false;
  }
}

function validateConfig() {
  console.log('🔍 Verifying Claude Desktop MCP Configuration...\n');
  
  // Check if config file exists
  if (!checkFileExists(CONFIG_PATH, 'Claude Desktop config')) {
    return false;
  }
  
  // Check if MCP server exists
  if (!checkFileExists(MCP_SERVER_PATH, 'MCP Media Server')) {
    return false;
  }
  
  try {
    // Parse config
    const config = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
    
    if (!config.mcpServers) {
      console.log('❌ No mcpServers section found in config');
      return false;
    }
    
    // Check for our media server
    const mediaServer = config.mcpServers['mcp-media-server'];
    if (!mediaServer) {
      console.log('❌ mcp-media-server not found in config');
      return false;
    }
    
    console.log('✅ mcp-media-server found in config');
    console.log(`   Command: ${mediaServer.command}`);
    console.log(`   Args: ${mediaServer.args.join(' ')}`);
    
    // Check environment variables
    if (mediaServer.env) {
      console.log('\n📊 Environment Variables:');
      Object.entries(mediaServer.env).forEach(([key, value]) => {
        const hasValue = value && value.length > 0;
        const status = hasValue ? '✅' : '⚠️';
        const displayValue = hasValue ? (key.includes('KEY') || key.includes('PASS') ? '[SET]' : value) : '[NOT SET]';
        console.log(`   ${status} ${key}: ${displayValue}`);
      });
    }
    
    // Check other MCP servers
    console.log('\n🔌 Other MCP Servers:');
    Object.keys(config.mcpServers).forEach(serverName => {
      if (serverName !== 'mcp-media-server') {
        console.log(`   ✅ ${serverName}`);
      }
    });
    
    console.log('\n✅ Configuration appears valid!');
    
    // Instructions
    console.log('\n📋 Next Steps:');
    console.log('1. Make sure your media services are running (Sonarr, Radarr, etc.)');
    console.log('2. Restart Claude Desktop to load the new configuration');
    console.log('3. Try using the MCP tools in Claude Desktop');
    console.log('4. Use tools like "get_system_status" to verify connectivity');
    
    return true;
    
  } catch (error) {
    console.log(`❌ Error parsing config: ${error.message}`);
    return false;
  }
}

// Run validation
const isValid = validateConfig();
process.exit(isValid ? 0 : 1);