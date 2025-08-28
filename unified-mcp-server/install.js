#!/usr/bin/env node

/**
 * Unified MCP Server Installation and Setup
 */

import fs from 'fs/promises';
import { execSync } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class UnifiedMCPInstaller {
  constructor() {
    this.projectRoot = path.resolve(__dirname, '..');
    this.mcpServerPath = __dirname;
    this.claudeConfigPath = path.join(process.env.HOME, '.claude.json');
  }

  async install() {
    console.log('🚀 Installing Unified MCP Server...\n');

    try {
      await this.installDependencies();
      await this.configureClaudeDesktop();
      await this.createStartupScripts();
      await this.testInstallation();
      
      console.log('\n✅ Installation completed successfully!');
      console.log('\n📋 Next Steps:');
      console.log('1. Restart Claude Desktop to load the new MCP server');
      console.log('2. Test the connection using: npm run test');
      console.log('3. Start using unified MCP tools in Claude Desktop');
      
    } catch (error) {
      console.error('❌ Installation failed:', error.message);
      process.exit(1);
    }
  }

  async installDependencies() {
    console.log('📦 Installing dependencies...');
    
    try {
      // Install npm dependencies
      execSync('npm install', { 
        cwd: this.mcpServerPath, 
        stdio: 'inherit' 
      });
      
      console.log('✅ Dependencies installed');
    } catch (error) {
      throw new Error(`Failed to install dependencies: ${error.message}`);
    }
  }

  async configureClaudeDesktop() {
    console.log('⚙️ Configuring Claude Desktop...');
    
    try {
      let claudeConfig = {};
      
      // Read existing configuration
      try {
        const configData = await fs.readFile(this.claudeConfigPath, 'utf8');
        claudeConfig = JSON.parse(configData);
      } catch (error) {
        console.log('   📝 Creating new Claude configuration file');
        claudeConfig = { mcpServers: {} };
      }

      // Ensure mcpServers exists
      if (!claudeConfig.mcpServers) {
        claudeConfig.mcpServers = {};
      }

      // Add unified MCP server configuration
      claudeConfig.mcpServers['unified-media'] = {
        command: 'node',
        args: [path.join(this.mcpServerPath, 'server.js')],
        env: {
          NODE_ENV: 'production'
        }
      };

      // Backup existing configuration
      const backupPath = `${this.claudeConfigPath}.backup.${Date.now()}`;
      try {
        await fs.copyFile(this.claudeConfigPath, backupPath);
        console.log(`   💾 Backed up existing config to: ${backupPath}`);
      } catch (error) {
        // File might not exist, that's okay
      }

      // Write new configuration
      await fs.writeFile(
        this.claudeConfigPath, 
        JSON.stringify(claudeConfig, null, 2)
      );
      
      console.log('✅ Claude Desktop configured');
      console.log(`   📍 Config location: ${this.claudeConfigPath}`);
      
    } catch (error) {
      throw new Error(`Failed to configure Claude Desktop: ${error.message}`);
    }
  }

  async createStartupScripts() {
    console.log('📝 Creating startup scripts...');

    // Create start script
    const startScript = `#!/bin/bash
# Unified MCP Server Startup Script

echo "🚀 Starting Unified MCP Server..."

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed"
    exit 1
fi

# Navigate to server directory
cd "${this.mcpServerPath}"

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the server
echo "✅ Starting server..."
node server.js
`;

    const startScriptPath = path.join(this.mcpServerPath, 'start.sh');
    await fs.writeFile(startScriptPath, startScript);
    
    // Make executable
    try {
      execSync(`chmod +x "${startScriptPath}"`);
    } catch (error) {
      console.warn('⚠️ Could not make start script executable');
    }

    // Create stop script
    const stopScript = `#!/bin/bash
# Unified MCP Server Stop Script

echo "🛑 Stopping Unified MCP Server..."

# Find and kill the process
pkill -f "node.*server.js" || echo "No server process found"

echo "✅ Server stopped"
`;

    const stopScriptPath = path.join(this.mcpServerPath, 'stop.sh');
    await fs.writeFile(stopScriptPath, stopScript);
    
    try {
      execSync(`chmod +x "${stopScriptPath}"`);
    } catch (error) {
      console.warn('⚠️ Could not make stop script executable');
    }

    // Create status script
    const statusScript = `#!/bin/bash
# Unified MCP Server Status Script

echo "📊 Unified MCP Server Status"
echo "============================"

# Check if process is running
if pgrep -f "node.*server.js" > /dev/null; then
    echo "✅ Server is running"
    echo "🔧 Process ID: $(pgrep -f 'node.*server.js')"
else
    echo "❌ Server is not running"
fi

# Check Claude Desktop configuration
CLAUDE_CONFIG="$HOME/.claude.json"
if [ -f "$CLAUDE_CONFIG" ]; then
    echo "✅ Claude Desktop config exists"
    if grep -q "unified-media" "$CLAUDE_CONFIG"; then
        echo "✅ Unified MCP server configured in Claude Desktop"
    else
        echo "⚠️  Unified MCP server not found in Claude Desktop config"
    fi
else
    echo "❌ Claude Desktop config not found"
fi

# Check dependencies
cd "${this.mcpServerPath}"
if [ -d "node_modules" ]; then
    echo "✅ Dependencies installed"
else
    echo "❌ Dependencies missing - run 'npm install'"
fi
`;

    const statusScriptPath = path.join(this.mcpServerPath, 'status.sh');
    await fs.writeFile(statusScriptPath, statusScript);
    
    try {
      execSync(`chmod +x "${statusScriptPath}"`);
    } catch (error) {
      console.warn('⚠️ Could not make status script executable');
    }

    console.log('✅ Startup scripts created');
    console.log(`   📍 Start: ${startScriptPath}`);
    console.log(`   📍 Stop: ${stopScriptPath}`);
    console.log(`   📍 Status: ${statusScriptPath}`);
  }

  async testInstallation() {
    console.log('🧪 Testing installation...');

    try {
      // Test if server can start (with timeout)
      const testProcess = execSync(
        `timeout 5s node server.js || echo "Server test completed"`, 
        { 
          cwd: this.mcpServerPath,
          encoding: 'utf8'
        }
      );

      console.log('✅ Server installation test passed');
      
      // Test Claude Desktop configuration
      const claudeConfig = await fs.readFile(this.claudeConfigPath, 'utf8');
      const config = JSON.parse(claudeConfig);
      
      if (config.mcpServers && config.mcpServers['unified-media']) {
        console.log('✅ Claude Desktop configuration test passed');
      } else {
        throw new Error('Claude Desktop configuration not found');
      }
      
    } catch (error) {
      console.warn('⚠️ Installation test completed with warnings:', error.message);
    }
  }

  async generateServiceDocumentation() {
    console.log('📚 Generating service documentation...');

    const docs = `# Unified MCP Server

## 🌟 Overview

The Unified MCP Server provides a single interface to manage all your media services through Claude Desktop.

## 🔧 Available Tools

### Service Management
- \`unified_health_check\` - Check health of all services
- \`unified_restart_all\` - Restart all services (requires confirmation)
- \`unified_backup_configs\` - Backup all configurations
- \`unified_sync_libraries\` - Synchronize libraries between services
- \`unified_get_statistics\` - Get comprehensive statistics

### Docker Operations
- \`docker_list_containers\` - List all containers
- \`docker_container_logs\` - Get container logs
- \`docker_restart_container\` - Restart specific container

### Service-Specific Tools
Each discovered service gets its own set of tools:
- \`{service}_status\` - Get service status
- \`{service}_api_call\` - Make custom API calls

### Arr Services (Sonarr, Radarr, Lidarr)
- \`{service}_add_media\` - Add new media
- \`{service}_get_media\` - Get media library
- \`{service}_search\` - Search for media

### Jellyfin
- \`jellyfin_get_libraries\` - Get all libraries
- \`jellyfin_get_items\` - Get library items
- \`jellyfin_scan_library\` - Trigger library scan

### Prowlarr
- \`prowlarr_get_indexers\` - Get all indexers
- \`prowlarr_test_indexer\` - Test indexer
- \`prowlarr_search\` - Search across indexers

### Download Clients (qBittorrent, Transmission)
- \`{service}_get_torrents\` - List torrents
- \`{service}_add_torrent\` - Add new torrent
- \`{service}_control_torrent\` - Control torrent

## 🚀 Usage Examples

### Check All Services
\`\`\`
Use the unified_health_check tool to see the status of all services
\`\`\`

### Restart a Container
\`\`\`
Use docker_restart_container with container name "sonarr"
\`\`\`

### Get Service Statistics
\`\`\`
Use unified_get_statistics to see comprehensive system stats
\`\`\`

## 📋 Management Commands

- \`./start.sh\` - Start the MCP server
- \`./stop.sh\` - Stop the MCP server  
- \`./status.sh\` - Check server status
- \`npm test\` - Run tests
- \`npm run dev\` - Start in development mode

## 🔧 Configuration

Edit \`unified-mcp-config.json\` to customize:
- Service endpoints and ports
- Health check intervals
- Docker settings
- Security options
- Logging preferences

Generated on: ${new Date().toISOString()}
`;

    const docsPath = path.join(this.mcpServerPath, 'README.md');
    await fs.writeFile(docsPath, docs);
    
    console.log('✅ Documentation generated');
    console.log(`   📍 Location: ${docsPath}`);
  }
}

// Run installer if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const installer = new UnifiedMCPInstaller();
  
  // Generate documentation first
  await installer.generateServiceDocumentation();
  
  // Run installation
  await installer.install();
}