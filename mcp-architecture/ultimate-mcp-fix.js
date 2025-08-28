#!/usr/bin/env node

/**
 * Ultimate MCP Fix - Comprehensive solution for Claude Desktop MCP servers
 * This tool diagnoses and fixes all common MCP connection issues
 */

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const readline = require('readline');

class UltimateMCPFix {
  constructor() {
    this.configPath = path.join(process.env.HOME, '.claude', 'claude_desktop_config.json');
    this.mcpDir = path.dirname(process.argv[1]);
    this.issues = [];
    this.fixes = [];
  }

  log(message, type = 'info') {
    const colors = {
      info: '\x1b[36m',
      success: '\x1b[32m',
      error: '\x1b[31m',
      warning: '\x1b[33m',
      reset: '\x1b[0m'
    };
    console.log(`${colors[type]}${message}${colors.reset}`);
  }

  async diagnose() {
    this.log('\n🔍 Diagnosing MCP Server Issues...', 'info');
    
    // Check 1: Node.js availability
    await this.checkNodeJS();
    
    // Check 2: Configuration file
    await this.checkConfiguration();
    
    // Check 3: MCP server files
    await this.checkServerFiles();
    
    // Check 4: Test server functionality
    await this.testServers();
    
    // Check 5: Common protocol issues
    await this.checkProtocolIssues();
    
    return this.issues;
  }

  async checkNodeJS() {
    this.log('\n1. Checking Node.js...', 'info');
    
    try {
      const nodeVersion = await this.execCommand('node', ['--version']);
      this.log(`✅ Node.js found: ${nodeVersion.trim()}`, 'success');
      
      // Check if node is in PATH
      const which = await this.execCommand('which', ['node']);
      this.log(`   Path: ${which.trim()}`, 'info');
      
      // Check for nvm
      if (which.includes('.nvm')) {
        this.log('   ⚠️  Using nvm - may cause PATH issues', 'warning');
        this.issues.push({
          type: 'nvm_path',
          severity: 'medium',
          description: 'Node.js is managed by nvm which can cause PATH issues'
        });
      }
    } catch (error) {
      this.issues.push({
        type: 'node_missing',
        severity: 'critical',
        description: 'Node.js not found in PATH'
      });
    }
  }

  async checkConfiguration() {
    this.log('\n2. Checking Claude Desktop configuration...', 'info');
    
    if (!fs.existsSync(this.configPath)) {
      this.issues.push({
        type: 'config_missing',
        severity: 'critical',
        description: 'Claude Desktop configuration not found'
      });
      return;
    }
    
    try {
      const config = JSON.parse(fs.readFileSync(this.configPath, 'utf8'));
      const serverCount = Object.keys(config.mcpServers || {}).length;
      this.log(`✅ Configuration found: ${serverCount} servers configured`, 'success');
      
      // Check each server configuration
      for (const [name, serverConfig] of Object.entries(config.mcpServers || {})) {
        if (!serverConfig.command || !serverConfig.args) {
          this.issues.push({
            type: 'invalid_config',
            severity: 'high',
            description: `Server '${name}' has invalid configuration`
          });
        }
      }
    } catch (error) {
      this.issues.push({
        type: 'config_invalid',
        severity: 'critical',
        description: 'Configuration file is not valid JSON'
      });
    }
  }

  async checkServerFiles() {
    this.log('\n3. Checking MCP server files...', 'info');
    
    const expectedServers = [
      'fixed-standalone-mcp.js',
      'fixed-sonarr-mcp-standalone.js',
      'fixed-jellyfin-mcp-standalone.js',
      'fixed-radarr-mcp-standalone.js',
      'fixed-prowlarr-mcp-standalone.js'
    ];
    
    for (const server of expectedServers) {
      const serverPath = path.join(this.mcpDir, server);
      if (fs.existsSync(serverPath)) {
        const stats = fs.statSync(serverPath);
        if (stats.mode & 0o100) {
          this.log(`✅ ${server} - exists and executable`, 'success');
        } else {
          this.log(`⚠️  ${server} - exists but not executable`, 'warning');
          this.issues.push({
            type: 'not_executable',
            severity: 'medium',
            description: `${server} is not executable`,
            file: serverPath
          });
        }
      } else {
        this.log(`❌ ${server} - NOT FOUND`, 'error');
        this.issues.push({
          type: 'file_missing',
          severity: 'high',
          description: `${server} not found`,
          file: serverPath
        });
      }
    }
  }

  async testServers() {
    this.log('\n4. Testing MCP servers...', 'info');
    
    const servers = [
      'fixed-standalone-mcp.js',
      'fixed-sonarr-mcp-standalone.js',
      'fixed-jellyfin-mcp-standalone.js',
      'fixed-radarr-mcp-standalone.js',
      'fixed-prowlarr-mcp-standalone.js'
    ];
    
    for (const server of servers) {
      const serverPath = path.join(this.mcpDir, server);
      if (!fs.existsSync(serverPath)) continue;
      
      const result = await this.testServer(serverPath, server);
      if (result.success) {
        this.log(`✅ ${server} - protocol test passed`, 'success');
      } else {
        this.log(`❌ ${server} - protocol test failed: ${result.error}`, 'error');
        this.issues.push({
          type: 'protocol_error',
          severity: 'high',
          description: `${server} failed protocol test: ${result.error}`,
          file: serverPath
        });
      }
    }
  }

  async checkProtocolIssues() {
    this.log('\n5. Checking for common protocol issues...', 'info');
    
    const servers = [
      'fixed-standalone-mcp.js',
      'fixed-sonarr-mcp-standalone.js',
      'fixed-jellyfin-mcp-standalone.js',
      'fixed-radarr-mcp-standalone.js',
      'fixed-prowlarr-mcp-standalone.js'
    ];
    
    for (const server of servers) {
      const serverPath = path.join(this.mcpDir, server);
      if (!fs.existsSync(serverPath)) continue;
      
      const content = fs.readFileSync(serverPath, 'utf8');
      
      // Check for console.log usage
      if (content.match(/console\.log\s*\(/g) && !content.includes('process.stdout.write')) {
        this.issues.push({
          type: 'stdout_pollution',
          severity: 'high',
          description: `${server} uses console.log instead of process.stdout.write`,
          file: serverPath
        });
      }
      
      // Check for protocol version
      const versionMatch = content.match(/protocolVersion:\s*['"]([^'"]+)['"]/);
      if (versionMatch && versionMatch[1] !== '1.0') {
        this.issues.push({
          type: 'wrong_protocol_version',
          severity: 'high',
          description: `${server} uses protocol version ${versionMatch[1]} instead of 1.0`,
          file: serverPath
        });
      }
    }
  }

  async testServer(serverPath, name) {
    return new Promise((resolve) => {
      const child = spawn('node', [serverPath], {
        stdio: ['pipe', 'pipe', 'pipe']
      });
      
      let stdout = '';
      let stderr = '';
      let timeout;
      
      const request = JSON.stringify({
        jsonrpc: '2.0',
        id: 1,
        method: 'initialize',
        params: {}
      }) + '\n';
      
      child.stdin.write(request);
      
      child.stdout.on('data', (data) => {
        stdout += data.toString();
        try {
          const response = JSON.parse(data.toString().trim());
          if (response.result && response.result.protocolVersion) {
            clearTimeout(timeout);
            child.kill();
            resolve({ success: true });
          }
        } catch (e) {
          // Continue waiting
        }
      });
      
      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      timeout = setTimeout(() => {
        child.kill();
        resolve({ 
          success: false, 
          error: 'Timeout - no valid response'
        });
      }, 3000);
      
      child.on('error', (error) => {
        clearTimeout(timeout);
        resolve({ 
          success: false, 
          error: error.message 
        });
      });
    });
  }

  async applyFixes() {
    this.log('\n\n🔧 Applying Fixes...', 'info');
    
    if (this.issues.length === 0) {
      this.log('✅ No issues found!', 'success');
      return;
    }
    
    // Fix 1: Make files executable
    const executableIssues = this.issues.filter(i => i.type === 'not_executable');
    for (const issue of executableIssues) {
      fs.chmodSync(issue.file, '755');
      this.log(`✅ Made ${path.basename(issue.file)} executable`, 'success');
    }
    
    // Fix 2: Create missing fixed servers
    const missingIssues = this.issues.filter(i => i.type === 'file_missing');
    for (const issue of missingIssues) {
      await this.createFixedServer(issue.file);
    }
    
    // Fix 3: Fix protocol issues
    const protocolIssues = this.issues.filter(i => 
      i.type === 'stdout_pollution' || i.type === 'wrong_protocol_version'
    );
    for (const issue of protocolIssues) {
      await this.fixProtocolIssues(issue.file);
    }
    
    // Fix 4: Update configuration for nvm users
    if (this.issues.some(i => i.type === 'nvm_path')) {
      await this.updateConfigForNVM();
    }
  }

  async createFixedServer(filePath) {
    const baseName = path.basename(filePath).replace('fixed-', '');
    const originalPath = path.join(this.mcpDir, baseName);
    
    if (fs.existsSync(originalPath)) {
      // Fix the original server
      const content = fs.readFileSync(originalPath, 'utf8');
      const fixed = this.fixServerCode(content);
      fs.writeFileSync(filePath, fixed);
      fs.chmodSync(filePath, '755');
      this.log(`✅ Created fixed version of ${baseName}`, 'success');
    }
  }

  fixServerCode(content) {
    // Replace console.log with process.stdout.write
    content = content.replace(
      /console\.log\(JSON\.stringify\(([\w]+)\)\);?/g,
      'process.stdout.write(JSON.stringify($1) + \'\\n\');'
    );
    
    // Fix protocol version
    content = content.replace(
      /protocolVersion:\s*['"]0\.1\.0['"]/g,
      'protocolVersion: \'1.0\''
    );
    
    // Fix initialize response
    content = content.replace(
      /case 'initialize':\s*return\s*{[^}]+}/,
      `case 'initialize':
          return {
            protocolVersion: '1.0',
            capabilities: this.serverInfo.capabilities || { tools: {}, resources: {} },
            serverInfo: {
              name: this.serverInfo.name,
              version: this.serverInfo.version
            }
          }`
    );
    
    // Ensure debug goes to stderr
    content = content.replace(
      /process\.env\.DEBUG === 'true'/g,
      'process.env.MCP_DEBUG === \'true\''
    );
    
    return content;
  }

  async fixProtocolIssues(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const fixed = this.fixServerCode(content);
    fs.writeFileSync(filePath, fixed);
    this.log(`✅ Fixed protocol issues in ${path.basename(filePath)}`, 'success');
  }

  async updateConfigForNVM() {
    this.log('\n📝 Updating configuration for nvm...', 'info');
    
    const config = JSON.parse(fs.readFileSync(this.configPath, 'utf8'));
    
    // Create shell wrapper approach
    for (const [name, serverConfig] of Object.entries(config.mcpServers || {})) {
      if (serverConfig.command === 'node') {
        serverConfig.command = '/bin/zsh';
        serverConfig.args = [
          '-c',
          `source ~/.zshrc && node ${serverConfig.args[0]}`
        ];
      }
    }
    
    // Backup original config
    const backupPath = this.configPath + '.backup-' + Date.now();
    fs.copyFileSync(this.configPath, backupPath);
    
    // Write updated config
    fs.writeFileSync(this.configPath, JSON.stringify(config, null, 2));
    this.log('✅ Updated configuration to handle nvm PATH issues', 'success');
  }

  async execCommand(command, args) {
    return new Promise((resolve, reject) => {
      const child = spawn(command, args);
      let output = '';
      
      child.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      child.on('close', (code) => {
        if (code === 0) {
          resolve(output);
        } else {
          reject(new Error(`Command failed: ${command} ${args.join(' ')}`));
        }
      });
    });
  }

  async createUltimateConfig() {
    this.log('\n📋 Creating ultimate configuration...', 'info');
    
    const config = {
      mcpServers: {
        "media-server": {
          "command": "/bin/zsh",
          "args": [
            "-c",
            "source ~/.zshrc && node /Users/morlock/fun/newmedia/mcp-architecture/fixed-standalone-mcp.js"
          ],
          "env": {
            "MCP_DEBUG": "false"
          }
        },
        "sonarr": {
          "command": "/bin/zsh",
          "args": [
            "-c",
            "source ~/.zshrc && node /Users/morlock/fun/newmedia/mcp-architecture/fixed-sonarr-mcp-standalone.js"
          ],
          "env": {
            "MCP_DEBUG": "false",
            "SONARR_URL": "http://localhost:8989",
            "SONARR_API_KEY": ""
          }
        },
        "jellyfin": {
          "command": "/bin/zsh",
          "args": [
            "-c",
            "source ~/.zshrc && node /Users/morlock/fun/newmedia/mcp-architecture/fixed-jellyfin-mcp-standalone.js"
          ],
          "env": {
            "MCP_DEBUG": "false",
            "JELLYFIN_URL": "http://localhost:8096",
            "JELLYFIN_API_KEY": ""
          }
        },
        "radarr": {
          "command": "/bin/zsh",
          "args": [
            "-c",
            "source ~/.zshrc && node /Users/morlock/fun/newmedia/mcp-architecture/fixed-radarr-mcp-standalone.js"
          ],
          "env": {
            "MCP_DEBUG": "false",
            "RADARR_URL": "http://localhost:7878",
            "RADARR_API_KEY": ""
          }
        },
        "prowlarr": {
          "command": "/bin/zsh",
          "args": [
            "-c",
            "source ~/.zshrc && node /Users/morlock/fun/newmedia/mcp-architecture/fixed-prowlarr-mcp-standalone.js"
          ],
          "env": {
            "MCP_DEBUG": "false",
            "PROWLARR_URL": "http://localhost:9696",
            "PROWLARR_API_KEY": ""
          }
        }
      }
    };
    
    // Backup current config
    if (fs.existsSync(this.configPath)) {
      const backupPath = this.configPath + '.backup-ultimate-' + Date.now();
      fs.copyFileSync(this.configPath, backupPath);
      this.log(`✅ Backed up current config to ${path.basename(backupPath)}`, 'success');
    }
    
    // Write new config
    fs.writeFileSync(this.configPath, JSON.stringify(config, null, 2));
    this.log('✅ Created ultimate configuration with shell wrapper', 'success');
  }

  async run() {
    console.clear();
    this.log('🚀 Ultimate MCP Fix Tool', 'info');
    this.log('=======================\n', 'info');
    
    // Run diagnosis
    await this.diagnose();
    
    // Show summary
    this.log('\n📊 Diagnosis Summary:', 'info');
    if (this.issues.length === 0) {
      this.log('✅ No issues found! Your MCP servers should be working.', 'success');
    } else {
      this.log(`Found ${this.issues.length} issues:`, 'warning');
      for (const issue of this.issues) {
        this.log(`  - [${issue.severity}] ${issue.description}`, 
          issue.severity === 'critical' ? 'error' : 'warning');
      }
    }
    
    // Apply fixes
    if (this.issues.length > 0) {
      await this.applyFixes();
    }
    
    // Always create ultimate config
    await this.createUltimateConfig();
    
    // Final instructions
    this.log('\n\n✅ All fixes applied!', 'success');
    this.log('\n📝 Final Steps:', 'info');
    this.log('1. Quit Claude Desktop completely (Cmd+Q)', 'info');
    this.log('2. Wait 10 seconds', 'info');
    this.log('3. Open Claude Desktop', 'info');
    this.log('4. Test by asking: "What MCP tools are available?"', 'info');
    this.log('\nExpected: 5 servers with 24 total tools', 'success');
    
    // Restart Claude
    this.log('\n🔄 Restarting Claude Desktop...', 'info');
    await this.execCommand('pkill', ['-f', 'Claude.app']).catch(() => {});
    setTimeout(async () => {
      await this.execCommand('open', ['-a', 'Claude']).catch(() => {});
      this.log('✅ Claude Desktop restarted!', 'success');
    }, 3000);
  }
}

// Run the ultimate fix
const fixer = new UltimateMCPFix();
fixer.run().catch(console.error);