#!/usr/bin/env node

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🔍 MCP Server Diagnostic Tool for Claude Desktop\n');

// Read Claude config
const configPath = path.join(process.env.HOME, '.claude', 'claude_desktop_config.json');
let config;

try {
  config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  console.log('✅ Found Claude Desktop config\n');
} catch (error) {
  console.error('❌ Error reading config:', error.message);
  process.exit(1);
}

// Test each MCP server
async function testMCPServer(name, serverConfig) {
  console.log(`Testing ${name}...`);
  console.log(`Command: ${serverConfig.command}`);
  console.log(`Args: ${serverConfig.args.join(' ')}`);
  
  return new Promise((resolve) => {
    const child = spawn(serverConfig.command, serverConfig.args, {
      env: { ...process.env, ...serverConfig.env },
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';
    let timeout;

    // Send initialize request
    const request = JSON.stringify({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {}
    }) + '\n';

    child.stdin.write(request);

    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    // Set timeout
    timeout = setTimeout(() => {
      child.kill();
      console.log(`❌ Timeout - no response within 5 seconds`);
      if (stderr) console.log(`Stderr: ${stderr}`);
      console.log('');
      resolve(false);
    }, 5000);

    child.on('error', (error) => {
      clearTimeout(timeout);
      console.log(`❌ Failed to start: ${error.message}`);
      console.log('');
      resolve(false);
    });

    // Check for response
    child.stdout.on('data', (data) => {
      const response = data.toString();
      try {
        const json = JSON.parse(response.trim());
        if (json.result && json.result.serverInfo) {
          clearTimeout(timeout);
          console.log(`✅ Server responded successfully`);
          console.log(`   Name: ${json.result.serverInfo.name}`);
          console.log(`   Version: ${json.result.serverInfo.version}`);
          child.kill();
          console.log('');
          resolve(true);
        }
      } catch (e) {
        // Not JSON, continue waiting
      }
    });

    child.on('exit', (code) => {
      clearTimeout(timeout);
      if (code !== 0 && code !== null) {
        console.log(`❌ Process exited with code ${code}`);
        if (stderr) console.log(`Stderr: ${stderr}`);
        console.log('');
        resolve(false);
      }
    });
  });
}

// Test all servers
async function testAll() {
  const servers = Object.entries(config.mcpServers || {});
  let passed = 0;
  let failed = 0;

  for (const [name, serverConfig] of servers) {
    const success = await testMCPServer(name, serverConfig);
    if (success) passed++;
    else failed++;
  }

  console.log('\n📊 Summary:');
  console.log(`✅ Passed: ${passed}`);
  console.log(`❌ Failed: ${failed}`);
  console.log(`📋 Total: ${servers.length}`);

  // Additional checks
  console.log('\n🔧 System Checks:');
  
  // Check Node.js version
  console.log(`Node.js version: ${process.version}`);
  console.log(`Node.js path: ${process.execPath}`);
  
  // Check if files exist
  console.log('\n📁 File Checks:');
  for (const [name, serverConfig] of servers) {
    const scriptPath = serverConfig.args[0];
    if (fs.existsSync(scriptPath)) {
      console.log(`✅ ${name}: ${scriptPath} exists`);
      // Check if executable
      try {
        fs.accessSync(scriptPath, fs.constants.X_OK);
        console.log(`   ✅ Executable`);
      } catch {
        console.log(`   ❌ Not executable`);
      }
    } else {
      console.log(`❌ ${name}: ${scriptPath} NOT FOUND`);
    }
  }
}

testAll().catch(console.error);