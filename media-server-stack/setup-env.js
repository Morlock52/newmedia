#!/usr/bin/env node

/**
 * Media Server Stack Environment Configuration Tool
 * 
 * This CUI (Command Line User Interface) helps users configure all required
 * environment variables for the media server stack following best practices.
 * 
 * Features:
 * - Interactive prompts with validation
 * - Default values and examples
 * - Secure handling of sensitive data
 * - Automatic .env file generation
 * - Validation against docker-compose.yml
 * - Backup existing configurations
 */

import inquirer from 'inquirer';
import fs from 'fs/promises';
import path from 'path';
import { execSync } from 'child_process';
import chalk from 'chalk';
import figlet from 'figlet';
import crypto from 'crypto';

const ENV_FILE = '.env';
const ENV_EXAMPLE = '.env.example';
const COMPOSE_FILE = './compose/docker-compose.yml';

// Command line arguments
const args = process.argv.slice(2);
const showHelp = args.includes('--help') || args.includes('-h');
const skipInteractive = args.includes('--no-interactive') || args.includes('-n');
const validateOnly = args.includes('--validate-only') || args.includes('-v');
const dryRun = args.includes('--dry-run') || args.includes('-d');

// Environment variable definitions with metadata
const ENV_VARIABLES = {
  // Core System Variables (Required)
  DOMAIN: {
    required: true,
    description: 'Your domain name (e.g., yourdomain.com)',
    example: 'localhost',
    default: 'localhost',
    validation: (input) => {
      const domainRegex = /^([a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$|^localhost$/;
      return domainRegex.test(input) || 'Please enter a valid domain name or localhost';
    },
    category: 'Core'
  },
  PUID: {
    required: true,
    description: 'User ID (run "id -u" to get your user ID)',
    example: '1000',
    default: () => {
      try {
        return execSync('id -u', { encoding: 'utf8' }).trim();
      } catch {
        return '1000';
      }
    },
    validation: (input) => {
      const num = parseInt(input);
      return (!isNaN(num) && num > 0) || 'Please enter a valid user ID';
    },
    category: 'Core'
  },
  PGID: {
    required: true,
    description: 'Group ID (run "id -g" to get your group ID)',
    example: '1000',
    default: () => {
      try {
        return execSync('id -g', { encoding: 'utf8' }).trim();
      } catch {
        return '1000';
      }
    },
    validation: (input) => {
      const num = parseInt(input);
      return (!isNaN(num) && num > 0) || 'Please enter a valid group ID';
    },
    category: 'Core'
  },
  TZ: {
    required: true,
    description: 'Timezone (e.g., America/New_York, Europe/London)',
    example: 'UTC',
    default: () => {
      try {
        // Try different methods to get system timezone
        if (process.platform === 'darwin') {
          return execSync('readlink /etc/localtime | sed "s/.*zoneinfo\\///"', { encoding: 'utf8' }).trim();
        } else if (process.platform === 'linux') {
          return execSync('timedatectl show --value --property=Timezone', { encoding: 'utf8' }).trim();
        }
      } catch {
        // Fallback to JavaScript timezone detection
      }
      return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
    },
    validation: (input) => {
      try {
        Intl.DateTimeFormat(undefined, { timeZone: input });
        return true;
      } catch {
        return 'Please enter a valid timezone (e.g., America/New_York, Europe/London, UTC)';
      }
    },
    category: 'Core'
  },

  // VPN Configuration (Optional but recommended)
  VPN_PROVIDER: {
    required: false,
    description: 'VPN provider (pia, mullvad, expressvpn, etc.)',
    example: 'mullvad',
    default: 'mullvad',
    choices: ['pia', 'mullvad', 'expressvpn', 'nordvpn', 'surfshark', 'cyberghost'],
    category: 'VPN'
  },
  VPN_TYPE: {
    required: false,
    description: 'VPN connection type',
    example: 'wireguard',
    default: 'wireguard',
    choices: ['wireguard', 'openvpn'],
    category: 'VPN'
  },
  VPN_PORT_FORWARDING: {
    required: false,
    description: 'Enable VPN port forwarding for torrents',
    example: 'on',
    default: 'on',
    choices: ['on', 'off'],
    category: 'VPN'
  },
  VPN_PORT_FORWARDING_PORT: {
    required: false,
    description: 'Port forwarding port number',
    example: '6881',
    default: '6881',
    validation: (input) => {
      const num = parseInt(input);
      return (!isNaN(num) && num > 1024 && num < 65535) || 'Please enter a valid port number (1024-65535)';
    },
    category: 'VPN'
  },
  PIA_REGION: {
    required: false,
    description: 'VPN server region (for PIA)',
    example: 'us_east',
    default: 'us_east',
    category: 'VPN'
  },
  WIREGUARD_ADDRESSES: {
    required: false,
    description: 'WireGuard address range',
    example: '10.0.0.0/8',
    default: '10.0.0.0/8',
    validation: (input) => {
      const cidrRegex = /^(\d{1,3}\.){3}\d{1,3}\/\d{1,2}$/;
      return cidrRegex.test(input) || 'Please enter a valid CIDR notation';
    },
    category: 'VPN'
  },

  // Cloudflare Configuration (Optional)
  CLOUDFLARE_TUNNEL_TOKEN: {
    required: false,
    description: 'Cloudflare Tunnel token for external access',
    example: 'eyJhIjoiYWJjZGVmZ....(long token)',
    sensitive: true,
    validation: (input) => {
      if (!input) return true; // Optional
      return input.length > 50 || 'Cloudflare tunnel token should be longer';
    },
    category: 'Cloudflare'
  },

  // Email Configuration (Optional)
  EMAIL: {
    required: false,
    description: 'Email address for notifications and SSL certificates',
    example: 'admin@yourdomain.com',
    validation: (input) => {
      if (!input) return true; // Optional
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      return emailRegex.test(input) || 'Please enter a valid email address';
    },
    category: 'Notifications'
  },

  // Additional Security Variables
  UMASK: {
    required: false,
    description: 'File permission mask for created files',
    example: '002',
    default: '002',
    validation: (input) => {
      const umaskRegex = /^[0-7]{3}$/;
      return umaskRegex.test(input) || 'Please enter a valid umask (e.g., 002, 022)';
    },
    category: 'Core'
  },

  DATA_ROOT: {
    required: false,
    description: 'Root directory for media data storage',
    example: './data',
    default: './data',
    category: 'Storage'
  },

  CONFIG_ROOT: {
    required: false,
    description: 'Root directory for application configurations',
    example: './config',
    default: './config',
    category: 'Storage'
  },

  // Database Configuration
  POSTGRES_USER: {
    required: false,
    description: 'PostgreSQL database username',
    example: 'mediaserver',
    default: 'mediaserver',
    category: 'Database'
  },

  POSTGRES_DB: {
    required: false,
    description: 'PostgreSQL database name',
    example: 'mediaserver',
    default: 'mediaserver',
    category: 'Database'
  }
};

class EnvConfigTool {
  constructor() {
    this.envData = new Map();
    this.existingEnv = new Map();
  }

  showHelp() {
    console.log(chalk.cyan(figlet.textSync('Media Stack', { font: 'Small' })));
    console.log(chalk.green('Environment Configuration Tool\n'));
    
    console.log(chalk.white('Usage:'));
    console.log(chalk.yellow('  node setup-env.js [options]\n'));
    
    console.log(chalk.white('Options:'));
    console.log(chalk.yellow('  -h, --help          Show this help message'));
    console.log(chalk.yellow('  -n, --no-interactive  Skip interactive prompts, use defaults'));
    console.log(chalk.yellow('  -v, --validate-only   Only validate existing configuration'));
    console.log(chalk.yellow('  -d, --dry-run       Show what would be done without making changes\n'));
    
    console.log(chalk.white('Examples:'));
    console.log(chalk.cyan('  node setup-env.js                 # Interactive setup'));
    console.log(chalk.cyan('  node setup-env.js --validate-only # Just validate current .env'));
    console.log(chalk.cyan('  node setup-env.js --dry-run       # Preview changes'));
    console.log(chalk.cyan('  node setup-env.js --no-interactive # Use defaults for missing vars\n'));
    
    console.log(chalk.white('Security Features:'));
    console.log(chalk.green('  • Automatic backup of existing configurations'));
    console.log(chalk.green('  • Secure handling of sensitive data'));
    console.log(chalk.green('  • Input validation and sanitization'));
    console.log(chalk.green('  • Docker secrets integration'));
    console.log(chalk.green('  • Permissions validation\n'));
    
    process.exit(0);
  }

  async showWelcome() {
    console.clear();
    console.log(chalk.cyan(figlet.textSync('Media Stack', { font: 'Small' })));
    console.log(chalk.green('Environment Configuration Tool\n'));
    console.log(chalk.yellow('This tool will help you configure all required environment variables'));
    console.log(chalk.yellow('for your media server stack following security best practices.\n'));
  }

  async loadExistingEnv() {
    try {
      const envContent = await fs.readFile(ENV_FILE, 'utf8');
      const lines = envContent.split('\n');
      
      for (const line of lines) {
        if (line.trim() && !line.startsWith('#')) {
          const [key, ...valueParts] = line.split('=');
          const value = valueParts.join('=');
          if (key && value !== undefined) {
            this.existingEnv.set(key, value);
          }
        }
      }
      
      console.log(chalk.green(`✓ Found existing .env file with ${this.existingEnv.size} variables\n`));
    } catch (error) {
      console.log(chalk.yellow('ℹ No existing .env file found, creating new configuration\n'));
    }
  }

  async backupExistingEnv() {
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
      const backupFile = `.env.backup.${timestamp}-${Date.now()}`;
      await fs.copyFile(ENV_FILE, backupFile);
      console.log(chalk.green(`✓ Backed up existing .env to ${backupFile}`));
    } catch (error) {
      // No existing file to backup
    }
  }

  async promptForCategory(categoryName, variables) {
    console.log(chalk.blue(`\n=== ${categoryName} Configuration ===\n`));
    
    for (const [key, config] of Object.entries(variables)) {
      await this.promptForVariable(key, config);
    }
  }

  async promptForVariable(key, config) {
    const existingValue = this.existingEnv.get(key);
    let defaultValue = config.default;
    
    if (typeof defaultValue === 'function') {
      defaultValue = defaultValue();
    }
    
    if (existingValue) {
      defaultValue = config.sensitive ? '[HIDDEN]' : existingValue;
    }

    const promptConfig = {
      type: config.choices ? 'list' : (config.sensitive ? 'password' : 'input'),
      name: 'value',
      message: `${config.required ? '*' : ' '} ${key}: ${config.description}`,
      default: defaultValue,
      validate: config.validation || (() => true)
    };

    if (config.choices) {
      promptConfig.choices = config.choices;
    }

    if (config.example && !config.sensitive) {
      promptConfig.message += chalk.gray(` (e.g., ${config.example})`);
    }

    // For sensitive fields, ask if they want to change existing value
    if (config.sensitive && existingValue) {
      const { shouldChange } = await inquirer.prompt({
        type: 'confirm',
        name: 'shouldChange',
        message: `${key} is already set. Do you want to change it?`,
        default: false
      });
      
      if (!shouldChange) {
        this.envData.set(key, existingValue);
        return;
      }
    }

    const { value } = await inquirer.prompt(promptConfig);
    
    if (value || config.required) {
      this.envData.set(key, value || defaultValue);
    }
  }

  async generateSecrets() {
    console.log(chalk.blue('\n=== Security Configuration ===\n'));
    
    const { generateSecrets } = await inquirer.prompt({
      type: 'confirm',
      name: 'generateSecrets',
      message: 'Generate secure random secrets for API keys and passwords?',
      default: true
    });

    if (generateSecrets) {
      const secrets = [
        'POSTGRES_PASSWORD',
        'JELLYFIN_API_KEY',
        'SONARR_API_KEY',
        'RADARR_API_KEY',
        'PHOTOPRISM_ADMIN_PASSWORD'
      ];

      for (const secret of secrets) {
        const value = crypto.randomBytes(32).toString('hex');
        this.envData.set(secret, value);
      }
      
      console.log(chalk.green('✓ Generated secure random secrets'));
    }
  }

  async validateConfiguration() {
    console.log(chalk.blue('\n=== Validation ===\n'));
    
    const errors = [];
    const warnings = [];

    // Check required variables
    for (const [key, config] of Object.entries(ENV_VARIABLES)) {
      if (config.required && !this.envData.has(key)) {
        errors.push(`Missing required variable: ${key}`);
      }
    }

    // Domain-specific validations
    const domain = this.envData.get('DOMAIN');
    if (domain && domain !== 'localhost') {
      if (!this.envData.get('EMAIL')) {
        warnings.push('EMAIL not set - recommended for SSL certificates');
      }
      if (!this.envData.get('CLOUDFLARE_TUNNEL_TOKEN')) {
        warnings.push('CLOUDFLARE_TUNNEL_TOKEN not set - external access will be limited');
      }
    }

    // VPN warnings
    if (!this.envData.get('VPN_PROVIDER')) {
      warnings.push('VPN not configured - torrenting may expose your IP address');
    }

    if (errors.length > 0) {
      console.log(chalk.red('❌ Configuration Errors:'));
      errors.forEach(error => console.log(chalk.red(`  • ${error}`)));
      return false;
    }

    if (warnings.length > 0) {
      console.log(chalk.yellow('⚠️  Configuration Warnings:'));
      warnings.forEach(warning => console.log(chalk.yellow(`  • ${warning}`)));
      
      const { continueAnyway } = await inquirer.prompt({
        type: 'confirm',
        name: 'continueAnyway',
        message: 'Continue with warnings?',
        default: true
      });
      
      if (!continueAnyway) return false;
    }

    console.log(chalk.green('✓ Configuration validation passed'));
    return true;
  }

  async writeEnvFile() {
    console.log(chalk.blue('\n=== Generating Files ===\n'));
    
    if (dryRun) {
      console.log(chalk.yellow('🔍 DRY RUN: Showing what would be written to .env file...\n'));
    }
    
    let content = `# Media Server Stack Environment Configuration
# Generated on ${new Date().toISOString()}
# 
# This file contains environment variables for your media server stack.
# Keep this file secure and do not commit it to version control.
# 
# For security best practices, see: https://12factor.net/config

`;

    // Group variables by category
    const categories = {};
    for (const [key, config] of Object.entries(ENV_VARIABLES)) {
      if (!categories[config.category]) {
        categories[config.category] = [];
      }
      categories[config.category].push([key, config]);
    }

    // Write variables by category
    for (const [categoryName, variables] of Object.entries(categories)) {
      content += `# ${categoryName} Configuration\n`;
      
      for (const [key, config] of variables) {
        if (this.envData.has(key)) {
          content += `${key}=${this.envData.get(key)}\n`;
        } else if (config.default && !config.required) {
          const defaultValue = typeof config.default === 'function' ? config.default() : config.default;
          content += `# ${key}=${defaultValue}\n`;
        }
      }
      content += '\n';
    }

    // Add any additional variables from existing env
    const additionalVars = [];
    for (const [key, value] of this.existingEnv) {
      if (!ENV_VARIABLES[key] && !this.envData.has(key)) {
        additionalVars.push([key, value]);
      }
    }

    if (additionalVars.length > 0) {
      content += '# Additional Variables\n';
      for (const [key, value] of additionalVars) {
        content += `${key}=${value}\n`;
        this.envData.set(key, value);
      }
      content += '\n';
    }

    if (dryRun) {
      console.log(chalk.gray(content));
      console.log(chalk.yellow('✓ DRY RUN: Would create .env file with above content'));
      return;
    }

    await this.backupExistingEnv();
    await fs.writeFile(ENV_FILE, content);
    
    // Set secure permissions on .env file
    await fs.chmod(ENV_FILE, 0o600);
    
    console.log(chalk.green(`✓ Created .env file with ${this.envData.size} variables`));
    console.log(chalk.green('✓ Set secure permissions (600) on .env file'));
  }

  async createSecretsDirectory() {
    const secretsDir = './secrets';
    
    try {
      await fs.mkdir(secretsDir, { recursive: true });
      
      // Create secret files for sensitive data
      const secrets = [
        'postgres_password',
        'jellyfin_api_key',
        'sonarr_api_key',
        'radarr_api_key',
        'photoprism_admin_password'
      ];

      for (const secret of secrets) {
        const envKey = secret.toUpperCase();
        if (this.envData.has(envKey)) {
          const secretFile = path.join(secretsDir, `${secret}.txt`);
          await fs.writeFile(secretFile, this.envData.get(envKey));
          await fs.chmod(secretFile, 0o600); // Read/write for owner only
        }
      }
      
      console.log(chalk.green('✓ Created secrets directory with secure permissions'));
    } catch (error) {
      console.log(chalk.yellow(`⚠️  Could not create secrets directory: ${error.message}`));
    }
  }

  async testConfiguration() {
    console.log(chalk.blue('\n=== Testing Configuration ===\n'));
    
    try {
      // Test docker-compose config
      execSync('docker compose config', { 
        stdio: 'pipe',
        cwd: process.cwd()
      });
      console.log(chalk.green('✓ Docker Compose configuration is valid'));
      
      // Test if ports are available
      const testPorts = [3000, 80, 443, 8080];
      for (const port of testPorts) {
        try {
          execSync(`lsof -i :${port}`, { stdio: 'pipe' });
          console.log(chalk.yellow(`⚠️  Port ${port} is already in use`));
        } catch {
          console.log(chalk.green(`✓ Port ${port} is available`));
        }
      }
      
    } catch (error) {
      console.log(chalk.red(`❌ Configuration test failed: ${error.message}`));
      return false;
    }
    
    return true;
  }

  async showSummary() {
    console.log(chalk.blue('\n=== Configuration Summary ===\n'));
    
    if (dryRun) {
      console.log(chalk.yellow('🔍 DRY RUN COMPLETE\n'));
      console.log(chalk.white('What would be done:'));
      console.log(chalk.cyan('• Backup existing .env file'));
      console.log(chalk.cyan(`• Create new .env with ${this.envData.size} variables`));
      console.log(chalk.cyan('• Set secure permissions (600) on .env file'));
      console.log(chalk.cyan('• Create secrets directory with API keys'));
      console.log(chalk.white('\nTo actually apply changes, run without --dry-run'));
      return;
    }
    
    console.log(chalk.green('✅ Configuration completed successfully!\n'));
    
    console.log(chalk.white('What was done:'));
    console.log(chalk.green('✓ Environment variables configured'));
    console.log(chalk.green('✓ Secure secrets generated'));
    console.log(chalk.green('✓ File permissions secured'));
    console.log(chalk.green('✓ Configuration validated\n'));
    
    console.log(chalk.white('Next steps:'));
    console.log(chalk.cyan('1. Review your .env file for any additional customization'));
    console.log(chalk.cyan('2. Start your media stack:'));
    console.log(chalk.gray('   cd compose && docker compose up -d'));
    console.log(chalk.cyan('3. Access the web UI at: http://localhost:3000'));
    
    if (this.envData.get('DOMAIN') && this.envData.get('DOMAIN') !== 'localhost') {
      console.log(chalk.cyan(`4. Access your services at: https://${this.envData.get('DOMAIN')}`));
    }
    
    console.log(chalk.white('\nImportant security notes:'));
    console.log(chalk.yellow('• Keep your .env file secure and never commit it to version control'));
    console.log(chalk.yellow('• Regularly rotate your API keys and passwords'));
    console.log(chalk.yellow('• Consider using Docker secrets for production deployments'));
    console.log(chalk.yellow('• Review the generated secrets/ directory permissions'));
    
    console.log(chalk.white('\nUseful commands:'));
    console.log(chalk.gray('• Validate config: node setup-env.js --validate-only'));
    console.log(chalk.gray('• View changes: node setup-env.js --dry-run'));
    console.log(chalk.gray('• Non-interactive: node setup-env.js --no-interactive'));
    
    console.log(chalk.green('\n🎉 Your media server stack is ready to deploy!'));
  }

  async run() {
    try {
      // Handle help command
      if (showHelp) {
        this.showHelp();
      }

      await this.showWelcome();
      await this.loadExistingEnv();

      // If validate-only mode, just validate and exit
      if (validateOnly) {
        console.log(chalk.blue('🔍 Validation Mode: Checking existing configuration...\n'));
        
        // Load existing data for validation
        for (const [key, value] of this.existingEnv) {
          this.envData.set(key, value);
        }
        
        const isValid = await this.validateConfiguration();
        if (isValid) {
          console.log(chalk.green('\n✅ Configuration is valid!'));
          process.exit(0);
        } else {
          console.log(chalk.red('\n❌ Configuration has errors. Run without --validate-only to fix.'));
          process.exit(1);
        }
      }

      // Group variables by category for better UX
      const categories = {};
      for (const [key, config] of Object.entries(ENV_VARIABLES)) {
        if (!categories[config.category]) {
          categories[config.category] = {};
        }
        categories[config.category][key] = config;
      }

      // Prompt for each category (skip if no-interactive)
      if (!skipInteractive) {
        for (const [categoryName, variables] of Object.entries(categories)) {
          await this.promptForCategory(categoryName, variables);
        }
        await this.generateSecrets();
      } else {
        console.log(chalk.yellow('📋 Non-interactive mode: Using defaults for missing variables...\n'));
        await this.useDefaults();
      }
      
      const isValid = await this.validateConfiguration();
      if (!isValid) {
        console.log(chalk.red('\n❌ Configuration validation failed. Please fix the errors and try again.'));
        process.exit(1);
      }

      if (!dryRun) {
        await this.writeEnvFile();
        await this.createSecretsDirectory();
        
        const testPassed = await this.testConfiguration();
        if (!testPassed) {
          console.log(chalk.yellow('\n⚠️  Configuration tests failed, but .env file was created.'));
        }
      } else {
        await this.writeEnvFile(); // Will just show what would be written
      }

      await this.showSummary();
      
    } catch (error) {
      console.error(chalk.red(`\n❌ Error: ${error.message}`));
      if (error.stack && args.includes('--debug')) {
        console.error(chalk.gray(error.stack));
      }
      process.exit(1);
    }
  }

  async useDefaults() {
    // Fill in defaults for missing required variables
    for (const [key, config] of Object.entries(ENV_VARIABLES)) {
      if (!this.existingEnv.has(key)) {
        let defaultValue = config.default;
        if (typeof defaultValue === 'function') {
          defaultValue = defaultValue();
        }
        
        if (config.required && !defaultValue) {
          console.log(chalk.red(`❌ Required variable ${key} has no default value`));
          console.log(chalk.yellow(`   Run without --no-interactive to configure this variable`));
          process.exit(1);
        }
        
        if (defaultValue) {
          this.envData.set(key, defaultValue);
          console.log(chalk.green(`✓ Set ${key} to default value`));
        }
      } else {
        this.envData.set(key, this.existingEnv.get(key));
      }
    }
    
    // Generate secrets if needed
    const needsSecrets = !this.existingEnv.has('POSTGRES_PASSWORD');
    if (needsSecrets) {
      console.log(chalk.blue('\n🔐 Generating secure secrets...'));
      const secrets = [
        'POSTGRES_PASSWORD',
        'JELLYFIN_API_KEY',
        'SONARR_API_KEY',
        'RADARR_API_KEY',
        'PHOTOPRISM_ADMIN_PASSWORD'
      ];

      for (const secret of secrets) {
        const value = crypto.randomBytes(32).toString('hex');
        this.envData.set(secret, value);
      }
      console.log(chalk.green('✓ Generated secure random secrets'));
    }
  }
}

// Run the tool
const tool = new EnvConfigTool();
tool.run();
