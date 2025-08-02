# 🚀 Advanced .env Configuration Manager 2025

A comprehensive, modern environment variable management interface built with 2025 web standards. This tool provides a complete solution for managing, validating, and deploying environment configurations across development, staging, and production environments.

## ✨ Features

### 🔧 Environment Variable Management
- **Visual Editor**: Advanced code editor with syntax highlighting and line numbers
- **Category Organization**: Organize variables by Database, API Keys, Security, Media Services, Network, and Features
- **Real-time Validation**: Instant validation with security analysis and best practice recommendations
- **Auto-completion**: Smart suggestions for common environment variable patterns

### 🛡️ Security Features
- **Client-side Encryption**: Sensitive values are encrypted using AES-GCM encryption
- **Security Analysis**: Comprehensive security scoring and vulnerability detection
- **Strong Key Generation**: Cryptographically secure key generation for JWT, API keys, passwords, and UUIDs
- **Secret Masking**: Automatic masking and unmasking of sensitive values
- **Audit Logging**: Track all changes and access to environment configurations

### 🎨 Modern UI/UX (2025 Standards)
- **Glass Morphism Design**: Modern glassmorphism effects with backdrop blur
- **Dark/Light Theme**: Seamless theme switching with system preference detection
- **Responsive Design**: Mobile-first design that works on all devices
- **Drag & Drop**: Support for drag-and-drop .env file loading
- **Progressive Web App**: Can be installed as a standalone app
- **Accessibility**: WCAG 2.1 AA compliant with screen reader support

### ⚡ Performance & Integration
- **Real-time Updates**: WebSocket-based real-time synchronization
- **Service Integration**: Monitor and control Docker services directly
- **Auto-save**: Automatic background saving with conflict resolution
- **Backup & Restore**: Automatic backups with one-click restore functionality
- **Template System**: Pre-built templates for common deployment scenarios
- **Bulk Operations**: Import/export multiple configurations at once

### 🤖 AI-Powered Features
- **Smart Suggestions**: AI-powered recommendations for configuration improvements
- **Auto-detection**: Automatically detect required variables per service
- **Dependency Validation**: Validate variable dependencies and relationships
- **Configuration Wizard**: Guided setup for new users with intelligent defaults

### 🔄 DevOps Integration
- **One-click Deployment**: Deploy configurations with automatic service restart
- **Health Monitoring**: Real-time service health checks and status monitoring
- **Configuration Rollback**: Quick rollback to previous configurations
- **CI/CD Integration**: API endpoints for automated deployment pipelines
- **Multi-environment Support**: Manage dev, staging, and production configurations

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm 8+
- Docker (optional, for service management)
- Modern web browser (Chrome 90+, Firefox 88+, Safari 14+)

### Installation

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start the Environment Manager Server**
   ```bash
   npm run env-manager
   ```
   Or for development with auto-restart:
   ```bash
   npm run env-dev
   ```

3. **Open the Dashboard**
   Navigate to: `http://localhost:3001/advanced-env-manager.html`

### Basic Usage

1. **Load Existing Configuration**
   - The interface will automatically load your `.env` file
   - Or drag and drop an existing `.env` file onto the editor

2. **Edit Variables**
   - Use the visual editor with syntax highlighting
   - Switch between categories using the tabs
   - Get real-time validation feedback

3. **Generate Secure Keys**
   - Use the key generator in the sidebar
   - Choose from JWT secrets, API keys, passwords, or UUIDs
   - Keys are cryptographically secure and configurable

4. **Deploy Changes**
   - Click "Deploy Changes" to save and apply
   - Optionally restart services automatically
   - Monitor deployment progress in real-time

## 📋 Configuration Templates

### Available Templates

- **Development**: Full development environment with debug features
- **Production**: Production-ready configuration with security hardening
- **Docker**: Containerized deployment configuration
- **Kubernetes**: Kubernetes-ready environment variables
- **Minimal**: Lightweight configuration for simple applications
- **Full-Stack**: Complete configuration for full-stack applications

### Loading Templates
```javascript
// Via UI
Click on any template tile in the sidebar

// Via API
GET /api/templates/development
```

## 🔐 Security Features

### Encryption
All sensitive values (passwords, secrets, API keys) are automatically encrypted using AES-GCM with a 256-bit key generated using the Web Crypto API.

### Security Analysis
The system performs comprehensive security analysis including:
- Weak or default password detection
- Empty secret validation
- Production configuration validation
- SSL/TLS configuration checks
- CORS and security header validation

### Best Practices
- Use the built-in key generator for all secrets
- Enable validation before deployment
- Use category-specific templates
- Regularly backup configurations
- Monitor security scores and address issues

## 🔌 API Reference

### Environment Operations
```bash
# Load environment file
GET /api/env/load?file=.env

# Save environment file
POST /api/env/save
{
  "content": "NODE_ENV=production\nPORT=3000",
  "backup": true,
  "validate": true,
  "deploy": false
}

# Validate configuration
POST /api/env/validate
{
  "content": "NODE_ENV=development\nPORT=3000"
}
```

### Service Management
```bash
# Get service status
GET /api/services/status

# Control services
POST /api/services/jellyfin/start
POST /api/services/jellyfin/stop
POST /api/services/jellyfin/restart
```

### Security Operations
```bash
# Generate secure key
POST /api/security/generate-key
{
  "type": "jwt",
  "options": { "length": 64 }
}

# Security analysis
POST /api/security/analyze
{
  "content": "JWT_SECRET=weak-secret"
}
```

### Real-time Updates
```javascript
// WebSocket connection
const ws = new WebSocket('ws://localhost:3001/api/env-updates');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Handle real-time updates
};
```

## 🎨 Customization

### Themes
The interface supports both dark and light themes with automatic system preference detection. Themes can be switched using the toggle button or programmatically:

```javascript
// Switch theme
document.documentElement.setAttribute('data-theme', 'light');
```

### Custom Templates
Add custom templates by extending the server's template system:

```javascript
// In env-manager-server.js
const customTemplates = {
  'my-template': `
    NODE_ENV=production
    # Your custom configuration
  `
};
```

### Styling
The interface uses CSS custom properties for easy theming:

```css
:root {
  --primary-hue: 250;        /* Primary color hue */
  --secondary-hue: 320;      /* Secondary color hue */
  --accent-hue: 180;         /* Accent color hue */
}
```

## 🔧 Advanced Configuration

### Server Options
```javascript
const server = new EnvManagerServer({
  port: 3001,                    // Server port
  envFilePath: '.env',           // Default .env file path
  backupDir: './backups',        // Backup directory
  configDir: './config'          // Configuration directory
});
```

### Client Options
```javascript
const api = new EnvManagerAPI({
  baseURL: 'http://localhost:3001/api',
  timeout: 10000,
  retryAttempts: 3,
  encryptionEnabled: true
});
```

### Environment Variables
```bash
# Server configuration
PORT=3001                      # Server port
ENV_FILE_PATH=.env            # Default .env file path
BACKUP_RETENTION_DAYS=30      # Backup retention period
ENABLE_WEBSOCKET=true         # Enable real-time updates
ENABLE_ENCRYPTION=true        # Enable client-side encryption
```

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
EXPOSE 3001

CMD ["npm", "run", "env-manager"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  env-manager:
    build: .
    ports:
      - "3001:3001"
    volumes:
      - ./.env:/app/.env
      - ./backups:/app/backups
    environment:
      - NODE_ENV=production
```

### Production Considerations
- Use a reverse proxy (nginx/Traefik) for SSL termination
- Enable rate limiting for API endpoints
- Configure proper backup and monitoring
- Use environment-specific configuration files
- Enable audit logging for compliance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/your-username/advanced-env-manager.git
cd advanced-env-manager
npm install
npm run env-dev
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern web standards and best practices
- Inspired by the need for better environment variable management
- Uses Three.js for 3D visualizations
- Implements glass morphism design patterns
- Follows accessibility guidelines and progressive web app standards

## 📞 Support

- 📧 Email: support@env-manager.dev
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/advanced-env-manager/issues)
- 📖 Documentation: [Wiki](https://github.com/your-username/advanced-env-manager/wiki)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-username/advanced-env-manager/discussions)

---

**Made with ❤️ for modern web development**

*Secure • Fast • Beautiful • Accessible*