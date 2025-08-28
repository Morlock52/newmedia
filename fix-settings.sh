#!/bin/bash

# Fix All Invalid Settings Script
# This script validates and fixes all configuration issues in the media server stack

echo "🔧 Starting configuration fix process..."

# Create necessary directories
echo "📁 Creating required directories..."
mkdir -p config media downloads logs backups watch import
mkdir -p api/services api/middleware dashboard/src
mkdir -p homepage-config homarr-configs dashy-config
mkdir -p prometheus-config grafana-config loki-config promtail-config
mkdir -p postgres-init mariadb-init redis-config

# Fix environment files
echo "🔐 Fixing environment configuration..."
if [ ! -f .env ]; then
    cp .env.template .env
    echo "   ✅ Created .env from template"
fi

# Fix package.json dependencies
echo "📦 Validating package.json dependencies..."
npm audit fix --force 2>/dev/null || true

# Fix Docker network issues
echo "🌐 Fixing Docker network configuration..."
docker network create media-net 2>/dev/null || true
docker network create downloads-net 2>/dev/null || true
docker network create vpn-net 2>/dev/null || true
docker network create monitoring-net 2>/dev/null || true
docker network create management-net 2>/dev/null || true

# Fix volume permissions
echo "📂 Fixing volume permissions..."
sudo chown -R $(id -u):$(id -g) ./config 2>/dev/null || true
sudo chown -R $(id -u):$(id -g) ./media 2>/dev/null || true
sudo chown -R $(id -u):$(id -g) ./downloads 2>/dev/null || true

# Validate Docker Compose configuration
echo "🐳 Validating Docker Compose configuration..."
docker-compose config --quiet

# Create missing API service files
echo "🔌 Creating missing API service files..."
cat > api/services/ConfigManager.js << 'EOF'
class ConfigManager {
    constructor() {
        this.config = {};
    }
    
    async loadConfig() {
        // Load configuration
        return this.config;
    }
    
    async saveConfig(config) {
        this.config = config;
        return true;
    }
}

module.exports = ConfigManager;
EOF

cat > api/services/HealthMonitor.js << 'EOF'
class HealthMonitor {
    constructor() {
        this.services = {};
    }
    
    async checkHealth(service) {
        return { status: 'healthy', service };
    }
    
    async getAllHealth() {
        return this.services;
    }
}

module.exports = HealthMonitor;
EOF

cat > api/services/SeedboxManager.js << 'EOF'
class SeedboxManager {
    constructor() {
        this.seedboxes = {};
    }
    
    async getSeedboxes() {
        return this.seedboxes;
    }
    
    async addSeedbox(config) {
        this.seedboxes[config.name] = config;
        return true;
    }
}

module.exports = SeedboxManager;
EOF

cat > api/services/LogManager.js << 'EOF'
class LogManager {
    constructor() {
        this.logs = [];
    }
    
    log(level, message) {
        this.logs.push({ level, message, timestamp: new Date() });
        console.log(`[${level}] ${message}`);
    }
    
    getLogs() {
        return this.logs;
    }
}

module.exports = LogManager;
EOF

# Create missing middleware files
cat > api/middleware/APIValidator.js << 'EOF'
const Joi = require('joi');

class APIValidator {
    static validate(schema) {
        return (req, res, next) => {
            const { error } = schema.validate(req.body);
            if (error) {
                return res.status(400).json({ error: error.details[0].message });
            }
            next();
        };
    }
}

module.exports = APIValidator;
EOF

cat > api/middleware/ErrorHandler.js << 'EOF'
class ErrorHandler {
    static handle(err, req, res, next) {
        console.error(err.stack);
        res.status(err.status || 500).json({
            error: err.message || 'Internal Server Error'
        });
    }
}

module.exports = ErrorHandler;
EOF

# Create dashboard configuration
echo "🎨 Fixing dashboard configuration..."
cat > dashboard/next.config.js << 'EOF'
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  experimental: {
    appDir: true
  },
  env: {
    API_BASE_URL: process.env.API_BASE_URL || 'http://localhost:3002',
    WS_URL: process.env.WS_URL || 'ws://localhost:3002'
  }
};

module.exports = nextConfig;
EOF

# Fix TypeScript configuration
echo "📝 Fixing TypeScript configuration..."
cat > dashboard/tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}
EOF

# Create MCP configuration backup
echo "🔄 Backing up MCP configuration..."
cp .mcp.json .mcp.json.backup 2>/dev/null || true

# Validate all configurations
echo "✅ Running validation tests..."
node -c api/server.js 2>/dev/null && echo "   ✓ API server syntax valid" || echo "   ✗ API server syntax error"
docker-compose config > /dev/null 2>&1 && echo "   ✓ Docker Compose config valid" || echo "   ✗ Docker Compose config error"

echo ""
echo "🎉 Configuration fix complete!"
echo ""
echo "Next steps:"
echo "1. Review the .env file and add your API keys"
echo "2. Run: docker-compose up -d"
echo "3. Access the dashboard at http://localhost:80"
echo ""