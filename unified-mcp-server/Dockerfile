# Unified MCP Server Docker Image
FROM node:18-bullseye-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Docker integration
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-dev \
    pkg-config \
    libopus-dev \
    libvpx-dev \
    libssl-dev \
    docker.io \
    curl \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy server files
COPY . .
COPY ../scripts/console-shim.js ./scripts/console-shim.js

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data

# Set permissions
RUN chmod +x *.sh 2>/dev/null || true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:9090/health || exit 1

# Expose MCP port
EXPOSE 3000

# Set environment variables
ENV NODE_ENV=production
ENV MCP_SERVER_PORT=3000

# Start the server
CMD ["node", "-r", "./scripts/console-shim.js", "server.js"]