# Unified MCP Server Docker Image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Install system dependencies for Docker integration
RUN apk add --no-cache \
    docker-cli \
    curl \
    bash

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy server files
COPY . .

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
CMD ["node", "server.js"]