# API Server Dockerfile - Node.js Production
FROM node:20-alpine

WORKDIR /app

# Install dependencies for system monitoring
RUN apk add --no-cache \
    curl \
    docker-cli \
    python3 \
    py3-pip

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application files
COPY . .

# Create logs directory
RUN mkdir -p /var/log/api

# Expose port
EXPOSE 3002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3002/health || exit 1

# Start the application
CMD ["node", "server.js"]