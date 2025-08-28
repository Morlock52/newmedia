# MediaServer MCP Suite Dockerfile
# Builds the complete MCP server suite with AI agent voting system

FROM node:20-alpine AS builder

# Install build dependencies
RUN apk add --no-cache python3 make g++

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY src/ ./src/
COPY public/ ./public/

# Production image
FROM node:20-alpine

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    tini \
    && addgroup -g 1001 -S nodejs \
    && adduser -S nodejs -u 1001

# Set working directory
WORKDIR /app

# Copy built application
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/src ./src
COPY --from=builder --chown=nodejs:nodejs /app/public ./public
COPY --from=builder --chown=nodejs:nodejs /app/package*.json ./

# Create logs directory
RUN mkdir -p logs && chown nodejs:nodejs logs

# Switch to non-root user
USER nodejs

# Environment variables
ENV NODE_ENV=production \
    PORT=8090 \
    LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8090/health || exit 1

# Expose ports
# 8090: Main dashboard and API
# 3001-3005: MCP servers (Jellyfin, Sonarr, Radarr, Prowlarr, qBittorrent)
EXPOSE 8090 3001 3002 3003 3004 3005

# Use tini as init system
ENTRYPOINT ["/sbin/tini", "--"]

# Start the application
CMD ["node", "src/index.js"]

# Labels
LABEL maintainer="MediaServer MCP Suite" \
      version="1.0.0" \
      description="Complete MCP server suite with AI agent voting system for media server management" \
      org.opencontainers.image.source="https://github.com/mediaserver/mcp-suite" \
      org.opencontainers.image.title="MediaServer MCP Suite" \
      org.opencontainers.image.description="AI-powered media server management with MCP protocol and agent voting"