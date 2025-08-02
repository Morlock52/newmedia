# Multi-stage build for quantum security system
FROM node:20-alpine AS builder

# Install build dependencies
RUN apk add --no-cache python3 make g++ git

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY src/ ./src/

# Second stage - runtime
FROM node:20-alpine

# Install runtime dependencies
RUN apk add --no-cache \
    openssl \
    ca-certificates \
    tini

# Create non-root user
RUN addgroup -g 1001 -S quantum && \
    adduser -u 1001 -S quantum -G quantum

# Set working directory
WORKDIR /app

# Copy from builder
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/src ./src
COPY package*.json ./

# Create directories for certificates and logs
RUN mkdir -p /app/certs /app/logs && \
    chown -R quantum:quantum /app

# Switch to non-root user
USER quantum

# Expose ports
EXPOSE 3000 8443

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/health', (res) => process.exit(res.statusCode === 200 ? 0 : 1))"

# Use tini for proper signal handling
ENTRYPOINT ["/sbin/tini", "--"]

# Start the server
CMD ["node", "src/index.js"]