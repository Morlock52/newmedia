# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build arguments for environment variables
ARG REACT_APP_API_URL
ARG REACT_APP_IPFS_GATEWAY
ARG REACT_APP_JELLYFIN_URL
ARG REACT_APP_CONTENT_OWNERSHIP_ADDRESS
ARG REACT_APP_MEDIA_DAO_ADDRESS
ARG REACT_APP_MARKETPLACE_ADDRESS
ARG REACT_APP_DEFAULT_CHAIN_ID
ARG REACT_APP_SUPPORTED_CHAINS

# Set environment variables
ENV REACT_APP_API_URL=$REACT_APP_API_URL
ENV REACT_APP_IPFS_GATEWAY=$REACT_APP_IPFS_GATEWAY
ENV REACT_APP_JELLYFIN_URL=$REACT_APP_JELLYFIN_URL
ENV REACT_APP_CONTENT_OWNERSHIP_ADDRESS=$REACT_APP_CONTENT_OWNERSHIP_ADDRESS
ENV REACT_APP_MEDIA_DAO_ADDRESS=$REACT_APP_MEDIA_DAO_ADDRESS
ENV REACT_APP_MARKETPLACE_ADDRESS=$REACT_APP_MARKETPLACE_ADDRESS
ENV REACT_APP_DEFAULT_CHAIN_ID=$REACT_APP_DEFAULT_CHAIN_ID
ENV REACT_APP_SUPPORTED_CHAINS=$REACT_APP_SUPPORTED_CHAINS

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine AS production

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built app from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/ || exit 1

# Expose port
EXPOSE 3000

# Start nginx
CMD ["nginx", "-g", "daemon off;"]