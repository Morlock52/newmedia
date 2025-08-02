# Docker Container Performance Optimization Strategies

## 1. Multi-Stage Build Optimization

### Strategy Overview
Multi-stage builds dramatically reduce final image size and build time by separating build dependencies from runtime requirements.

### Best Practices

#### A. Language-Specific Optimizations

**Node.js Example:**
```dockerfile
# Build stage
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Runtime stage
FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]
```

**Python Example:**
```dockerfile
# Build stage
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "app.py"]
```

**Go Example:**
```dockerfile
# Build stage
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o main .

# Runtime stage
FROM scratch
COPY --from=builder /app/main /
EXPOSE 8080
CMD ["/main"]
```

### Performance Metrics
- Image size reduction: 50-90%
- Build time improvement: 30-60%
- Memory footprint: 40-70% reduction

## 2. Layer Caching Strategies

### Optimization Techniques

#### A. Order Dependencies by Change Frequency
```dockerfile
# Least frequently changed (cached more often)
COPY package.json package-lock.json ./
RUN npm ci

# More frequently changed
COPY src/ ./src/
COPY public/ ./public/

# Most frequently changed
COPY config/ ./config/
```

#### B. Combine RUN Commands Intelligently
```dockerfile
# GOOD: Single layer, atomic operation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        vim && \
    rm -rf /var/lib/apt/lists/*

# BAD: Multiple layers, inefficient
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y git
RUN apt-get install -y vim
```

#### C. Use .dockerignore Effectively
```
# .dockerignore
node_modules
npm-debug.log
.git
.gitignore
.env
coverage
.nyc_output
.vscode
.idea
*.swp
*.swo
*~
.DS_Store
```

## 3. BuildKit Features Utilization

### Enable BuildKit
```bash
# Environment variable
export DOCKER_BUILDKIT=1

# Docker daemon config
{
  "features": {
    "buildkit": true
  }
}
```

### Advanced Features

#### A. Cache Mounts
```dockerfile
# Persistent package manager cache
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y git

# NPM cache
RUN --mount=type=cache,target=/root/.npm \
    npm ci --only=production
```

#### B. Secret Mounts
```dockerfile
# Build-time secrets (not persisted in image)
RUN --mount=type=secret,id=npm_token \
    NPM_TOKEN=$(cat /run/secrets/npm_token) \
    npm install
```

#### C. SSH Mounts
```dockerfile
# Private repository access
RUN --mount=type=ssh \
    git clone git@github.com:private/repo.git
```

## 4. Resource Limits and Requests

### Container Resource Configuration

#### A. Memory Optimization
```yaml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M
    environment:
      # JVM example
      - JAVA_OPTS=-Xmx384m -Xms256m -XX:MaxMetaspaceSize=128m
      # Node.js example
      - NODE_OPTIONS=--max-old-space-size=384
```

#### B. CPU Optimization
```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '2.0'
        reservations:
          cpus: '0.5'
    # CPU pinning for performance
    cpuset: "0,1"
```

#### C. I/O Limits
```yaml
services:
  database:
    blkio_config:
      weight: 500
      device_read_bps:
        - path: /dev/sda
          rate: '100m'
      device_write_bps:
        - path: /dev/sda
          rate: '50m'
```

## 5. Health Check Optimization

### Efficient Health Checks

#### A. Lightweight Checks
```dockerfile
# GOOD: Minimal overhead
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# BETTER: Built-in check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD ["/app/healthcheck"]
```

#### B. Custom Health Check Script
```bash
#!/bin/sh
# healthcheck.sh
set -e

# Check process
pgrep -f "node server.js" > /dev/null || exit 1

# Check port
nc -z localhost 3000 || exit 1

# Check endpoint (lightweight)
curl -sf http://localhost:3000/ping > /dev/null || exit 1

exit 0
```

## 6. Container Startup Time Reduction

### Optimization Techniques

#### A. Precompile and Cache
```dockerfile
# Python: Precompile bytecode
RUN python -m compileall -q /app

# Node.js: Precompile with V8 cache
RUN node --max-old-space-size=4096 /app/scripts/precompile.js
```

#### B. Lazy Loading
```javascript
// Defer heavy imports
let heavyModule;
function getHeavyModule() {
  if (!heavyModule) {
    heavyModule = require('./heavy-module');
  }
  return heavyModule;
}
```

#### C. Init Containers
```yaml
services:
  app:
    depends_on:
      init-db:
        condition: service_completed_successfully
  
  init-db:
    image: migrate/migrate
    command: ["-path", "/migrations", "-database", "postgres://...", "up"]
```

## 7. Volume Performance Tuning

### Volume Optimization Strategies

#### A. Volume Types and Performance
```yaml
volumes:
  # Named volume (best for data persistence)
  db_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /fast-ssd/postgres

  # Tmpfs mount (best for temporary data)
  cache_data:
    driver: local
    driver_opts:
      type: tmpfs
      device: tmpfs
      o: size=2g,uid=1000,gid=1000

  # Bind mount with cache mode
  app_data:
    driver: local
    driver_opts:
      type: none
      o: bind,cached  # macOS performance
      device: ./data
```

#### B. Volume Mount Options
```yaml
services:
  app:
    volumes:
      # Read-only for security and performance
      - ./config:/app/config:ro
      # Cached for better macOS performance
      - ./src:/app/src:cached
      # Delegated for write-heavy workloads
      - ./logs:/app/logs:delegated
```

## 8. Network Performance Optimization

### Network Configuration

#### A. Custom Networks
```yaml
networks:
  frontend:
    driver: bridge
    driver_opts:
      com.docker.network.driver.mtu: 9000  # Jumbo frames
    ipam:
      config:
        - subnet: 172.20.0.0/24
          gateway: 172.20.0.1

  backend:
    driver: bridge
    internal: true  # No external access
```

#### B. Network Policies
```yaml
services:
  app:
    networks:
      - frontend
      - backend
    sysctls:
      - net.core.somaxconn=65535
      - net.ipv4.tcp_syncookies=1
      - net.ipv4.tcp_tw_reuse=1
```

#### C. DNS Optimization
```yaml
services:
  app:
    dns:
      - 1.1.1.1
      - 8.8.8.8
    dns_search:
      - service.local
    dns_opt:
      - use-vc
      - no-tld-query
```

## Performance Monitoring Commands

### Real-time Monitoring
```bash
# Container stats
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# Container processes
docker top <container> aux

# Resource usage over time
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  google/cadvisor:latest
```

### Benchmarking Tools
```bash
# CPU benchmark
docker run --rm stress-ng --cpu 4 --timeout 60s --metrics

# Memory benchmark
docker run --rm stress-ng --vm 2 --vm-bytes 256M --timeout 60s

# I/O benchmark
docker run --rm -v /tmp:/tmp fio --name=randwrite --ioengine=libaio \
  --iodepth=64 --rw=randwrite --bs=4k --direct=1 --size=1G --numjobs=4
```