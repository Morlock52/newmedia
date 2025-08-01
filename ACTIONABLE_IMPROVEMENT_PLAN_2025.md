# Actionable Improvement Plan - Media Server Project 2025

## 🚨 Phase 1: Critical Security Fixes (24-48 hours)

### 1.1 Remove Hardcoded API Keys
```bash
# Step 1: Create secure environment file
cat > .env.secure << EOF
SONARR_API_KEY=
RADARR_API_KEY=
PROWLARR_API_KEY=
LIDARR_API_KEY=
BAZARR_API_KEY=
OVERSEERR_API_KEY=
TAUTULLI_API_KEY=
EOF

# Step 2: Update docker-compose.yml to use env variables
# Replace hardcoded keys with ${SONARR_API_KEY} etc.

# Step 3: Regenerate all API keys in each service's UI
# Sonarr: Settings → General → API Key → Regenerate
# Repeat for all services
```

### 1.2 Implement Secrets Management
```yaml
# docker-compose.secure.yml
services:
  sonarr:
    environment:
      - SONARR__ApiKey=${SONARR_API_KEY}
    env_file:
      - .env.secure
```

### 1.3 Deploy Authentication Layer
```bash
# Use the provided secure-docker-compose-template.yml
docker-compose -f secure-docker-compose-template.yml up -d authelia
```

### 1.4 Clean Git History
```bash
# If secrets were committed
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch docker-compose.yml' \
  --prune-empty --tag-name-filter cat -- --all
```

---

## 🔧 Phase 2: Code Quality Improvements (1 week)

### 2.1 Refactor Large Python Files

**Create modular structure:**
```
scripts/
├── core/
│   ├── __init__.py
│   ├── config_manager.py
│   ├── logger.py
│   └── exceptions.py
├── maintenance/
│   ├── __init__.py
│   ├── backup.py
│   ├── monitoring.py
│   └── cleanup.py
├── user_experience/
│   ├── __init__.py
│   ├── analytics.py
│   ├── recommendations.py
│   └── ui_manager.py
└── content_discovery/
    ├── __init__.py
    ├── api_clients.py
    ├── search.py
    └── database.py
```

### 2.2 Extract Shared Configuration Module
```python
# scripts/core/config_manager.py
import os
import json
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_dir: str = "/config"):
        self.config_dir = Path(config_dir)
        self._cache = {}
    
    def load_config(self, service: str) -> Dict[str, Any]:
        if service in self._cache:
            return self._cache[service]
        
        config_path = self.config_dir / f"{service}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Override with environment variables
        for key, value in config.items():
            env_key = f"{service.upper()}_{key.upper()}"
            if env_value := os.getenv(env_key):
                config[key] = env_value
        
        self._cache[service] = config
        return config
```

### 2.3 Improve Error Handling
```python
# scripts/core/exceptions.py
class MediaServerError(Exception):
    """Base exception for media server"""
    pass

class ConfigurationError(MediaServerError):
    """Configuration related errors"""
    pass

class ProcessingError(MediaServerError):
    """Media processing errors"""
    pass

class APIError(MediaServerError):
    """External API errors"""
    def __init__(self, service: str, status_code: int, message: str):
        self.service = service
        self.status_code = status_code
        super().__init__(f"{service} API error ({status_code}): {message}")
```

### 2.4 Shell Script Hardening
```bash
#!/usr/bin/env bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures
IFS=$'\n\t'       # Set secure Internal Field Separator

# Add to all shell scripts
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# Function for logging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$SCRIPT_NAME] $*" >&2
}

# Error handling
trap 'log "Error on line $LINENO"' ERR
```

---

## ⚡ Phase 3: Performance Optimization (2 weeks)

### 3.1 Add Redis Caching
```yaml
# docker-compose.cache.yml
services:
  redis:
    image: redis:7-alpine
    container_name: redis
    command: redis-server --appendonly yes
    volumes:
      - ./data/redis:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    networks:
      - media_network
```

### 3.2 Implement Tdarr for Transcoding
```yaml
services:
  tdarr:
    image: ghcr.io/haveagitgat/tdarr:latest
    container_name: tdarr
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=America/New_York
      - serverIP=0.0.0.0
      - serverPort=8266
    ports:
      - "8265:8265" # Web UI
      - "8266:8266" # Server
    volumes:
      - ./config/tdarr/server:/app/server
      - ./config/tdarr/configs:/app/configs
      - ./config/tdarr/logs:/app/logs
      - ./data/media:/media
      - ./temp/transcodes:/temp
    devices:
      - /dev/dri:/dev/dri  # Intel GPU
    restart: unless-stopped
```

### 3.3 Storage Optimization
```bash
# Optimize Docker volumes
cat > /etc/docker/daemon.json << EOF
{
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

# Tune kernel parameters
cat > /etc/sysctl.d/99-media-server.conf << EOF
# Increase inotify limits
fs.inotify.max_user_watches=524288
fs.inotify.max_user_instances=512

# Network performance
net.core.rmem_max=134217728
net.core.wmem_max=134217728
net.ipv4.tcp_rmem=4096 87380 134217728
net.ipv4.tcp_wmem=4096 65536 134217728

# File system
vm.dirty_ratio=10
vm.dirty_background_ratio=5
EOF

sysctl -p /etc/sysctl.d/99-media-server.conf
```

### 3.4 Database Tuning
```sql
-- PostgreSQL tuning for Immich
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET work_mem = '10MB';
ALTER SYSTEM SET min_wal_size = '1GB';
ALTER SYSTEM SET max_wal_size = '4GB';
```

---

## 🏗️ Phase 4: Architecture Improvements (1 month)

### 4.1 Consolidate Docker Compose Files
```yaml
# docker-compose.base.yml - Core services
# docker-compose.media.yml - Media services
# docker-compose.download.yml - Download clients
# docker-compose.monitoring.yml - Monitoring stack
# docker-compose.security.yml - Security services

# Usage:
# docker-compose -f docker-compose.base.yml -f docker-compose.media.yml up -d
```

### 4.2 Implement Message Queue
```yaml
services:
  rabbitmq:
    image: rabbitmq:3-management-alpine
    container_name: rabbitmq
    environment:
      RABBITMQ_DEFAULT_USER: ${RABBITMQ_USER}
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASS}
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - ./data/rabbitmq:/var/lib/rabbitmq
    restart: unless-stopped
```

### 4.3 Add Service Mesh (Optional)
```bash
# Install Linkerd
curl --proto '=https' --tlsv1.2 -sSfL https://run.linkerd.io/install | sh
linkerd install | kubectl apply -f -
linkerd check

# Inject into services
kubectl get -n media-server deploy -o yaml | linkerd inject - | kubectl apply -f -
```

---

## 📊 Phase 5: Monitoring & Alerting (2 weeks)

### 5.1 Enhanced Prometheus Configuration
```yaml
# prometheus/alert.rules.yml
groups:
  - name: media_server
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
      
      - alert: HighCPUUsage
        expr: rate(process_cpu_seconds_total[5m]) > 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage on {{ $labels.job }}"
      
      - alert: LowDiskSpace
        expr: node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes < 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space (< 10% free)"
```

### 5.2 Grafana Dashboards
```json
{
  "dashboard": {
    "title": "Media Server Overview",
    "panels": [
      {
        "title": "Service Status",
        "targets": [{"expr": "up"}]
      },
      {
        "title": "Media Processing Queue",
        "targets": [{"expr": "media_queue_size"}]
      },
      {
        "title": "Storage Usage",
        "targets": [{"expr": "node_filesystem_used_bytes"}]
      }
    ]
  }
}
```

---

## 🧪 Phase 6: Testing Implementation (3 weeks)

### 6.1 Unit Tests
```python
# tests/test_config_manager.py
import pytest
from scripts.core.config_manager import ConfigManager

def test_load_config():
    manager = ConfigManager("/test/config")
    config = manager.load_config("test_service")
    assert "api_key" in config
    assert config["api_key"] == "test_key"

def test_env_override(monkeypatch):
    monkeypatch.setenv("TEST_SERVICE_API_KEY", "env_key")
    manager = ConfigManager("/test/config")
    config = manager.load_config("test_service")
    assert config["api_key"] == "env_key"
```

### 6.2 Integration Tests
```python
# tests/integration/test_media_processing.py
import pytest
from scripts.media_processing import MediaProcessor

@pytest.mark.integration
def test_video_transcoding():
    processor = MediaProcessor()
    result = processor.process_file("/test/video.mp4")
    assert result.success
    assert result.output_codec == "h265"
```

### 6.3 Load Testing
```python
# tests/load/test_api_performance.py
import asyncio
import aiohttp
import time

async def test_concurrent_requests():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(100):
            task = session.get(f"http://localhost:8003/api/stats")
            tasks.append(task)
        
        start = time.time()
        responses = await asyncio.gather(*tasks)
        duration = time.time() - start
        
        assert all(r.status == 200 for r in responses)
        assert duration < 5.0  # Should handle 100 requests in < 5 seconds
```

---

## 📋 Implementation Timeline

| Week | Phase | Tasks |
|------|-------|-------|
| 1 | Security | Remove API keys, implement auth, secrets management |
| 2 | Code Quality | Refactor Python files, improve error handling |
| 3-4 | Performance | Add caching, implement Tdarr, optimize storage |
| 5-6 | Architecture | Consolidate configs, add message queue |
| 7-8 | Monitoring | Enhanced alerts, custom dashboards |
| 9-11 | Testing | Unit, integration, and load tests |
| 12 | Documentation | Update all docs, create migration guide |

---

## 🎯 Success Metrics

1. **Security**: Zero hardcoded secrets, all services authenticated
2. **Performance**: 40% reduction in transcoding time, <100ms API response
3. **Code Quality**: 80% test coverage, <500 lines per file
4. **Reliability**: 99.9% uptime, automatic error recovery
5. **User Experience**: <3s page load, intuitive navigation

---

## 🚀 Quick Wins (Can implement today)

1. Remove hardcoded API keys (1 hour)
2. Add `set -euo pipefail` to all scripts (30 mins)
3. Create shared config module (2 hours)
4. Deploy Redis cache (30 mins)
5. Add basic health check endpoints (1 hour)

Start with these quick wins to see immediate improvements while planning the larger refactoring efforts.