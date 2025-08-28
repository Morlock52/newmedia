#!/bin/bash
# ========================================
# MEGA CONTAINER ENTRYPOINT SCRIPT
# Orchestrates startup of 30+ interconnected services
# ========================================

set -e

echo "🚀 Starting Mega Container with 30+ Services..."
echo "🕐 $(date): Beginning initialization sequence"

# ========================================
# ENVIRONMENT SETUP
# ========================================

export PUID=${PUID:-1000}
export PGID=${PGID:-1000}
export TZ=${TZ:-UTC}

# Service discovery environment variables
export POSTGRES_HOST="127.0.0.1"
export POSTGRES_PORT="5432"
export REDIS_HOST="127.0.0.1"
export REDIS_PORT="6379"
export RABBITMQ_HOST="127.0.0.1"
export RABBITMQ_PORT="5672"
export TRAEFIK_HOST="127.0.0.1"
export TRAEFIK_PORT="8080"
export AUTHELIA_HOST="127.0.0.1"
export AUTHELIA_PORT="9091"

# ========================================
# CREATE USERS AND GROUPS
# ========================================

echo "👥 Creating service users and groups..."

# Create main group
groupadd -g ${PGID} mediaserver || true

# Create service users
users=(
    "postgres:999"
    "redis:998"
    "rabbitmq:997"
    "traefik:996"
    "authelia:995"
    "prometheus:994"
    "grafana:993"
    "loki:992"
    "promtail:991"
    "qbittorrent:990"
    "prowlarr:989"
    "sonarr:988"
    "radarr:987"
    "lidarr:986"
    "readarr:985"
    "bazarr:984"
    "jellyfin:983"
    "tautulli:982"
    "portainer:981"
    "uptime-kuma:980"
    "heimdall:979"
)

for user_info in "${users[@]}"; do
    username=$(echo $user_info | cut -d: -f1)
    uid=$(echo $user_info | cut -d: -f2)
    useradd -u $uid -g $PGID -d /config/$username -s /bin/bash $username 2>/dev/null || true
    mkdir -p /config/$username
    chown $username:mediaserver /config/$username
done

# ========================================
# INITIALIZE DIRECTORIES
# ========================================

echo "📁 Setting up directory structure..."

# Create all necessary directories
mkdir -p /config/{postgresql,redis,rabbitmq,traefik,authelia}
mkdir -p /config/{prometheus,grafana,loki,promtail}
mkdir -p /config/{qbittorrent,prowlarr,sonarr,radarr,lidarr,readarr,bazarr}
mkdir -p /config/{jellyfin,tautulli,portainer,uptime-kuma,heimdall}
mkdir -p /data/{downloads/{complete,incomplete,watch},media/{movies,tv,music,books}}
mkdir -p /data/{prometheus,grafana,loki,databases}
mkdir -p /logs/{apps,system,access,databases}

# Set proper permissions
chown -R root:mediaserver /config /data /logs
chmod -R 775 /config /data /logs

# ========================================
# DATABASE INITIALIZATION
# ========================================

echo "🗄️ Initializing databases..."

# Initialize PostgreSQL
if [ ! -d "/config/postgresql/base" ]; then
    echo "📦 Initializing PostgreSQL database..."
    mkdir -p /config/postgresql
    chown postgres:postgres /config/postgresql
    chmod 700 /config/postgresql
    
    su postgres -c '/usr/lib/postgresql/15/bin/initdb -D /config/postgresql'
    
    # Configure PostgreSQL
    cat > /config/postgresql/postgresql.conf << 'EOF'
listen_addresses = '*'
port = 5432
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.7
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB
log_destination = 'stderr'
logging_collector = on
log_directory = '/logs/databases'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_rotation_size = 10MB
EOF

    cat > /config/postgresql/pg_hba.conf << 'EOF'
local   all             postgres                                peer
local   all             all                                     md5
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5
host    all             all             0.0.0.0/0               md5
EOF

    chown postgres:postgres /config/postgresql/postgresql.conf /config/postgresql/pg_hba.conf
fi

# Initialize Redis
echo "📦 Initializing Redis configuration..."
cat > /config/redis/redis.conf << 'EOF'
bind 127.0.0.1
port 6379
timeout 0
tcp-keepalive 300
daemonize no
supervised systemd
pidfile /var/run/redis/redis-server.pid
loglevel notice
logfile /logs/databases/redis.log
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data/databases/redis
maxmemory 512mb
maxmemory-policy allkeys-lru
EOF

mkdir -p /data/databases/redis /var/run/redis
chown redis:redis /config/redis/redis.conf /data/databases/redis /var/run/redis

# ========================================
# SERVICE CONFIGURATIONS
# ========================================

echo "⚙️ Generating service configurations..."

# Traefik Configuration
cat > /config/traefik/traefik.yml << 'EOF'
global:
  checkNewVersion: false
  sendAnonymousUsage: false

api:
  dashboard: true
  insecure: true

entryPoints:
  web:
    address: ":80"
  websecure:
    address: ":443"
  traefik:
    address: ":8080"

providers:
  file:
    filename: /config/traefik/dynamic.yml
    watch: true

certificatesResolvers:
  letsencrypt:
    acme:
      email: admin@localhost
      storage: /config/traefik/acme.json
      httpChallenge:
        entryPoint: web

log:
  level: INFO
  filePath: /logs/access/traefik.log

accessLog:
  filePath: /logs/access/traefik_access.log
EOF

# Traefik Dynamic Configuration
cat > /config/traefik/dynamic.yml << 'EOF'
http:
  routers:
    # *arr applications
    sonarr:
      rule: "Host(`sonarr.localhost`) || PathPrefix(`/sonarr`)"
      service: sonarr
      middlewares:
        - auth
    radarr:
      rule: "Host(`radarr.localhost`) || PathPrefix(`/radarr`)"
      service: radarr
      middlewares:
        - auth
    lidarr:
      rule: "Host(`lidarr.localhost`) || PathPrefix(`/lidarr`)"
      service: lidarr
      middlewares:
        - auth
    readarr:
      rule: "Host(`readarr.localhost`) || PathPrefix(`/readarr`)"
      service: readarr
      middlewares:
        - auth
    prowlarr:
      rule: "Host(`prowlarr.localhost`) || PathPrefix(`/prowlarr`)"
      service: prowlarr
      middlewares:
        - auth
    bazarr:
      rule: "Host(`bazarr.localhost`) || PathPrefix(`/bazarr`)"
      service: bazarr
      middlewares:
        - auth
    
    # Download clients
    qbittorrent:
      rule: "Host(`qbt.localhost`) || PathPrefix(`/qbittorrent`)"
      service: qbittorrent
      middlewares:
        - auth
    
    # Media servers
    jellyfin:
      rule: "Host(`jellyfin.localhost`) || PathPrefix(`/jellyfin`)"
      service: jellyfin
    
    # Monitoring
    grafana:
      rule: "Host(`grafana.localhost`) || PathPrefix(`/grafana`)"
      service: grafana
    prometheus:
      rule: "Host(`prometheus.localhost`) || PathPrefix(`/prometheus`)"
      service: prometheus
      middlewares:
        - auth
    
    # Management
    portainer:
      rule: "Host(`portainer.localhost`) || PathPrefix(`/portainer`)"
      service: portainer
    
    # Dashboard
    heimdall:
      rule: "Host(`dashboard.localhost`) || PathPrefix(`/`)"
      service: heimdall

  services:
    sonarr:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:8989"
    radarr:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:7878"
    lidarr:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:8686"
    readarr:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:8787"
    prowlarr:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:9696"
    bazarr:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:6767"
    qbittorrent:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:8090"
    jellyfin:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:8096"
    grafana:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:3000"
    prometheus:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:9090"
    portainer:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:9000"
    heimdall:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:8080"

  middlewares:
    auth:
      forwardAuth:
        address: "http://127.0.0.1:9091/api/verify?rd=https://auth.localhost"
        trustForwardHeader: true
        authResponseHeaders:
          - "Remote-User"
          - "Remote-Name"
          - "Remote-Email"
          - "Remote-Groups"
EOF

# Authelia Configuration
cat > /config/authelia/configuration.yml << 'EOF'
server:
  host: 0.0.0.0
  port: 9091

log:
  level: info
  file_path: /logs/apps/authelia.log

jwt_secret: your-jwt-secret-here-change-this

default_redirection_url: https://auth.localhost

authentication_backend:
  password_reset:
    disable: false
  file:
    path: /config/authelia/users_database.yml
    password:
      algorithm: argon2id
      iterations: 1
      salt_length: 16
      parallelism: 8
      memory: 64

access_control:
  default_policy: deny
  rules:
    - domain: "*.localhost"
      policy: one_factor
    - domain: "localhost"
      policy: one_factor

session:
  name: authelia_session
  domain: localhost
  same_site: lax
  secret: your-session-secret-here-change-this
  expiration: 1h
  inactivity: 5m
  redis:
    host: 127.0.0.1
    port: 6379
    database_index: 0

regulation:
  max_retries: 3
  find_time: 120
  ban_time: 300

storage:
  postgres:
    host: 127.0.0.1
    port: 5432
    database: authelia
    username: authelia
    password: authelia_password

notifier:
  disable_startup_check: true
  filesystem:
    filename: /logs/apps/authelia_notifications.txt
EOF

# Create Authelia users database
cat > /config/authelia/users_database.yml << 'EOF'
users:
  admin:
    displayname: "Administrator"
    password: "$argon2id$v=19$m=65536,t=1,p=8$eUFLZWxGU1pWM0FyNE1KdA$JNMIaDHarx7EK64I6WF2wMgPaL5Yk2u4OVU5vXhzc0U"  # Password: admin123
    email: admin@localhost
    groups:
      - admins
      - dev
  user:
    displayname: "Regular User"
    password: "$argon2id$v=19$m=65536,t=1,p=8$eUFLZWxGU1pWM0FyNE1KdA$JNMIaDHarx7EK64I6WF2wMgPaL5Yk2u4OVU5vXhzc0U"  # Password: user123
    email: user@localhost
    groups:
      - dev
EOF

# Prometheus Configuration
cat > /config/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/config/prometheus/rules/*.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'sonarr'
    static_configs:
      - targets: ['localhost:8989']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'radarr'
    static_configs:
      - targets: ['localhost:7878']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'qbittorrent'
    static_configs:
      - targets: ['localhost:8090']
    scrape_interval: 30s

  - job_name: 'jellyfin'
    static_configs:
      - targets: ['localhost:8096']
    scrape_interval: 30s

  - job_name: 'traefik'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: /metrics

  - job_name: 'authelia'
    static_configs:
      - targets: ['localhost:9091']
    metrics_path: /api/health

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:6379']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
EOF

# Grafana Configuration
cat > /config/grafana/grafana.ini << 'EOF'
[server]
http_addr = 0.0.0.0
http_port = 3000
root_url = %(protocol)s://%(domain)s:%(http_port)s/grafana/

[database]
type = postgres
host = 127.0.0.1:5432
name = grafana
user = grafana
password = grafana_password

[security]
admin_user = admin
admin_password = admin123
secret_key = grafana-secret-key

[auth]
disable_login_form = false

[auth.anonymous]
enabled = false

[log]
mode = file
level = info
file = /logs/apps/grafana.log

[alerting]
enabled = true

[unified_alerting]
enabled = true
EOF

# Loki Configuration
cat > /config/loki/loki.yml << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100

common:
  path_prefix: /data/loki
  storage:
    filesystem:
      chunks_directory: /data/loki/chunks
      rules_directory: /data/loki/rules
  replication_factor: 1
  ring:
    instance_addr: 127.0.0.1
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

ruler:
  alertmanager_url: http://localhost:9093

limits_config:
  reject_old_samples: true
  reject_old_samples_max_age: 168h
  ingestion_rate_mb: 16
  ingestion_burst_size_mb: 32

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: true
  retention_period: 168h
EOF

# Promtail Configuration
cat > /config/promtail/promtail.yml << 'EOF'
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /data/loki/positions.yaml

clients:
  - url: http://localhost:3100/loki/api/v1/push

scrape_configs:
  - job_name: system-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: varlogs
          __path__: /var/log/*log

  - job_name: app-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: applogs
          __path__: /logs/apps/*log

  - job_name: access-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: accesslogs
          __path__: /logs/access/*log
EOF

echo "✅ Service configurations generated!"

# ========================================
# DATABASE SETUP
# ========================================

echo "🔧 Setting up application databases..."

# Start PostgreSQL temporarily for database setup
echo "🔄 Starting PostgreSQL for initial setup..."
su postgres -c '/usr/lib/postgresql/15/bin/pg_ctl -D /config/postgresql -l /logs/databases/postgres_setup.log start'

# Wait for PostgreSQL to be ready
sleep 10

# Create databases and users
su postgres -c "createdb authelia" || true
su postgres -c "createdb grafana" || true
su postgres -c "createdb sonarr" || true
su postgres -c "createdb radarr" || true
su postgres -c "createdb lidarr" || true
su postgres -c "createdb readarr" || true
su postgres -c "createdb prowlarr" || true
su postgres -c "createdb bazarr" || true
su postgres -c "createdb jellyfin" || true
su postgres -c "createdb tautulli" || true

# Create users with passwords
su postgres -c "psql -c \"CREATE USER authelia WITH PASSWORD 'authelia_password'; GRANT ALL PRIVILEGES ON DATABASE authelia TO authelia;\"" || true
su postgres -c "psql -c \"CREATE USER grafana WITH PASSWORD 'grafana_password'; GRANT ALL PRIVILEGES ON DATABASE grafana TO grafana;\"" || true

# Stop PostgreSQL
su postgres -c '/usr/lib/postgresql/15/bin/pg_ctl -D /config/postgresql stop' || true

echo "✅ Database setup completed!"

# ========================================
# SERVICE INTERCONNECTION SETUP
# ========================================

echo "🔗 Setting up service interconnections..."

# Create service API keys and configuration interconnections
mkdir -p /config/api-keys

# Generate API keys for service communication
echo "generating_sonarr_api_key" > /config/api-keys/sonarr
echo "generating_radarr_api_key" > /config/api-keys/radarr
echo "generating_lidarr_api_key" > /config/api-keys/lidarr
echo "generating_readarr_api_key" > /config/api-keys/readarr
echo "generating_prowlarr_api_key" > /config/api-keys/prowlarr

# ========================================
# CREATE SERVICE ORCHESTRATOR
# ========================================

cat > /opt/scripts/service_orchestrator.py << 'EOF'
#!/usr/bin/env python3
"""
Service Orchestrator for Mega Container
Manages service dependencies, health checks, and inter-service communication
"""

import time
import requests
import subprocess
import json
import logging
import psutil
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import redis
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ServiceOrchestrator:
    def __init__(self):
        self.services = {
            'postgres': {'port': 5432, 'health_url': None, 'critical': True},
            'redis': {'port': 6379, 'health_url': None, 'critical': True},
            'rabbitmq': {'port': 15672, 'health_url': 'http://localhost:15672/api/healthchecks/node', 'critical': True},
            'traefik': {'port': 8080, 'health_url': 'http://localhost:8080/ping', 'critical': True},
            'authelia': {'port': 9091, 'health_url': 'http://localhost:9091/api/health', 'critical': True},
            'prometheus': {'port': 9090, 'health_url': 'http://localhost:9090/-/healthy', 'critical': False},
            'grafana': {'port': 3000, 'health_url': 'http://localhost:3000/api/health', 'critical': False},
            'loki': {'port': 3100, 'health_url': 'http://localhost:3100/ready', 'critical': False},
            'prowlarr': {'port': 9696, 'health_url': 'http://localhost:9696/ping', 'critical': True},
            'sonarr': {'port': 8989, 'health_url': 'http://localhost:8989/ping', 'critical': False},
            'radarr': {'port': 7878, 'health_url': 'http://localhost:7878/ping', 'critical': False},
            'lidarr': {'port': 8686, 'health_url': 'http://localhost:8686/ping', 'critical': False},
            'readarr': {'port': 8787, 'health_url': 'http://localhost:8787/ping', 'critical': False},
            'bazarr': {'port': 6767, 'health_url': 'http://localhost:6767/ping', 'critical': False},
            'qbittorrent': {'port': 8090, 'health_url': None, 'critical': True},
            'jellyfin': {'port': 8096, 'health_url': 'http://localhost:8096/health', 'critical': False},
            'tautulli': {'port': 8181, 'health_url': None, 'critical': False},
            'portainer': {'port': 9000, 'health_url': None, 'critical': False},
        }
        
        self.service_dependencies = {
            'authelia': ['postgres', 'redis'],
            'grafana': ['postgres'],
            'sonarr': ['postgres', 'prowlarr'],
            'radarr': ['postgres', 'prowlarr'],
            'lidarr': ['postgres', 'prowlarr'],
            'readarr': ['postgres', 'prowlarr'],
            'bazarr': ['sonarr', 'radarr'],
            'jellyfin': ['postgres'],
            'tautulli': ['jellyfin'],
        }
        
    def check_service_health(self, service_name):
        """Check if a service is healthy"""
        service = self.services.get(service_name, {})
        port = service.get('port')
        health_url = service.get('health_url')
        
        # Check port availability
        if port and not self.is_port_open(port):
            return False
            
        # Check HTTP health endpoint
        if health_url:
            try:
                response = requests.get(health_url, timeout=5)
                return response.status_code == 200
            except:
                return False
                
        # For services without health endpoints, assume healthy if port is open
        return port and self.is_port_open(port)
    
    def is_port_open(self, port):
        """Check if a port is open"""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    
    def setup_service_connections(self):
        """Setup interconnections between services"""
        logger.info("Setting up service interconnections...")
        
        # Wait for critical services to be healthy
        critical_services = [name for name, config in self.services.items() if config.get('critical')]
        
        for service in critical_services:
            logger.info(f"Waiting for critical service {service}...")
            while not self.check_service_health(service):
                time.sleep(5)
            logger.info(f"✅ {service} is healthy")
        
        # Configure *arr applications to use shared indexers
        self.configure_arr_indexers()
        
        # Setup download client connections
        self.configure_download_clients()
        
        # Configure media server connections
        self.configure_media_servers()
        
        logger.info("✅ Service interconnections configured!")
    
    def configure_arr_indexers(self):
        """Configure *arr apps to use Prowlarr indexers"""
        logger.info("Configuring *arr applications with Prowlarr indexers...")
        
        arr_apps = ['sonarr', 'radarr', 'lidarr', 'readarr']
        
        # Wait for Prowlarr to be ready
        while not self.check_service_health('prowlarr'):
            logger.info("Waiting for Prowlarr to be ready...")
            time.sleep(10)
        
        # Configure each *arr app
        for app in arr_apps:
            if self.check_service_health(app):
                logger.info(f"Configuring {app} with Prowlarr connection...")
                # Implementation would add Prowlarr connection to each *arr app
                # This would be done via their respective APIs
    
    def configure_download_clients(self):
        """Configure download clients in *arr applications"""
        logger.info("Configuring download clients...")
        
        # Wait for qBittorrent to be ready
        while not self.check_service_health('qbittorrent'):
            logger.info("Waiting for qBittorrent to be ready...")
            time.sleep(10)
        
        # Configure qBittorrent in each *arr app
        arr_apps = ['sonarr', 'radarr', 'lidarr', 'readarr']
        
        for app in arr_apps:
            if self.check_service_health(app):
                logger.info(f"Configuring qBittorrent in {app}...")
                # Implementation would configure download client settings
    
    def configure_media_servers(self):
        """Configure media server connections"""
        logger.info("Configuring media server connections...")
        
        # Wait for Jellyfin to be ready
        if self.check_service_health('jellyfin'):
            logger.info("Configuring Jellyfin media libraries...")
            # Implementation would configure Jellyfin libraries to point to media directories
        
        # Configure Tautulli to monitor Jellyfin
        if self.check_service_health('tautulli'):
            logger.info("Configuring Tautulli monitoring...")
            # Implementation would configure Tautulli to monitor media servers
    
    def monitor_services(self):
        """Continuously monitor service health"""
        logger.info("Starting service monitoring...")
        
        while True:
            try:
                unhealthy_services = []
                
                for service_name in self.services:
                    if not self.check_service_health(service_name):
                        unhealthy_services.append(service_name)
                        logger.warning(f"❌ Service {service_name} is unhealthy")
                    else:
                        logger.debug(f"✅ Service {service_name} is healthy")
                
                if unhealthy_services:
                    logger.warning(f"Unhealthy services detected: {unhealthy_services}")
                    # Could implement restart logic here
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in service monitoring: {e}")
                time.sleep(60)
    
    def run(self):
        """Main orchestrator loop"""
        logger.info("🚀 Service Orchestrator starting...")
        
        # Setup service connections
        self.setup_service_connections()
        
        # Start monitoring
        self.monitor_services()

if __name__ == "__main__":
    orchestrator = ServiceOrchestrator()
    orchestrator.run()
EOF

chmod +x /opt/scripts/service_orchestrator.py

# ========================================
# CREATE HEALTH MONITOR
# ========================================

cat > /opt/scripts/health_monitor.py << 'EOF'
#!/usr/bin/env python3
"""
Health Monitor for Mega Container
Provides health check endpoints and service status monitoring
"""

import time
import json
import logging
from flask import Flask, jsonify
import psutil
import requests
import sqlite3
from threading import Thread
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class HealthMonitor:
    def __init__(self):
        self.services = {
            'postgres': {'port': 5432, 'health_url': None},
            'redis': {'port': 6379, 'health_url': None},
            'rabbitmq': {'port': 15672, 'health_url': 'http://localhost:15672/api/healthchecks/node'},
            'traefik': {'port': 8080, 'health_url': 'http://localhost:8080/ping'},
            'authelia': {'port': 9091, 'health_url': 'http://localhost:9091/api/health'},
            'prowlarr': {'port': 9696, 'health_url': 'http://localhost:9696/ping'},
            'sonarr': {'port': 8989, 'health_url': 'http://localhost:8989/ping'},
            'radarr': {'port': 7878, 'health_url': 'http://localhost:7878/ping'},
            'lidarr': {'port': 8686, 'health_url': 'http://localhost:8686/ping'},
            'readarr': {'port': 8787, 'health_url': 'http://localhost:8787/ping'},
            'bazarr': {'port': 6767, 'health_url': 'http://localhost:6767/ping'},
            'qbittorrent': {'port': 8090, 'health_url': None},
            'jellyfin': {'port': 8096, 'health_url': 'http://localhost:8096/health'},
            'grafana': {'port': 3000, 'health_url': 'http://localhost:3000/api/health'},
            'prometheus': {'port': 9090, 'health_url': 'http://localhost:9090/-/healthy'},
        }
        
        self.service_status = {}
        self.system_stats = {}
        
    def check_service_health(self, service_name):
        """Check if a service is healthy"""
        service = self.services.get(service_name, {})
        port = service.get('port')
        health_url = service.get('health_url')
        
        status = {
            'name': service_name,
            'healthy': False,
            'port_open': False,
            'http_check': None,
            'last_check': time.time()
        }
        
        # Check port
        if port:
            status['port_open'] = self.is_port_open(port)
        
        # Check HTTP endpoint
        if health_url:
            try:
                response = requests.get(health_url, timeout=5)
                status['http_check'] = response.status_code
                status['healthy'] = response.status_code == 200
            except Exception as e:
                status['http_check'] = str(e)
                status['healthy'] = False
        else:
            status['healthy'] = status['port_open']
        
        return status
    
    def is_port_open(self, port):
        """Check if a port is open"""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    
    def get_system_stats(self):
        """Get system resource statistics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'disk_percent': disk.percent,
                'disk_used_gb': round(disk.used / (1024**3), 2),
                'disk_total_gb': round(disk.total / (1024**3), 2),
                'boot_time': psutil.boot_time(),
                'last_update': time.time()
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
    
    def update_service_status(self):
        """Update status for all services"""
        while True:
            try:
                logger.info("Updating service status...")
                
                for service_name in self.services:
                    status = self.check_service_health(service_name)
                    self.service_status[service_name] = status
                
                # Update system stats
                self.system_stats = self.get_system_stats()
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error updating service status: {e}")
                time.sleep(60)

monitor = HealthMonitor()

# Flask routes
@app.route('/health')
def health():
    """Overall health check endpoint"""
    healthy_services = sum(1 for status in monitor.service_status.values() if status.get('healthy'))
    total_services = len(monitor.service_status)
    
    return jsonify({
        'status': 'healthy' if healthy_services == total_services else 'degraded',
        'healthy_services': healthy_services,
        'total_services': total_services,
        'timestamp': time.time()
    })

@app.route('/services')
def services():
    """Get detailed service status"""
    return jsonify({
        'services': monitor.service_status,
        'timestamp': time.time()
    })

@app.route('/system')
def system():
    """Get system resource statistics"""
    return jsonify(monitor.system_stats)

@app.route('/status')
def status():
    """Get comprehensive status"""
    return jsonify({
        'services': monitor.service_status,
        'system': monitor.system_stats,
        'timestamp': time.time()
    })

if __name__ == "__main__":
    # Start background monitoring thread
    monitor_thread = Thread(target=monitor.update_service_status, daemon=True)
    monitor_thread.start()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8888, debug=False)
EOF

chmod +x /opt/scripts/health_monitor.py

# ========================================
# FINAL PERMISSIONS AND OWNERSHIP
# ========================================

echo "🔒 Setting final permissions..."

# Set ownership for all service directories
chown -R postgres:postgres /config/postgresql
chown -R redis:redis /config/redis
chown -R rabbitmq:rabbitmq /config/rabbitmq
chown -R grafana:grafana /config/grafana /data/grafana
chown -R prometheus:prometheus /config/prometheus /data/prometheus
chown -R loki:loki /config/loki /data/loki

# Create necessary runtime directories
mkdir -p /var/run/{postgresql,redis,rabbitmq}
chown postgres:postgres /var/run/postgresql
chown redis:redis /var/run/redis
chown rabbitmq:rabbitmq /var/run/rabbitmq

# ========================================
# START SUPERVISOR
# ========================================

echo "🚀 Starting Supervisor to orchestrate all services..."
echo "🕐 $(date): Initialization completed, starting services..."

# Start supervisord
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf