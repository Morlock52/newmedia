# Single-Container Media Architecture Design
## Comprehensive 30+ Services Integration with AI Features

### Executive Summary
This document outlines a comprehensive single-container architecture integrating 30+ media services with AI features using s6-overlay for process management, service mesh for inter-process communication, and shared infrastructure components.

## 🏗️ Core Architecture Principles

### 1. Process Management Strategy
- **s6-overlay v3.2.1.0** as the primary process supervisor (replacing supervisord)
- Hierarchical service dependencies with proper startup sequencing  
- Graceful shutdown handling and auto-restart capabilities
- Resource isolation using Linux cgroups and namespaces

### 2. Service Mesh Design
- Internal API gateway using **Traefik** for service routing
- Service discovery via DNS and health check endpoints
- Circuit breaker patterns for fault tolerance
- Load balancing for scalable components

### 3. Shared Infrastructure Layer
- **PostgreSQL 16** as primary database (multi-database setup)
- **Redis 7** for caching and session storage
- **RabbitMQ** for async messaging between services
- **Elasticsearch** for centralized logging and search

## 📋 Service Inventory & Categories

### Media Server Core (3 services)
```yaml
jellyfin:
  port: 8096
  dependencies: [postgres, redis]
  resources: {cpu: "2000m", memory: "2Gi"}
  
plex:
  port: 32400
  dependencies: [postgres, redis]
  resources: {cpu: "1500m", memory: "1.5Gi"}
  
emby:
  port: 8097
  dependencies: [postgres, redis]
  resources: {cpu: "1000m", memory: "1Gi"}
```

### *arr Stack Management (6 services)
```yaml
sonarr:
  port: 8989
  dependencies: [postgres, prowlarr, qbittorrent]
  api_integration: true
  
radarr:
  port: 7878
  dependencies: [postgres, prowlarr, qbittorrent]
  api_integration: true
  
lidarr:
  port: 8686
  dependencies: [postgres, prowlarr, qbittorrent]
  api_integration: true
  
readarr:
  port: 8787
  dependencies: [postgres, prowlarr, qbittorrent]
  api_integration: true
  
prowlarr:
  port: 9696
  dependencies: [postgres]
  service_type: indexer_manager
  
bazarr:
  port: 6767
  dependencies: [sonarr, radarr]
  service_type: subtitle_manager
```

### Download Clients (4 services)
```yaml
qbittorrent:
  port: 8080
  dependencies: [redis]
  volume_mounts: [downloads, torrents]
  
transmission:
  port: 9091
  dependencies: [redis]
  volume_mounts: [downloads, torrents]
  
sabnzbd:
  port: 8081
  dependencies: [redis]
  volume_mounts: [downloads, usenet]
  
nzbget:
  port: 6789
  dependencies: [redis]
  volume_mounts: [downloads, usenet]
```

### Request Management (3 services)
```yaml
overseerr:
  port: 5055
  dependencies: [postgres, plex, sonarr, radarr]
  
jellyseerr:
  port: 5056
  dependencies: [postgres, jellyfin, sonarr, radarr]
  
ombi:
  port: 3579
  dependencies: [postgres, plex, emby]
```

### Management & Dashboards (3 services)
```yaml
tautulli:
  port: 8181
  dependencies: [postgres, plex]
  
organizr:
  port: 8083
  dependencies: [postgres]
  
heimdall:
  port: 8082
  dependencies: []
```

### AI Services (4 services)
```yaml
ai_safety_system:
  port: 8901
  dependencies: [postgres, redis, elasticsearch]
  resources: {cpu: "1000m", memory: "2Gi"}
  
content_moderation:
  port: 8902
  dependencies: [postgres, redis, ai_safety_system]
  resources: {cpu: "500m", memory: "1Gi"}
  
recommendation_engine:
  port: 8903
  dependencies: [postgres, redis, jellyfin, plex]
  resources: {cpu: "1000m", memory: "1.5Gi"}
  
social_media_integration:
  port: 8904
  dependencies: [postgres, redis, rabbitmq]
  resources: {cpu: "500m", memory: "512Mi"}
```

### Monitoring & Observability (4 services)
```yaml
prometheus:
  port: 9090
  dependencies: []
  scrape_targets: all_services
  
grafana:
  port: 3000
  dependencies: [postgres, prometheus]
  
uptime_kuma:
  port: 3001
  dependencies: [postgres]
  
elasticsearch:
  port: 9200
  dependencies: []
```

### Infrastructure Services (3 services)
```yaml
postgres:
  port: 5432
  databases: [jellyfin, plex, sonarr, radarr, grafana, overseerr]
  
redis:
  port: 6379
  databases: 16
  
rabbitmq:
  port: 5672
  management_port: 15672
```

## 🔧 s6-overlay Process Management Structure

### Service Tree Architecture
```
/etc/s6-overlay/s6-rc.d/
├── infrastructure/          # Tier 1: Core services
│   ├── postgres/
│   ├── redis/
│   └── rabbitmq/
├── platform/               # Tier 2: Platform services  
│   ├── traefik/
│   ├── elasticsearch/
│   └── auth_service/
├── media_core/             # Tier 3: Media servers
│   ├── jellyfin/
│   ├── plex/
│   └── emby/
├── media_management/       # Tier 4: Management services
│   ├── sonarr/
│   ├── radarr/
│   ├── prowlarr/
│   └── bazarr/
├── download_clients/       # Tier 4: Download clients
│   ├── qbittorrent/
│   ├── transmission/
│   ├── sabnzbd/
│   └── nzbget/
├── request_services/       # Tier 5: Request handlers
│   ├── overseerr/
│   ├── jellyseerr/
│   └── ombi/
├── ai_services/           # Tier 5: AI features
│   ├── ai_safety_system/
│   ├── content_moderation/
│   ├── recommendation_engine/
│   └── social_media_integration/
└── monitoring/            # Tier 6: Monitoring
    ├── prometheus/
    ├── grafana/
    └── uptime_kuma/
```

### Service Dependency Matrix
```yaml
tier_dependencies:
  infrastructure: []
  platform: [infrastructure]
  media_core: [infrastructure, platform]
  media_management: [infrastructure, platform, media_core]
  download_clients: [infrastructure, platform]
  request_services: [infrastructure, platform, media_core, media_management]
  ai_services: [infrastructure, platform, media_core]
  monitoring: [infrastructure, platform]
```

## 🌐 Service Mesh & Inter-Process Communication

### API Gateway Configuration (Traefik)
```yaml
traefik_config:
  entrypoints:
    web:
      address: ":80"
    websecure:
      address: ":443"
    
  routers:
    jellyfin:
      rule: "PathPrefix(`/jellyfin`) || Host(`jellyfin.local`)"
      service: jellyfin-service
      middlewares: [auth, cors]
      
    sonarr:
      rule: "PathPrefix(`/sonarr`) || Host(`sonarr.local`)"  
      service: sonarr-service
      middlewares: [auth, api-ratelimit]
      
    ai-api:
      rule: "PathPrefix(`/ai`) || Host(`ai.local`)"
      service: ai-gateway-service
      middlewares: [auth, ai-ratelimit]
  
  services:
    jellyfin-service:
      loadBalancer:
        servers:
          - url: "http://localhost:8096"
        healthCheck:
          path: "/health"
          interval: "30s"
```

### Inter-Service Communication Patterns
```yaml
communication_patterns:
  sync_api:
    - sonarr -> prowlarr (indexer queries)
    - radarr -> prowlarr (indexer queries)  
    - overseerr -> sonarr/radarr (media requests)
    - ai_services -> media_servers (content analysis)
    
  async_messaging:
    - download_complete -> media_servers (library refresh)
    - content_added -> ai_services (analysis queue)
    - user_activity -> social_media (sharing events)
    
  shared_storage:
    - postgres (configuration, metadata)
    - redis (sessions, cache, temp data)
    - filesystem (media, downloads, logs)
```

## 🗄️ Shared Infrastructure Design

### PostgreSQL Multi-Database Setup
```sql
-- Database allocation per service
CREATE DATABASE jellyfin_db;
CREATE DATABASE plex_db; 
CREATE DATABASE sonarr_db;
CREATE DATABASE radarr_db;
CREATE DATABASE lidarr_db;
CREATE DATABASE readarr_db;
CREATE DATABASE prowlarr_db;
CREATE DATABASE overseerr_db;
CREATE DATABASE jellyseerr_db;
CREATE DATABASE ombi_db;
CREATE DATABASE tautulli_db;
CREATE DATABASE grafana_db;
CREATE DATABASE ai_services_db;

-- Shared tables for cross-service data
CREATE DATABASE shared_db;
```

### Redis Database Allocation
```yaml
redis_databases:
  0: session_storage      # User sessions
  1: api_cache           # API response cache
  2: media_metadata      # Media information cache
  3: download_queue      # Download task queue
  4: ai_processing       # AI task queue
  5: notifications       # Notification queue
  6: user_preferences    # User settings cache
  7: system_stats        # System metrics cache
  8-15: reserved         # Future expansion
```

### RabbitMQ Queue Architecture
```yaml
queues:
  media_events:
    - library.scan.requested
    - library.item.added
    - library.item.updated
    
  download_events:
    - download.started
    - download.completed
    - download.failed
    
  ai_events:
    - content.analysis.requested
    - content.moderation.required
    - recommendation.update.needed
    
  user_events:
    - user.activity.tracked
    - user.preference.changed
    - social.share.requested
```

## 🔐 Unified Authentication & Authorization

### OAuth2/OIDC Integration Hub
```yaml
auth_service:
  providers:
    - local_users
    - ldap
    - google_oauth
    - github_oauth
    
  authorization_matrix:
    admin:
      - full_access_all_services
      - system_configuration
      - user_management
      
    power_user:
      - media_server_access
      - request_services_access
      - download_management
      - ai_features_access
      
    standard_user:
      - media_consumption
      - basic_requests
      - social_features
      
    guest:
      - limited_media_access
      - read_only_access
```

### Service Integration Points
```yaml
protected_services:
  jellyfin:
    auth_method: header_injection
    user_mapping: true
    
  sonarr/radarr:
    auth_method: api_key_proxy
    admin_required: true
    
  ai_services:
    auth_method: jwt_validation
    rbac_enabled: true
```

## 🤖 AI Services Integration Architecture

### AI Processing Pipeline
```yaml
ai_safety_system:
  models:
    - content_classifier
    - nsfw_detector
    - violence_detector
    - copyright_checker
    
  inputs:
    - new_media_files
    - user_uploads
    - external_content
    
  outputs:
    - safety_scores
    - content_flags
    - moderation_actions
    
content_moderation:
  automated_actions:
    - quarantine_suspicious_content
    - apply_content_warnings
    - block_prohibited_content
    
  human_review_queue:
    - borderline_content
    - appeal_requests
    - policy_violations

recommendation_engine:
  data_sources:
    - viewing_history (jellyfin/plex)
    - user_ratings
    - content_metadata
    - social_interactions
    
  algorithms:
    - collaborative_filtering
    - content_based_filtering
    - deep_learning_recommendations
    
  personalization:
    - user_preference_learning
    - contextual_recommendations
    - social_influence_modeling

social_media_integration:
  platforms:
    - twitter
    - facebook
    - discord
    - mastodon
    
  features:
    - auto_sharing
    - social_recommendations
    - friend_activity_feed
    - group_watch_coordination
```

## 🔍 Service Discovery & Health Monitoring

### Health Check Framework
```yaml
health_checks:
  infrastructure:
    postgres:
      endpoint: "tcp://localhost:5432"
      query: "SELECT 1"
      interval: 10s
      
    redis:
      endpoint: "tcp://localhost:6379"
      command: "PING"
      interval: 5s
      
  media_services:
    jellyfin:
      endpoint: "http://localhost:8096/health"
      expected_status: 200
      interval: 30s
      
    sonarr:
      endpoint: "http://localhost:8989/ping"
      expected_response: "pong"
      interval: 30s
```

### Service Registry & Discovery
```yaml
service_registry:
  consul_alternative: "internal_dns + health_checks"
  
  service_catalog:
    - name: jellyfin
      address: localhost:8096
      health_check: /health
      tags: [media-server, streaming]
      
    - name: sonarr-api
      address: localhost:8989
      health_check: /ping
      tags: [arr-stack, api, tv-management]
      
    - name: ai-safety
      address: localhost:8901
      health_check: /v1/health
      tags: [ai-service, content-moderation]
```

## 📊 Monitoring & Observability Architecture

### Metrics Collection Strategy
```yaml
prometheus_config:
  global:
    scrape_interval: 15s
    evaluation_interval: 15s
    
  scrape_configs:
    - job_name: media-services
      static_configs:
        - targets: 
          - localhost:8096  # jellyfin
          - localhost:8989  # sonarr
          - localhost:7878  # radarr
      metrics_path: /metrics
      scrape_interval: 30s
      
    - job_name: ai-services
      static_configs:
        - targets:
          - localhost:8901  # ai-safety
          - localhost:8903  # recommendations
      metrics_path: /v1/metrics
      scrape_interval: 60s
```

### Logging Architecture
```yaml
logging_pipeline:
  collection:
    - filebeat (log files)
    - fluentd (application logs)
    - docker_logs (container output)
    
  processing:
    - elasticsearch (indexing & search)
    - logstash (parsing & enrichment)
    
  visualization:
    - grafana (dashboards)
    - kibana (log analysis)
```

## 🌐 Internal Networking & Port Allocation

### Port Management Strategy
```yaml
port_ranges:
  infrastructure: 3000-3999, 5000-6999, 9000-9999
  media_servers: 8090-8199  
  arr_services: 7870-7899, 8980-8999, 9690-9699
  download_clients: 6780-6799, 8070-8099, 9090-9099
  request_services: 3570-3599, 5050-5099
  ai_services: 8900-8999
  monitoring: 3000-3099, 9090-9199
```

### Internal Service Communication
```yaml
network_topology:
  internal_dns:
    jellyfin.internal: 127.0.0.1:8096
    sonarr.internal: 127.0.0.1:8989
    postgres.internal: 127.0.0.1:5432
    
  service_mesh:
    type: traefik_proxy
    load_balancing: round_robin
    circuit_breaker: enabled
    rate_limiting: per_service_config
```

## 💾 Storage Architecture & Volume Management

### Shared Volume Strategy
```yaml
volume_structure:
  config:
    path: /config
    services: all
    backup_strategy: daily_incremental
    
  data:
    media: /data/media/{movies,tv,music,books}
    downloads: /data/downloads/{complete,incomplete}
    cache: /data/cache/{service_name}
    
  logs:
    path: /var/log
    rotation: daily
    retention: 30_days
    
  databases:
    postgres: /var/lib/postgresql/data
    redis: /var/lib/redis
    elasticsearch: /var/lib/elasticsearch
```

## 🚀 Startup Sequence & Dependencies

### Tiered Startup Process
```yaml
startup_sequence:
  tier_1_infrastructure:
    order: 1
    services: [postgres, redis, rabbitmq, elasticsearch]
    wait_strategy: health_check_pass
    timeout: 120s
    
  tier_2_platform:
    order: 2
    services: [traefik, auth_service]
    dependencies: [tier_1_infrastructure]
    timeout: 60s
    
  tier_3_media_core:
    order: 3
    services: [jellyfin, plex, emby]
    dependencies: [tier_1_infrastructure, tier_2_platform]
    timeout: 90s
    
  tier_4_management:
    order: 4
    services: [sonarr, radarr, prowlarr, qbittorrent]
    dependencies: [tier_3_media_core]
    timeout: 60s
    
  tier_5_features:
    order: 5
    services: [overseerr, ai_services]
    dependencies: [tier_4_management]
    timeout: 45s
    
  tier_6_monitoring:
    order: 6
    services: [prometheus, grafana, uptime_kuma]
    dependencies: [all_previous_tiers]
    timeout: 30s
```

## 🔒 Security Considerations

### Container Security
```yaml
security_measures:
  process_isolation:
    - separate_user_per_service
    - minimal_capabilities
    - read_only_filesystems_where_possible
    
  network_security:
    - internal_service_communication_only
    - tls_encryption_for_external_apis
    - firewall_rules_for_port_access
    
  data_protection:
    - encrypted_database_connections
    - secure_secret_management
    - regular_security_updates
```

## 📈 Performance Optimization

### Resource Management
```yaml
resource_limits:
  cpu_intensive:
    jellyfin: 2000m
    plex: 1500m
    ai_services: 1000m each
    
  memory_intensive:
    media_servers: 1-2Gi
    databases: 512Mi-1Gi
    ai_services: 1-2Gi
    
  i_o_intensive:
    download_clients: optimized_disk_access
    media_servers: ssd_cache_layer
```

### Caching Strategy
```yaml
caching_layers:
  redis_cache:
    - api_responses (TTL: 5min)
    - user_sessions (TTL: 24h) 
    - media_metadata (TTL: 1h)
    
  application_cache:
    - thumbnail_generation
    - transcoding_segments
    - search_results
```

## 🏁 Deployment Strategy

### Build Process
```dockerfile
FROM debian:bookworm-slim

# Install s6-overlay v3.2.1.0
ADD s6-overlay-installer.sh /tmp/
RUN /tmp/s6-overlay-installer.sh

# Install infrastructure services
RUN install-postgres.sh && \
    install-redis.sh && \
    install-rabbitmq.sh

# Install media services
RUN install-media-servers.sh && \
    install-arr-stack.sh && \
    install-download-clients.sh

# Install AI services
RUN install-ai-services.sh

# Configure s6 services
COPY s6-services/ /etc/s6-overlay/s6-rc.d/

# Configure service mesh
COPY traefik.yml /etc/traefik/
COPY service-configs/ /etc/services/

EXPOSE 80 443 8096 32400 8989 7878 9696

ENTRYPOINT ["/init"]
```

### Runtime Configuration
```yaml
container_run:
  environment:
    - PUID=1000
    - PGID=1000
    - TZ=UTC
    - S6_BEHAVIOUR_IF_STAGE2_FAILS=2
    
  volumes:
    - ./config:/config
    - ./data:/data
    - ./logs:/var/log
    
  ports:
    - "80:80"      # Main web interface
    - "443:443"    # HTTPS
    - "8096:8096"  # Jellyfin direct
    - "32400:32400" # Plex direct
```

## 📋 Summary & Benefits

### Architecture Benefits
1. **Single Container Deployment** - Simplified deployment and management
2. **Service Mesh Integration** - Reliable inter-service communication
3. **Hierarchical Dependencies** - Proper startup sequencing and fault isolation
4. **AI-Enhanced Features** - Modern content analysis and recommendations
5. **Unified Authentication** - Single sign-on across all services
6. **Comprehensive Monitoring** - Full observability stack included
7. **Scalable Design** - Can be broken apart into microservices later

### Resource Requirements
- **CPU**: 8+ cores recommended (4 minimum)
- **Memory**: 16GB+ recommended (8GB minimum)  
- **Storage**: 100GB+ for system, unlimited for media
- **Network**: Gigabit ethernet recommended

### Maintenance Considerations
- Regular security updates through automated patching
- Database maintenance and optimization
- Log rotation and cleanup
- Performance monitoring and tuning
- Backup and disaster recovery procedures

This architecture provides a comprehensive, production-ready media server solution with modern AI features, proper service mesh integration, and enterprise-grade monitoring capabilities, all contained within a single, manageable container.