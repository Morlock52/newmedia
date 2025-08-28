# Ultimate Media Server Performance Optimization Guide 2025

## 🚀 Overview

This guide provides comprehensive performance optimization strategies for achieving sub-second page loads, smooth media streaming, and efficient resource utilization in your media server stack.

## 📊 Performance Targets

- **Page Load Time**: < 1 second
- **Media Start Time**: < 2 seconds
- **API Response Time**: < 200ms
- **Cache Hit Rate**: > 80%
- **CPU Usage**: < 70% under normal load
- **Memory Usage**: < 80% of available RAM

## 🏗️ Architecture Components

### 1. Edge Gateway (Nginx)
- HTTP/3 support for faster connections
- Brotli compression for smaller payloads
- Smart caching policies
- Connection pooling
- Request rate limiting

### 2. Cache Layer (Varnish)
- 4GB in-memory cache
- Intelligent cache invalidation
- Grace mode for stale content
- ESI support for dynamic content

### 3. Application Cache (Redis)
- Session management
- API response caching
- Real-time data storage
- Pub/sub for live updates

### 4. CDN Integration (Cloudflare)
- Global edge caching
- Image optimization
- DDoS protection
- Analytics and monitoring

## 🔧 Quick Start

### Deploy the Performance Stack

```bash
# Clone the repository
git clone <your-repo>
cd newmedia

# Deploy the performance-optimized stack
./deploy-performance-stack.sh
```

### System Optimization (Run as root)

```bash
sudo ./scripts/optimize-performance.sh
```

## ⚡ Performance Features

### 1. Advanced Caching Strategy

#### Static Assets
- **Images**: Cached for 30 days with automatic WebP conversion
- **CSS/JS**: Cached for 30 days with minification
- **Fonts**: Cached for 1 year

#### Media Content
- **Video files**: Cached for 7 days at edge
- **HLS/DASH segments**: 5-minute cache for live content
- **Thumbnails**: Indefinite caching with versioning

#### API Responses
- **GET requests**: 60-second cache for non-authenticated
- **Search results**: 5-minute cache
- **Metadata**: 1-hour cache

### 2. GPU Acceleration

Supports multiple GPU vendors:
- **NVIDIA**: NVENC/NVDEC hardware encoding
- **Intel**: Quick Sync Video
- **AMD**: VCE/VCN encoding

### 3. Database Optimization

PostgreSQL tuning:
- Connection pooling
- Query optimization
- Index management
- Vacuum scheduling

### 4. Network Optimization

- Jumbo frames (MTU 9000)
- BBR congestion control
- TCP Fast Open
- HTTP/3 with QUIC

## 📈 Monitoring & Analytics

### Grafana Dashboards

Access at: http://localhost:3000

1. **Media Server Performance**: Overall system metrics
2. **Streaming Analytics**: Playback quality and buffering
3. **Cache Performance**: Hit rates and efficiency
4. **Resource Usage**: CPU, memory, disk, network

### Key Metrics to Monitor

#### Response Times
```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

#### Cache Hit Rate
```promql
sum(rate(varnish_cache_hit[5m])) / 
(sum(rate(varnish_cache_hit[5m])) + sum(rate(varnish_cache_miss[5m])))
```

#### Active Streams
```promql
sum(jellyfin_active_streams) + sum(plex_active_streams)
```

## 🛠️ Optimization Techniques

### 1. Content Delivery Optimization

#### Enable Cloudflare CDN
```yaml
# config/cloudflare/cdn-config.yml
page_rules:
  - url_pattern: "*.{jpg,jpeg,png,gif,webp}"
    actions:
      cache_level: "cache_everything"
      edge_cache_ttl: 2592000
      polish: "lossless"
      mirage: "on"
```

#### Configure Varnish for Media
```vcl
# config/varnish/default.vcl
sub vcl_backend_response {
    if (bereq.url ~ "\.(mp4|webm|mkv)$") {
        set beresp.ttl = 7d;
        set beresp.do_stream = true;
    }
}
```

### 2. Database Query Optimization

#### Create Indexes
```sql
-- Optimize media queries
CREATE INDEX idx_media_title ON media(title);
CREATE INDEX idx_media_year ON media(year);
CREATE INDEX idx_media_added ON media(date_added DESC);

-- Optimize user activity
CREATE INDEX idx_activity_user_time ON activity(user_id, timestamp DESC);
```

#### Enable Query Caching
```sql
-- In postgresql.conf
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.track = all
```

### 3. Media Transcoding Optimization

#### Configure Hardware Acceleration
```yaml
# Jellyfin settings
JELLYFIN_FFmpeg__hwaccel=nvenc
JELLYFIN_FFmpeg__hwaccel_output_format=cuda

# Plex settings
TRANSCODE_HW=1
TRANSCODE_DEVICE=/dev/dri/renderD128
```

#### Optimize Transcoding Settings
```bash
# Pre-transcode popular content
./scripts/pre-transcode.sh --quality=high --formats=h265,h264
```

### 4. Network Performance Tuning

#### Enable BBR Congestion Control
```bash
echo "net.core.default_qdisc=fq" >> /etc/sysctl.conf
echo "net.ipv4.tcp_congestion_control=bbr" >> /etc/sysctl.conf
sysctl -p
```

#### Optimize Network Buffers
```bash
echo "net.core.rmem_max=134217728" >> /etc/sysctl.conf
echo "net.core.wmem_max=134217728" >> /etc/sysctl.conf
sysctl -p
```

## 🔍 Troubleshooting Performance Issues

### High CPU Usage

1. Check transcoding activity:
```bash
docker exec jellyfin ps aux | grep ffmpeg
```

2. Review container resource usage:
```bash
docker stats --no-stream
```

3. Analyze CPU bottlenecks:
```bash
htop -d 10
```

### Slow Page Loads

1. Check cache hit rates:
```bash
curl -I http://localhost/test.jpg | grep X-Cache
```

2. Analyze response times:
```bash
curl -w "@curl-format.txt" -o /dev/null -s http://localhost
```

3. Review nginx access logs:
```bash
tail -f logs/nginx/access.log | grep -E "time=[0-9]+\.[0-9]+"
```

### Memory Issues

1. Check memory usage:
```bash
free -h
docker system df
```

2. Clear caches if needed:
```bash
docker exec redis_cache redis-cli FLUSHALL
docker exec varnish varnishadm "ban req.url ~ ."
```

## 📊 Performance Testing

### Load Testing with K6

```javascript
// performance-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },
    { duration: '5m', target: 100 },
    { duration: '2m', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<1000'],
  },
};

export default function() {
  let response = http.get('http://localhost');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'page loaded in < 1s': (r) => r.timings.duration < 1000,
  });
  sleep(1);
}
```

Run the test:
```bash
k6 run performance-test.js
```

### Monitoring Script

```bash
#!/bin/bash
# monitor-performance.sh

while true; do
  clear
  echo "=== Performance Monitor ==="
  echo
  echo "Cache Hit Rate:"
  curl -s http://localhost:9090/api/v1/query?query=cache_hit_rate | jq '.data.result[0].value[1]'
  echo
  echo "Active Streams:"
  curl -s http://localhost:9090/api/v1/query?query=active_streams | jq '.data.result[0].value[1]'
  echo
  echo "CPU Usage:"
  docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
  sleep 5
done
```

## 🚀 Advanced Optimizations

### 1. Enable HTTP/3
```nginx
# nginx.conf
listen 443 quic reuseport;
add_header Alt-Svc 'h3=":443"; ma=86400';
```

### 2. Implement Service Worker
```javascript
// sw.js - Progressive Web App caching
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});
```

### 3. Database Read Replicas
```yaml
# docker-compose addition
postgres-replica:
  image: postgres:16-alpine
  environment:
    - POSTGRES_REPLICATION_MODE=slave
    - POSTGRES_MASTER_HOST=postgres
```

### 4. Content Preloading
```javascript
// Predictive prefetching
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const link = entry.target.querySelector('a');
      if (link) prefetch(link.href);
    }
  });
});
```

## 📋 Maintenance Schedule

### Daily
- Monitor cache hit rates
- Check error logs
- Review resource usage

### Weekly
- Clear old cache entries
- Update container images
- Backup performance data

### Monthly
- Analyze performance trends
- Optimize database indexes
- Review CDN analytics
- Update security patches

## 🔗 Additional Resources

- [Nginx Performance Tuning](https://www.nginx.com/blog/tuning-nginx/)
- [Varnish Best Practices](https://varnish-cache.org/docs/6.0/users-guide/performance.html)
- [PostgreSQL Optimization](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Docker Performance Guide](https://docs.docker.com/config/containers/resource_constraints/)

## 📞 Support

For performance issues or optimization questions:
1. Check Grafana dashboards for metrics
2. Review logs in `/logs` directory
3. Run performance diagnostics: `./scripts/diagnose-performance.sh`
4. Consult the troubleshooting section above

---

**Remember**: Performance optimization is an iterative process. Start with the basics, measure results, and gradually apply advanced optimizations based on your specific needs.