# Deployment Guide

This guide covers deploying the Holographic Media Dashboard in various environments.

## 🚀 Quick Deployment

### Static Hosting (Recommended for demo)

Deploy to any static hosting service:

#### Netlify
1. Connect your GitHub repository
2. Build settings: None required (static files)
3. Deploy directory: `/` (root)
4. Deploy!

#### Vercel
1. Import GitHub repository
2. Framework preset: None
3. Root directory: `/`
4. Deploy!

#### GitHub Pages
1. Enable GitHub Pages in repository settings
2. Source: Deploy from branch `main`
3. Folder: `/` (root)
4. Access at: `https://yourusername.github.io/holographic-dashboard`

### Docker Deployment

```dockerfile
# Dockerfile
FROM nginx:alpine

# Copy static files
COPY . /usr/share/nginx/html

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

```bash
# Build and run
docker build -t holographic-dashboard .
docker run -p 8080:80 holographic-dashboard
```

## 🌐 Production Deployment

### Prerequisites

- Modern web server (Nginx, Apache, or Caddy)
- HTTPS certificate (required for WebGL features)
- CDN for static assets (optional but recommended)

### Nginx Configuration

```nginx
# /etc/nginx/sites-available/holographic-dashboard
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Document root
    root /var/www/holographic-dashboard;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_types text/css application/javascript application/json image/svg+xml;
    gzip_min_length 1000;

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Main application
    location / {
        try_files $uri $uri/ /index.html;
        
        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    }

    # WebSocket proxy (if using backend)
    location /ws {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Apache Configuration

```apache
# .htaccess
RewriteEngine On

# Force HTTPS
RewriteCond %{HTTPS} off
RewriteRule ^(.*)$ https://%{HTTP_HOST}%{REQUEST_URI} [L,R=301]

# Gzip compression
<IfModule mod_deflate.c>
    AddOutputFilterByType DEFLATE text/plain
    AddOutputFilterByType DEFLATE text/html
    AddOutputFilterByType DEFLATE text/xml
    AddOutputFilterByType DEFLATE text/css
    AddOutputFilterByType DEFLATE application/xml
    AddOutputFilterByType DEFLATE application/xhtml+xml
    AddOutputFilterByType DEFLATE application/rss+xml
    AddOutputFilterByType DEFLATE application/javascript
    AddOutputFilterByType DEFLATE application/x-javascript
</IfModule>

# Cache static assets
<IfModule mod_expires.c>
    ExpiresActive on
    ExpiresByType text/css "access plus 1 year"
    ExpiresByType application/javascript "access plus 1 year"
    ExpiresByType image/png "access plus 1 year"
    ExpiresByType image/jpg "access plus 1 year"
    ExpiresByType image/jpeg "access plus 1 year"
    ExpiresByType image/gif "access plus 1 year"
    ExpiresByType image/svg+xml "access plus 1 year"
</IfModule>

# Security headers
<IfModule mod_headers.c>
    Header always set X-Frame-Options "SAMEORIGIN"
    Header always set X-Content-Type-Options "nosniff"
    Header always set X-XSS-Protection "1; mode=block"
    Header always set Referrer-Policy "strict-origin-when-cross-origin"
</IfModule>
```

## 🔧 Environment Configuration

### Environment Variables

Create `.env` file for environment-specific settings:

```bash
# .env
NODE_ENV=production
WEBSOCKET_URL=wss://your-media-server.com/ws
API_BASE_URL=https://your-api.com
ENABLE_ANALYTICS=true
DEBUG_MODE=false
```

### Configuration Management

```javascript
// js/config.prod.js
const CONFIG = {
    websocket: {
        url: process.env.WEBSOCKET_URL || 'wss://localhost:8080',
        reconnectInterval: 5000
    },
    
    performance: {
        shadowsEnabled: true,
        antialias: true,
        adaptiveQuality: true
    },
    
    analytics: {
        enabled: process.env.ENABLE_ANALYTICS === 'true',
        trackingId: process.env.ANALYTICS_ID
    }
};
```

## 📊 CDN Integration

### CloudFlare Setup

1. Add your domain to CloudFlare
2. Enable these optimizations:
   - Auto Minify (CSS, HTML, JS)
   - Brotli compression
   - Browser Cache TTL: 4 hours
   - Rocket Loader (optional, test first)

### AWS CloudFront

```json
{
    "Origins": [{
        "Id": "holographic-dashboard",
        "DomainName": "your-origin.com",
        "CustomOriginConfig": {
            "HTTPPort": 443,
            "OriginProtocolPolicy": "https-only"
        }
    }],
    "DefaultCacheBehavior": {
        "TargetOriginId": "holographic-dashboard",
        "ViewerProtocolPolicy": "redirect-to-https",
        "Compress": true,
        "CachePolicyId": "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"
    }
}
```

## 🔒 Security Considerations

### Content Security Policy

Add to HTML head or server headers:

```html
<meta http-equiv="Content-Security-Policy" content="
    default-src 'self';
    script-src 'self' 'unsafe-eval' https://cdnjs.cloudflare.com;
    style-src 'self' 'unsafe-inline';
    img-src 'self' data: https:;
    connect-src 'self' ws: wss:;
    font-src 'self' https://fonts.gstatic.com;
    worker-src 'self' blob:;
">
```

### HTTPS Requirements

WebGL and WebSocket features require HTTPS:

- Use Let's Encrypt for free certificates
- Configure HTTP to HTTPS redirects
- Enable HSTS headers
- Consider certificate pinning for high-security deployments

## 📈 Performance Optimization

### Build Optimization

```bash
# Minify JavaScript
npx terser js/*.js --compress --mangle --output dist/app.min.js

# Optimize CSS
npx csso css/*.css --output dist/styles.min.css

# Optimize images
npx imagemin assets/**/*.{jpg,png,svg} --out-dir=dist/assets
```

### Resource Optimization

1. **Lazy Loading**: Load non-critical resources after initial render
2. **Code Splitting**: Split large JavaScript files
3. **Tree Shaking**: Remove unused code
4. **Image Optimization**: Use WebP format where supported
5. **Font Loading**: Use font-display: swap

### Performance Monitoring

```javascript
// Add performance monitoring
const observer = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
        if (entry.entryType === 'navigation') {
            console.log('Page load time:', entry.loadEventEnd - entry.fetchStart);
        }
    }
});

observer.observe({ entryTypes: ['navigation'] });
```

## 🐳 Container Deployment

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  dashboard:
    build: .
    ports:
      - "8080:80"
    environment:
      - NODE_ENV=production
    volumes:
      - ./logs:/var/log/nginx
    restart: unless-stopped

  media-server:
    image: your-media-server:latest
    ports:
      - "8081:8080"
    environment:
      - WEBSOCKET_PORT=8080
    restart: unless-stopped
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: holographic-dashboard
spec:
  replicas: 3
  selector:
    matchLabels:
      app: holographic-dashboard
  template:
    metadata:
      labels:
        app: holographic-dashboard
    spec:
      containers:
      - name: dashboard
        image: holographic-dashboard:latest
        ports:
        - containerPort: 80
        env:
        - name: NODE_ENV
          value: "production"
---
apiVersion: v1
kind: Service
metadata:
  name: dashboard-service
spec:
  selector:
    app: holographic-dashboard
  ports:
  - protocol: TCP
    port: 80
    targetPort: 80
  type: LoadBalancer
```

## 📱 Mobile Optimization

### PWA Configuration

```json
// manifest.json
{
    "name": "Holographic Media Dashboard",
    "short_name": "HoloDash",
    "description": "3D Media Server Dashboard",
    "start_url": "/",
    "display": "fullscreen",
    "background_color": "#0a0a0a",
    "theme_color": "#00ffff",
    "icons": [
        {
            "src": "assets/icon-192.png",
            "sizes": "192x192",
            "type": "image/png"
        },
        {
            "src": "assets/icon-512.png",
            "sizes": "512x512",
            "type": "image/png"
        }
    ]
}
```

### Service Worker

```javascript
// sw.js
const CACHE_NAME = 'holographic-dashboard-v1';
const urlsToCache = [
    '/',
    '/css/main.css',
    '/js/main.js',
    '/js/lib/three.min.js'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request)
            .then((response) => response || fetch(event.request))
    );
});
```

## 🔧 Monitoring & Analytics

### Error Tracking

```javascript
// Add error tracking
window.addEventListener('error', (event) => {
    fetch('/api/errors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message: event.message,
            filename: event.filename,
            lineno: event.lineno,
            colno: event.colno,
            stack: event.error?.stack,
            userAgent: navigator.userAgent,
            url: window.location.href,
            timestamp: Date.now()
        })
    });
});
```

### Performance Metrics

```javascript
// Track WebGL performance
const trackPerformance = () => {
    const info = renderer.info;
    
    fetch('/api/metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            fps: performanceMonitor.getFPS(),
            triangles: info.render.triangles,
            calls: info.render.calls,
            memory: info.memory.geometries + info.memory.textures,
            timestamp: Date.now()
        })
    });
};

setInterval(trackPerformance, 60000); // Every minute
```

## 🚨 Troubleshooting

### Common Issues

1. **WebGL Context Lost**: Implement context restoration
2. **CORS Errors**: Configure proper CORS headers
3. **WebSocket Connection Fails**: Check firewall and proxy settings
4. **Poor Performance**: Enable adaptive quality settings
5. **Assets Not Loading**: Verify file paths and permissions

### Debug Mode

Enable debug mode for troubleshooting:

```javascript
// Add to config
const CONFIG = {
    debug: {
        enabled: true,
        showStats: true,
        logLevel: 'verbose'
    }
};
```

### Health Checks

```javascript
// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: Date.now(),
        version: '2.0.0',
        uptime: process.uptime()
    });
});
```

## 📋 Deployment Checklist

- [ ] HTTPS configured and working
- [ ] Static assets optimized and compressed
- [ ] Cache headers configured
- [ ] Security headers in place
- [ ] Error tracking implemented
- [ ] Performance monitoring active
- [ ] Mobile responsiveness tested
- [ ] Browser compatibility verified
- [ ] WebSocket connection tested
- [ ] CDN configured (if applicable)
- [ ] Backup strategy in place
- [ ] Monitoring alerts configured

---

For additional support, please check the [API documentation](API.md) or open an issue on GitHub.