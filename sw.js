// Performance-Optimized Service Worker
// Implements advanced caching strategies for media server dashboard

const CACHE_NAME = 'media-server-v1.0.0';
const RUNTIME_CACHE = 'runtime-cache-v1.0.0';
const API_CACHE = 'api-cache-v1.0.0';

// Resources to cache immediately
const PRECACHE_RESOURCES = [
    '/',
    '/performance-optimized-dashboard.html',
    '/manifest.json',
    // Add critical resources
];

// API endpoints to cache
const API_CACHE_PATTERNS = [
    /\/health$/,
    /\/api\/services/,
    /\/api\/status/,
];

// Cache-first strategy for static assets
const CACHE_FIRST_PATTERNS = [
    /\.(?:css|js|ico|png|jpg|jpeg|svg|gif|woff|woff2)$/,
];

// Network-first strategy for dynamic content
const NETWORK_FIRST_PATTERNS = [
    /\/api\//,
    /localhost:\d+/,
];

self.addEventListener('install', event => {
    console.log('Service Worker: Installing...');
    
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('Service Worker: Precaching resources');
                return cache.addAll(PRECACHE_RESOURCES);
            })
            .then(() => {
                console.log('Service Worker: Installation complete');
                return self.skipWaiting();
            })
    );
});

self.addEventListener('activate', event => {
    console.log('Service Worker: Activating...');
    
    event.waitUntil(
        caches.keys()
            .then(cacheNames => {
                return Promise.all(
                    cacheNames
                        .filter(cacheName => {
                            return cacheName.startsWith('media-server-') && 
                                   cacheName !== CACHE_NAME &&
                                   cacheName !== RUNTIME_CACHE &&
                                   cacheName !== API_CACHE;
                        })
                        .map(cacheName => {
                            console.log('Service Worker: Deleting old cache', cacheName);
                            return caches.delete(cacheName);
                        })
                );
            })
            .then(() => {
                console.log('Service Worker: Activation complete');
                return self.clients.claim();
            })
    );
});

self.addEventListener('fetch', event => {
    const { request } = event;
    const url = new URL(request.url);
    
    // Skip non-GET requests
    if (request.method !== 'GET') {
        return;
    }
    
    // Skip chrome-extension requests
    if (url.protocol === 'chrome-extension:') {
        return;
    }
    
    // Handle different request types
    if (shouldUseApiCache(request)) {
        event.respondWith(handleApiRequest(request));
    } else if (shouldUseCacheFirst(request)) {
        event.respondWith(handleCacheFirst(request));
    } else if (shouldUseNetworkFirst(request)) {
        event.respondWith(handleNetworkFirst(request));
    } else {
        event.respondWith(handleDefault(request));
    }
});

// Cache strategy implementations
async function handleApiRequest(request) {
    const cache = await caches.open(API_CACHE);
    
    try {
        // Try network first for fresh data
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok) {
            // Cache successful responses for 30 seconds
            const responseClone = networkResponse.clone();
            const headers = new Headers(responseClone.headers);
            headers.set('sw-cache-timestamp', Date.now());
            
            const modifiedResponse = new Response(await responseClone.blob(), {
                status: responseClone.status,
                statusText: responseClone.statusText,
                headers: headers
            });
            
            cache.put(request, modifiedResponse);
        }
        
        return networkResponse;
    } catch (error) {
        console.log('Service Worker: Network failed, trying cache');
        
        const cachedResponse = await cache.match(request);
        if (cachedResponse) {
            const cacheTimestamp = cachedResponse.headers.get('sw-cache-timestamp');
            const age = Date.now() - parseInt(cacheTimestamp || 0);
            
            // Use cached response if less than 30 seconds old
            if (age < 30000) {
                return cachedResponse;
            }
        }
        
        // Return a fallback response for API endpoints
        return new Response(JSON.stringify({
            error: 'Service unavailable',
            message: 'Unable to connect to service',
            cached: false
        }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

async function handleCacheFirst(request) {
    const cache = await caches.open(CACHE_NAME);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
        // Update cache in background
        updateCacheInBackground(request, cache);
        return cachedResponse;
    }
    
    try {
        const networkResponse = await fetch(request);
        if (networkResponse.ok) {
            cache.put(request, networkResponse.clone());
        }
        return networkResponse;
    } catch (error) {
        console.log('Service Worker: Cache-first failed', error);
        return new Response('Resource not available', { status: 404 });
    }
}

async function handleNetworkFirst(request) {
    const cache = await caches.open(RUNTIME_CACHE);
    
    try {
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok) {
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        console.log('Service Worker: Network-first failed, trying cache');
        
        const cachedResponse = await cache.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }
        
        return new Response('Service unavailable', { status: 503 });
    }
}

async function handleDefault(request) {
    const cache = await caches.open(RUNTIME_CACHE);
    
    try {
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok) {
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        const cachedResponse = await cache.match(request);
        return cachedResponse || new Response('Not found', { status: 404 });
    }
}

// Helper functions
function shouldUseApiCache(request) {
    return API_CACHE_PATTERNS.some(pattern => pattern.test(request.url));
}

function shouldUseCacheFirst(request) {
    return CACHE_FIRST_PATTERNS.some(pattern => pattern.test(request.url));
}

function shouldUseNetworkFirst(request) {
    return NETWORK_FIRST_PATTERNS.some(pattern => pattern.test(request.url));
}

async function updateCacheInBackground(request, cache) {
    try {
        const networkResponse = await fetch(request);
        if (networkResponse.ok) {
            cache.put(request, networkResponse);
        }
    } catch (error) {
        console.log('Service Worker: Background cache update failed', error);
    }
}

// Background sync for offline functionality
self.addEventListener('sync', event => {
    console.log('Service Worker: Background sync triggered');
    
    if (event.tag === 'service-status-sync') {
        event.waitUntil(syncServiceStatus());
    }
});

async function syncServiceStatus() {
    console.log('Service Worker: Syncing service status');
    
    const services = ['8096', '8989', '7878', '9696', '8080'];
    const results = {};
    
    for (const port of services) {
        try {
            const response = await fetch(`http://localhost:${port}`, {
                mode: 'no-cors'
            });
            results[port] = { online: true, timestamp: Date.now() };
        } catch (error) {
            results[port] = { online: false, timestamp: Date.now() };
        }
    }
    
    // Store results in IndexedDB or broadcast to clients
    const clients = await self.clients.matchAll();
    clients.forEach(client => {
        client.postMessage({
            type: 'SERVICE_STATUS_UPDATE',
            data: results
        });
    });
}

// Handle push notifications for service status changes
self.addEventListener('push', event => {
    if (!event.data) return;
    
    const data = event.data.json();
    
    const options = {
        body: data.body || 'Service status update',
        icon: '/icon-192.png',
        badge: '/badge-72.png',
        vibrate: [200, 100, 200],
        data: data.data || {},
        actions: [
            {
                action: 'view',
                title: 'View Dashboard',
                icon: '/icon-view.png'
            },
            {
                action: 'dismiss',
                title: 'Dismiss',
                icon: '/icon-dismiss.png'
            }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification(data.title || 'Media Server Update', options)
    );
});

// Handle notification clicks
self.addEventListener('notificationclick', event => {
    const { action, notification } = event;
    
    event.notification.close();
    
    if (action === 'view') {
        event.waitUntil(
            clients.openWindow('/')
        );
    }
});

// Performance monitoring
self.addEventListener('message', event => {
    if (event.data && event.data.type === 'PERFORMANCE_METRICS') {
        console.log('Service Worker: Received performance metrics', event.data.metrics);
        
        // Store metrics for analysis
        caches.open('performance-metrics').then(cache => {
            const metricsResponse = new Response(JSON.stringify({
                timestamp: Date.now(),
                metrics: event.data.metrics
            }));
            
            cache.put('/performance-metrics-' + Date.now(), metricsResponse);
        });
    }
});

console.log('Service Worker: Loaded successfully');