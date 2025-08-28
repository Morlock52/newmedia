const CACHE_NAME = 'ultimate-media-server-2025-v1'
const urlsToCache = [
  '/',
  '/media',
  '/downloads',
  '/requests',
  '/analytics',
  '/settings',
  '/admin',
  '/manifest.json',
  '/icons/icon-192x192.png',
  '/icons/icon-512x512.png'
]

// Install Service Worker
self.addEventListener('install', (event) => {
  console.log('Service Worker installing...')
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('Opened cache')
        return cache.addAll(urlsToCache)
      })
      .catch((error) => {
        console.error('Cache installation failed:', error)
      })
  )
})

// Fetch event
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        // Return cached version or fetch from network
        return response || fetch(event.request)
      })
      .catch(() => {
        // Fallback for offline pages
        if (event.request.destination === 'document') {
          return caches.match('/')
        }
      })
  )
})

// Activate Service Worker
self.addEventListener('activate', (event) => {
  console.log('Service Worker activating...')
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('Deleting old cache:', cacheName)
            return caches.delete(cacheName)
          }
        })
      )
    })
  )
})

// Background Sync for offline downloads
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-download') {
    event.waitUntil(
      // Handle background download sync
      handleBackgroundDownload()
    )
  }
})

// Push notifications
self.addEventListener('push', (event) => {
  const options = {
    body: event.data ? event.data.text() : 'New media server notification',
    icon: '/icons/icon-192x192.png',
    badge: '/icons/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'View Dashboard',
        icon: '/icons/dashboard-shortcut.png'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/icons/close-action.png'
      }
    ]
  }

  event.waitUntil(
    self.registration.showNotification('Ultimate Media Server', options)
  )
})

// Notification click
self.addEventListener('notificationclick', (event) => {
  event.notification.close()

  if (event.action === 'explore') {
    event.waitUntil(
      clients.openWindow('/')
    )
  } else if (event.action === 'close') {
    // Just close the notification
    return
  } else {
    event.waitUntil(
      clients.openWindow('/')
    )
  }
})

// Message handling
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting()
  }
})

// Background download handling
async function handleBackgroundDownload() {
  try {
    // Sync download status with server
    const response = await fetch('/api/gateway?action=status')
    if (response.ok) {
      const data = await response.json()
      
      // Send notification if downloads completed
      if (data.completedDownloads > 0) {
        self.registration.showNotification('Downloads Complete', {
          body: `${data.completedDownloads} downloads finished`,
          icon: '/icons/icon-192x192.png',
          tag: 'download-complete'
        })
      }
    }
  } catch (error) {
    console.error('Background sync failed:', error)
  }
}

// Network-first strategy for API calls
const networkFirst = async (request) => {
  try {
    const response = await fetch(request)
    const cache = await caches.open(CACHE_NAME)
    cache.put(request, response.clone())
    return response
  } catch (error) {
    const cache = await caches.open(CACHE_NAME)
    return await cache.match(request)
  }
}

// Cache-first strategy for static assets
const cacheFirst = async (request) => {
  const cache = await caches.open(CACHE_NAME)
  const cached = await cache.match(request)
  
  if (cached) {
    return cached
  }
  
  try {
    const response = await fetch(request)
    cache.put(request, response.clone())
    return response
  } catch (error) {
    console.error('Network request failed:', error)
    throw error
  }
}