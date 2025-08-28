import { NextRequest, NextResponse } from 'next/server'
import { headers } from 'next/headers'

// Service endpoints configuration
const SERVICES = {
  jellyfin: { url: 'http://localhost:8096', auth: false },
  plex: { url: 'http://localhost:32400', auth: true },
  sonarr: { url: 'http://localhost:8989', auth: true },
  radarr: { url: 'http://localhost:7878', auth: true },
  lidarr: { url: 'http://localhost:8686', auth: true },
  bazarr: { url: 'http://localhost:6767', auth: true },
  prowlarr: { url: 'http://localhost:9696', auth: true },
  qbittorrent: { url: 'http://localhost:8080', auth: true },
  transmission: { url: 'http://localhost:9091', auth: true },
  overseerr: { url: 'http://localhost:5055', auth: false },
  jellyseerr: { url: 'http://localhost:5056', auth: false },
  tautulli: { url: 'http://localhost:8181', auth: true },
  portainer: { url: 'http://localhost:9000', auth: true },
  grafana: { url: 'http://localhost:3000', auth: true },
  prometheus: { url: 'http://localhost:9090', auth: false },
  sabnzbd: { url: 'http://localhost:8082', auth: true },
  nzbget: { url: 'http://localhost:6789', auth: true },
  jackett: { url: 'http://localhost:9117', auth: true },
  ombi: { url: 'http://localhost:3579', auth: false },
  requestrr: { url: 'http://localhost:4545', auth: true },
  varken: { url: 'http://localhost:5454', auth: false },
  organizr: { url: 'http://localhost:8080', auth: true },
  heimdall: { url: 'http://localhost:8443', auth: false },
  homer: { url: 'http://localhost:8092', auth: false },
  uptimeKuma: { url: 'http://localhost:3001', auth: false }
}

interface ServiceResponse {
  service: string
  status: 'online' | 'offline' | 'error'
  data?: any
  error?: string
  responseTime?: number
}

// Health check endpoint
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const service = searchParams.get('service')
  const endpoint = searchParams.get('endpoint')
  const action = searchParams.get('action') || 'health'

  try {
    switch (action) {
      case 'health':
        return await handleHealthCheck(service)
      case 'status':
        return await handleStatusCheck()
      case 'proxy':
        return await handleProxyRequest(service, endpoint, request)
      default:
        return NextResponse.json({ error: 'Invalid action' }, { status: 400 })
    }
  } catch (error) {
    console.error('API Gateway error:', error)
    return NextResponse.json({ 
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}

// POST method for service actions
export async function POST(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const service = searchParams.get('service')
  const action = searchParams.get('action')
  
  try {
    const body = await request.json()
    
    switch (action) {
      case 'command':
        return await handleServiceCommand(service, body)
      case 'search':
        return await handleSearch(service, body)
      case 'download':
        return await handleDownload(service, body)
      default:
        return NextResponse.json({ error: 'Invalid action' }, { status: 400 })
    }
  } catch (error) {
    console.error('POST API Gateway error:', error)
    return NextResponse.json({ 
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}

async function handleHealthCheck(serviceName: string | null): Promise<NextResponse> {
  if (serviceName && SERVICES[serviceName as keyof typeof SERVICES]) {
    const result = await checkSingleService(serviceName)
    return NextResponse.json(result)
  }

  // Check all services
  const results = await Promise.allSettled(
    Object.keys(SERVICES).map(service => checkSingleService(service))
  )

  const serviceStatuses = results.map((result, index) => {
    const serviceName = Object.keys(SERVICES)[index]
    if (result.status === 'fulfilled') {
      return result.value
    } else {
      return {
        service: serviceName,
        status: 'error' as const,
        error: result.reason?.message || 'Unknown error'
      }
    }
  })

  return NextResponse.json({
    timestamp: new Date().toISOString(),
    services: serviceStatuses,
    summary: {
      total: serviceStatuses.length,
      online: serviceStatuses.filter(s => s.status === 'online').length,
      offline: serviceStatuses.filter(s => s.status === 'offline').length,
      error: serviceStatuses.filter(s => s.status === 'error').length
    }
  })
}

async function checkSingleService(serviceName: string): Promise<ServiceResponse> {
  const service = SERVICES[serviceName as keyof typeof SERVICES]
  if (!service) {
    return {
      service: serviceName,
      status: 'error',
      error: 'Service not configured'
    }
  }

  const startTime = Date.now()
  
  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout

    const response = await fetch(`${service.url}/api/v1/system/status`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'User-Agent': 'Ultimate-Media-Server-2025'
      },
      signal: controller.signal
    })

    clearTimeout(timeoutId)
    const responseTime = Date.now() - startTime

    if (response.ok) {
      const data = await response.json()
      return {
        service: serviceName,
        status: 'online',
        data,
        responseTime
      }
    } else {
      return {
        service: serviceName,
        status: 'offline',
        error: `HTTP ${response.status}`,
        responseTime
      }
    }
  } catch (error) {
    const responseTime = Date.now() - startTime
    
    if (error instanceof Error && error.name === 'AbortError') {
      return {
        service: serviceName,
        status: 'offline',
        error: 'Timeout',
        responseTime
      }
    }

    return {
      service: serviceName,
      status: 'offline',
      error: error instanceof Error ? error.message : 'Connection failed',
      responseTime
    }
  }
}

async function handleStatusCheck(): Promise<NextResponse> {
  // Return system-wide status information
  const systemInfo = {
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    version: '2025.1.0',
    environment: process.env.NODE_ENV || 'development',
    features: {
      webSockets: true,
      realTimeUpdates: true,
      graphQL: true,
      authentication: true,
      caching: true
    }
  }

  return NextResponse.json(systemInfo)
}

async function handleProxyRequest(
  serviceName: string | null, 
  endpoint: string | null, 
  request: NextRequest
): Promise<NextResponse> {
  if (!serviceName || !endpoint) {
    return NextResponse.json({ error: 'Service and endpoint required' }, { status: 400 })
  }

  const service = SERVICES[serviceName as keyof typeof SERVICES]
  if (!service) {
    return NextResponse.json({ error: 'Service not found' }, { status: 404 })
  }

  try {
    const targetUrl = `${service.url}${endpoint}`
    const response = await fetch(targetUrl, {
      method: request.method,
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'User-Agent': 'Ultimate-Media-Server-2025'
      }
    })

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    return NextResponse.json({ 
      error: 'Proxy request failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 502 })
  }
}

async function handleServiceCommand(serviceName: string | null, body: any): Promise<NextResponse> {
  if (!serviceName) {
    return NextResponse.json({ error: 'Service name required' }, { status: 400 })
  }

  const service = SERVICES[serviceName as keyof typeof SERVICES]
  if (!service) {
    return NextResponse.json({ error: 'Service not found' }, { status: 404 })
  }

  // Mock service command execution
  const mockResult = {
    service: serviceName,
    command: body.command,
    status: 'success',
    timestamp: new Date().toISOString(),
    result: `${body.command} executed successfully on ${serviceName}`
  }

  return NextResponse.json(mockResult)
}

async function handleSearch(serviceName: string | null, body: any): Promise<NextResponse> {
  if (!serviceName) {
    return NextResponse.json({ error: 'Service name required' }, { status: 400 })
  }

  // Mock search results
  const mockResults = {
    service: serviceName,
    query: body.query,
    results: [
      {
        id: '1',
        title: `${body.query} - Sample Result 1`,
        year: 2023,
        type: body.type || 'movie',
        quality: '2160p',
        size: '15.2 GB',
        seeders: 45,
        leechers: 12
      },
      {
        id: '2',
        title: `${body.query} - Sample Result 2`,
        year: 2023,
        type: body.type || 'movie',
        quality: '1080p',
        size: '8.7 GB',
        seeders: 89,
        leechers: 3
      }
    ],
    timestamp: new Date().toISOString()
  }

  return NextResponse.json(mockResults)
}

async function handleDownload(serviceName: string | null, body: any): Promise<NextResponse> {
  if (!serviceName) {
    return NextResponse.json({ error: 'Service name required' }, { status: 400 })
  }

  // Mock download initiation
  const mockDownload = {
    service: serviceName,
    id: `download_${Date.now()}`,
    title: body.title,
    status: 'queued',
    progress: 0,
    timestamp: new Date().toISOString()
  }

  return NextResponse.json(mockDownload)
}