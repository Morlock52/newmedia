/**
 * Enhanced Health Checker with Circuit Breaker Pattern
 * Monitors all 30 media services with advanced health metrics
 */

export interface ServiceConfig {
  url: string
  auth: boolean
  timeout?: number
  retries?: number
  healthEndpoint?: string
  apiKey?: string
  credentials?: {
    username: string
    password: string
  }
}

export interface HealthStatus {
  service: string
  status: 'healthy' | 'unhealthy' | 'degraded' | 'unknown'
  responseTime: number
  lastCheck: string
  uptime: number
  errorCount: number
  circuitState: 'closed' | 'open' | 'half-open'
  metadata?: Record<string, any>
}

export interface CircuitBreakerState {
  failures: number
  lastFailure: Date | null
  state: 'closed' | 'open' | 'half-open'
  nextAttempt: Date | null
}

export class HealthChecker {
  private circuitBreakers = new Map<string, CircuitBreakerState>()
  private healthHistory = new Map<string, HealthStatus[]>()
  private readonly FAILURE_THRESHOLD = 5
  private readonly RECOVERY_TIMEOUT = 30000 // 30 seconds
  private readonly MAX_HISTORY = 100

  constructor(private services: Record<string, ServiceConfig>) {}

  async checkAllServices(): Promise<HealthStatus[]> {
    const results = await Promise.allSettled(
      Object.entries(this.services).map(([name, config]) => 
        this.checkService(name, config)
      )
    )

    return results.map((result, index) => {
      const serviceName = Object.keys(this.services)[index]
      if (result.status === 'fulfilled') {
        this.updateHistory(serviceName, result.value)
        return result.value
      } else {
        return this.createErrorStatus(serviceName, result.reason)
      }
    })
  }

  async checkService(name: string, config: ServiceConfig): Promise<HealthStatus> {
    const circuitBreaker = this.getCircuitBreaker(name)
    
    // Check circuit breaker state
    if (circuitBreaker.state === 'open') {
      if (Date.now() < (circuitBreaker.nextAttempt?.getTime() || 0)) {
        return this.createCircuitOpenStatus(name)
      } else {
        circuitBreaker.state = 'half-open'
      }
    }

    const startTime = Date.now()
    
    try {
      const healthEndpoint = config.healthEndpoint || this.getDefaultHealthEndpoint(name)
      const response = await this.makeHealthRequest(config.url + healthEndpoint, config)
      
      const responseTime = Date.now() - startTime
      const status = this.evaluateHealthResponse(response, responseTime)
      
      // Update circuit breaker on success
      if (status.status === 'healthy') {
        this.resetCircuitBreaker(name)
      } else {
        this.recordFailure(name)
      }
      
      return status
    } catch (error) {
      this.recordFailure(name)
      return this.createErrorStatus(name, error)
    }
  }

  private async makeHealthRequest(url: string, config: ServiceConfig): Promise<Response> {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), config.timeout || 5000)

    try {
      const headers: Record<string, string> = {
        'Accept': 'application/json',
        'User-Agent': 'Ultimate-Media-Server-2025'
      }

      // Add authentication if required
      if (config.auth && config.apiKey) {
        headers['X-Api-Key'] = config.apiKey
      } else if (config.auth && config.credentials) {
        const auth = btoa(`${config.credentials.username}:${config.credentials.password}`)
        headers['Authorization'] = `Basic ${auth}`
      }

      const response = await fetch(url, {
        method: 'GET',
        headers,
        signal: controller.signal
      })

      clearTimeout(timeout)
      return response
    } finally {
      clearTimeout(timeout)
    }
  }

  private evaluateHealthResponse(response: Response, responseTime: number): HealthStatus {
    const serviceName = new URL(response.url).hostname.split('.')[0]
    
    let status: HealthStatus['status'] = 'unknown'
    
    if (response.ok) {
      if (responseTime < 1000) {
        status = 'healthy'
      } else if (responseTime < 5000) {
        status = 'degraded'
      } else {
        status = 'unhealthy'
      }
    } else {
      status = 'unhealthy'
    }

    return {
      service: serviceName,
      status,
      responseTime,
      lastCheck: new Date().toISOString(),
      uptime: this.calculateUptime(serviceName),
      errorCount: this.getCircuitBreaker(serviceName).failures,
      circuitState: this.getCircuitBreaker(serviceName).state,
      metadata: {
        httpStatus: response.status,
        httpStatusText: response.statusText
      }
    }
  }

  private getCircuitBreaker(serviceName: string): CircuitBreakerState {
    if (!this.circuitBreakers.has(serviceName)) {
      this.circuitBreakers.set(serviceName, {
        failures: 0,
        lastFailure: null,
        state: 'closed',
        nextAttempt: null
      })
    }
    return this.circuitBreakers.get(serviceName)!
  }

  private recordFailure(serviceName: string): void {
    const breaker = this.getCircuitBreaker(serviceName)
    breaker.failures++
    breaker.lastFailure = new Date()

    if (breaker.failures >= this.FAILURE_THRESHOLD) {
      breaker.state = 'open'
      breaker.nextAttempt = new Date(Date.now() + this.RECOVERY_TIMEOUT)
    }
  }

  private resetCircuitBreaker(serviceName: string): void {
    const breaker = this.getCircuitBreaker(serviceName)
    breaker.failures = 0
    breaker.lastFailure = null
    breaker.state = 'closed'
    breaker.nextAttempt = null
  }

  private createErrorStatus(serviceName: string, error: any): HealthStatus {
    return {
      service: serviceName,
      status: 'unhealthy',
      responseTime: 0,
      lastCheck: new Date().toISOString(),
      uptime: 0,
      errorCount: this.getCircuitBreaker(serviceName).failures,
      circuitState: this.getCircuitBreaker(serviceName).state,
      metadata: {
        error: error?.message || 'Unknown error'
      }
    }
  }

  private createCircuitOpenStatus(serviceName: string): HealthStatus {
    return {
      service: serviceName,
      status: 'unhealthy',
      responseTime: 0,
      lastCheck: new Date().toISOString(),
      uptime: 0,
      errorCount: this.getCircuitBreaker(serviceName).failures,
      circuitState: 'open',
      metadata: {
        reason: 'Circuit breaker open'
      }
    }
  }

  private updateHistory(serviceName: string, status: HealthStatus): void {
    if (!this.healthHistory.has(serviceName)) {
      this.healthHistory.set(serviceName, [])
    }
    
    const history = this.healthHistory.get(serviceName)!
    history.push(status)
    
    if (history.length > this.MAX_HISTORY) {
      history.shift()
    }
  }

  private calculateUptime(serviceName: string): number {
    const history = this.healthHistory.get(serviceName) || []
    if (history.length === 0) return 0
    
    const healthyChecks = history.filter(h => h.status === 'healthy').length
    return (healthyChecks / history.length) * 100
  }

  private getDefaultHealthEndpoint(serviceName: string): string {
    const endpoints: Record<string, string> = {
      jellyfin: '/health',
      plex: '/identity',
      sonarr: '/api/v3/system/status',
      radarr: '/api/v3/system/status',
      lidarr: '/api/v1/system/status',
      bazarr: '/api/system/status',
      prowlarr: '/api/v1/system/status',
      qbittorrent: '/api/v2/app/version',
      transmission: '/transmission/rpc',
      sabnzbd: '/api?mode=version',
      overseerr: '/api/v1/status',
      jellyseerr: '/api/v1/status',
      tautulli: '/api/v2?cmd=get_server_info',
      portainer: '/api/status',
      grafana: '/api/health',
      prometheus: '/-/healthy',
      nzbget: '/jsonrpc',
      jackett: '/api/v2.0/server/config',
      ombi: '/api/v1/status',
      requestrr: '/api/health',
      varken: '/health',
      organizr: '/api/v2/status',
      heimdall: '/health',
      homer: '/health',
      uptimeKuma: '/api/status-page'
    }
    
    return endpoints[serviceName] || '/health'
  }

  getServiceHistory(serviceName: string): HealthStatus[] {
    return this.healthHistory.get(serviceName) || []
  }

  getCircuitBreakerStatus(serviceName: string): CircuitBreakerState {
    return this.getCircuitBreaker(serviceName)
  }

  async forceCircuitBreakerReset(serviceName: string): Promise<void> {
    this.resetCircuitBreaker(serviceName)
  }
}