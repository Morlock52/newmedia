/**
 * Service Discovery with Load Balancing
 * Manages service registration, discovery, and health-aware load balancing
 */

export interface ServiceInstance {
  id: string
  name: string
  host: string
  port: number
  protocol: 'http' | 'https'
  health: 'healthy' | 'unhealthy' | 'unknown'
  metadata: Record<string, any>
  lastSeen: Date
  weight: number
}

export interface LoadBalancingStrategy {
  type: 'round-robin' | 'weighted' | 'least-connections' | 'health-aware'
  healthThreshold: number
}

export class ServiceDiscovery {
  private services = new Map<string, ServiceInstance[]>()
  private roundRobinCounters = new Map<string, number>()
  private connectionCounts = new Map<string, number>()
  private healthChecker: any // Will be injected
  
  constructor(
    private strategy: LoadBalancingStrategy = {
      type: 'health-aware',
      healthThreshold: 80
    }
  ) {}

  /**
   * Register a service instance
   */
  registerService(instance: ServiceInstance): void {
    const serviceName = instance.name
    
    if (!this.services.has(serviceName)) {
      this.services.set(serviceName, [])
    }
    
    const instances = this.services.get(serviceName)!
    const existingIndex = instances.findIndex(i => i.id === instance.id)
    
    if (existingIndex >= 0) {
      instances[existingIndex] = { ...instance, lastSeen: new Date() }
    } else {
      instances.push({ ...instance, lastSeen: new Date() })
    }
  }

  /**
   * Deregister a service instance
   */
  deregisterService(serviceName: string, instanceId: string): void {
    const instances = this.services.get(serviceName)
    if (instances) {
      const filtered = instances.filter(i => i.id !== instanceId)
      this.services.set(serviceName, filtered)
    }
  }

  /**
   * Get a service instance using load balancing strategy
   */
  getServiceInstance(serviceName: string): ServiceInstance | null {
    const instances = this.getHealthyInstances(serviceName)
    if (instances.length === 0) return null

    switch (this.strategy.type) {
      case 'round-robin':
        return this.getRoundRobinInstance(serviceName, instances)
      case 'weighted':
        return this.getWeightedInstance(instances)
      case 'least-connections':
        return this.getLeastConnectionsInstance(instances)
      case 'health-aware':
        return this.getHealthAwareInstance(instances)
      default:
        return instances[0]
    }
  }

  /**
   * Get all instances of a service
   */
  getAllInstances(serviceName: string): ServiceInstance[] {
    return this.services.get(serviceName) || []
  }

  /**
   * Get only healthy instances
   */
  getHealthyInstances(serviceName: string): ServiceInstance[] {
    const instances = this.services.get(serviceName) || []
    return instances.filter(instance => 
      instance.health === 'healthy' || 
      (instance.health === 'unknown' && this.isRecentlySeen(instance))
    )
  }

  /**
   * Update service health status
   */
  updateServiceHealth(serviceName: string, instanceId: string, health: ServiceInstance['health']): void {
    const instances = this.services.get(serviceName)
    if (instances) {
      const instance = instances.find(i => i.id === instanceId)
      if (instance) {
        instance.health = health
        instance.lastSeen = new Date()
      }
    }
  }

  /**
   * Get service topology for visualization
   */
  getServiceTopology(): Record<string, ServiceInstance[]> {
    const topology: Record<string, ServiceInstance[]> = {}
    
    for (const [serviceName, instances] of this.services.entries()) {
      topology[serviceName] = instances.map(instance => ({
        ...instance,
        health: this.isRecentlySeen(instance) ? instance.health : 'unknown'
      }))
    }
    
    return topology
  }

  /**
   * Get load balancing statistics
   */
  getLoadBalancingStats(): Record<string, any> {
    const stats: Record<string, any> = {}
    
    for (const [serviceName, instances] of this.services.entries()) {
      const healthyCount = instances.filter(i => i.health === 'healthy').length
      const totalCount = instances.length
      
      stats[serviceName] = {
        totalInstances: totalCount,
        healthyInstances: healthyCount,
        healthPercentage: totalCount > 0 ? (healthyCount / totalCount) * 100 : 0,
        strategy: this.strategy.type,
        roundRobinPosition: this.roundRobinCounters.get(serviceName) || 0,
        connections: this.connectionCounts.get(serviceName) || 0
      }
    }
    
    return stats
  }

  /**
   * Clean up stale service instances
   */
  cleanupStaleServices(maxAge: number = 300000): void { // 5 minutes
    const now = new Date()
    
    for (const [serviceName, instances] of this.services.entries()) {
      const activeInstances = instances.filter(instance => 
        (now.getTime() - instance.lastSeen.getTime()) < maxAge
      )
      
      if (activeInstances.length !== instances.length) {
        this.services.set(serviceName, activeInstances)
      }
    }
  }

  private getRoundRobinInstance(serviceName: string, instances: ServiceInstance[]): ServiceInstance {
    const counter = this.roundRobinCounters.get(serviceName) || 0
    const instance = instances[counter % instances.length]
    this.roundRobinCounters.set(serviceName, counter + 1)
    return instance
  }

  private getWeightedInstance(instances: ServiceInstance[]): ServiceInstance {
    const totalWeight = instances.reduce((sum, instance) => sum + instance.weight, 0)
    let random = Math.random() * totalWeight
    
    for (const instance of instances) {
      random -= instance.weight
      if (random <= 0) {
        return instance
      }
    }
    
    return instances[0]
  }

  private getLeastConnectionsInstance(instances: ServiceInstance[]): ServiceInstance {
    return instances.reduce((least, current) => {
      const leastConnections = this.connectionCounts.get(least.id) || 0
      const currentConnections = this.connectionCounts.get(current.id) || 0
      return currentConnections < leastConnections ? current : least
    })
  }

  private getHealthAwareInstance(instances: ServiceInstance[]): ServiceInstance {
    // Filter by health threshold first
    const highQualityInstances = instances.filter(instance => {
      const health = this.getInstanceHealthScore(instance)
      return health >= this.strategy.healthThreshold
    })
    
    const candidateInstances = highQualityInstances.length > 0 ? highQualityInstances : instances
    
    // Apply weighted selection based on health score
    return this.getWeightedHealthInstance(candidateInstances)
  }

  private getWeightedHealthInstance(instances: ServiceInstance[]): ServiceInstance {
    const weightedInstances = instances.map(instance => ({
      ...instance,
      adjustedWeight: instance.weight * (this.getInstanceHealthScore(instance) / 100)
    }))
    
    const totalWeight = weightedInstances.reduce((sum, instance) => sum + instance.adjustedWeight, 0)
    let random = Math.random() * totalWeight
    
    for (const instance of weightedInstances) {
      random -= instance.adjustedWeight
      if (random <= 0) {
        return instance
      }
    }
    
    return instances[0]
  }

  private getInstanceHealthScore(instance: ServiceInstance): number {
    // Base score from health status
    let score = instance.health === 'healthy' ? 100 : 
                instance.health === 'unknown' ? 50 : 0
    
    // Adjust for recency
    const age = Date.now() - instance.lastSeen.getTime()
    const ageSeconds = age / 1000
    
    if (ageSeconds < 30) {
      score = Math.min(100, score + 10) // Bonus for recent updates
    } else if (ageSeconds > 300) {
      score = Math.max(0, score - 20) // Penalty for stale data
    }
    
    return score
  }

  private isRecentlySeen(instance: ServiceInstance): boolean {
    const age = Date.now() - instance.lastSeen.getTime()
    return age < 300000 // 5 minutes
  }

  /**
   * Track connection for least-connections load balancing
   */
  incrementConnection(instanceId: string): void {
    const current = this.connectionCounts.get(instanceId) || 0
    this.connectionCounts.set(instanceId, current + 1)
  }

  /**
   * Release connection for least-connections load balancing
   */
  decrementConnection(instanceId: string): void {
    const current = this.connectionCounts.get(instanceId) || 0
    this.connectionCounts.set(instanceId, Math.max(0, current - 1))
  }

  /**
   * Auto-register services from configuration
   */
  autoRegisterServices(serviceConfigs: Record<string, any>): void {
    for (const [serviceName, config] of Object.entries(serviceConfigs)) {
      const url = new URL(config.url)
      
      this.registerService({
        id: `${serviceName}-primary`,
        name: serviceName,
        host: url.hostname,
        port: parseInt(url.port) || (url.protocol === 'https:' ? 443 : 80),
        protocol: url.protocol.replace(':', '') as 'http' | 'https',
        health: 'unknown',
        metadata: {
          auth: config.auth,
          apiKey: config.apiKey,
          version: config.version || 'unknown'
        },
        lastSeen: new Date(),
        weight: config.weight || 1
      })
    }
  }
}