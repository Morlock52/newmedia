/**
 * Unit Tests for HealthChecker
 * Tests circuit breaker patterns, health monitoring, and service discovery
 */

import { HealthChecker, ServiceConfig } from '../../src/lib/health/HealthChecker'

// Mock fetch for testing
global.fetch = jest.fn()

describe('HealthChecker', () => {
  let healthChecker: HealthChecker
  let mockServices: Record<string, ServiceConfig>

  beforeEach(() => {
    mockServices = {
      jellyfin: {
        url: 'http://localhost:8096',
        auth: false,
        timeout: 5000
      },
      sonarr: {
        url: 'http://localhost:8989',
        auth: true,
        apiKey: 'test-api-key',
        timeout: 3000
      }
    }
    
    healthChecker = new HealthChecker(mockServices)
    jest.clearAllMocks()
  })

  describe('checkService', () => {
    it('should return healthy status for successful response', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        json: () => Promise.resolve({ status: 'running' }),
        url: 'http://localhost:8096/health'
      }
      
      ;(fetch as jest.Mock).mockResolvedValueOnce(mockResponse)

      const result = await healthChecker.checkService('jellyfin', mockServices.jellyfin)

      expect(result.status).toBe('healthy')
      expect(result.service).toBe('jellyfin')
      expect(result.responseTime).toBeGreaterThan(0)
      expect(result.circuitState).toBe('closed')
    })

    it('should return unhealthy status for failed response', async () => {
      const mockResponse = {
        ok: false,
        status: 500,
        json: () => Promise.resolve({}),
        url: 'http://localhost:8096/health'
      }
      
      ;(fetch as jest.Mock).mockResolvedValueOnce(mockResponse)

      const result = await healthChecker.checkService('jellyfin', mockServices.jellyfin)

      expect(result.status).toBe('unhealthy')
      expect(result.metadata?.httpStatus).toBe(500)
    })

    it('should handle network errors gracefully', async () => {
      ;(fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'))

      const result = await healthChecker.checkService('jellyfin', mockServices.jellyfin)

      expect(result.status).toBe('unhealthy')
      expect(result.metadata?.error).toBe('Network error')
    })

    it('should trigger circuit breaker after threshold failures', async () => {
      // Mock 5 consecutive failures
      ;(fetch as jest.Mock).mockRejectedValue(new Error('Service down'))

      // Trigger failures
      for (let i = 0; i < 5; i++) {
        await healthChecker.checkService('jellyfin', mockServices.jellyfin)
      }

      // Next check should show circuit breaker open
      const result = await healthChecker.checkService('jellyfin', mockServices.jellyfin)
      expect(result.circuitState).toBe('open')
    })

    it('should respect timeout configuration', async () => {
      const slowResponse = new Promise((resolve) => {
        setTimeout(() => resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({}),
          url: 'http://localhost:8096/health'
        }), 6000) // 6 seconds, longer than 5 second timeout
      })

      ;(fetch as jest.Mock).mockReturnValueOnce(slowResponse)

      const result = await healthChecker.checkService('jellyfin', mockServices.jellyfin)
      expect(result.status).toBe('unhealthy')
      expect(result.metadata?.error).toContain('Timeout')
    })
  })

  describe('checkAllServices', () => {
    it('should check all configured services', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        json: () => Promise.resolve({ status: 'running' }),
        url: 'http://localhost:8096/health'
      }
      
      ;(fetch as jest.Mock).mockResolvedValue(mockResponse)

      const results = await healthChecker.checkAllServices()

      expect(results).toHaveLength(2)
      expect(results[0].service).toBe('jellyfin')
      expect(results[1].service).toBe('sonarr')
      expect(fetch).toHaveBeenCalledTimes(2)
    })

    it('should handle partial service failures', async () => {
      ;(fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: () => Promise.resolve({}),
          url: 'http://localhost:8096/health'
        })
        .mockRejectedValueOnce(new Error('Service down'))

      const results = await healthChecker.checkAllServices()

      expect(results).toHaveLength(2)
      expect(results[0].status).toBe('healthy')
      expect(results[1].status).toBe('unhealthy')
    })
  })

  describe('Circuit Breaker', () => {
    it('should reset circuit breaker on successful request after failures', async () => {
      // Cause some failures but not enough to open circuit
      ;(fetch as jest.Mock).mockRejectedValueOnce(new Error('Temporary failure'))
      await healthChecker.checkService('jellyfin', mockServices.jellyfin)

      // Then succeed
      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({}),
        url: 'http://localhost:8096/health'
      })

      const result = await healthChecker.checkService('jellyfin', mockServices.jellyfin)
      expect(result.status).toBe('healthy')
      expect(result.errorCount).toBe(0) // Should be reset
    })

    it('should force reset circuit breaker', async () => {
      // Trigger circuit breaker
      ;(fetch as jest.Mock).mockRejectedValue(new Error('Service down'))
      for (let i = 0; i < 5; i++) {
        await healthChecker.checkService('jellyfin', mockServices.jellyfin)
      }

      // Force reset
      await healthChecker.forceCircuitBreakerReset('jellyfin')

      const status = healthChecker.getCircuitBreakerStatus('jellyfin')
      expect(status.state).toBe('closed')
      expect(status.failures).toBe(0)
    })
  })

  describe('Health History', () => {
    it('should maintain health history for services', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        json: () => Promise.resolve({}),
        url: 'http://localhost:8096/health'
      }
      
      ;(fetch as jest.Mock).mockResolvedValue(mockResponse)

      // Make multiple checks
      await healthChecker.checkService('jellyfin', mockServices.jellyfin)
      await healthChecker.checkService('jellyfin', mockServices.jellyfin)

      const history = healthChecker.getServiceHistory('jellyfin')
      expect(history).toHaveLength(2)
      expect(history[0].status).toBe('healthy')
      expect(history[1].status).toBe('healthy')
    })

    it('should limit history size', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        json: () => Promise.resolve({}),
        url: 'http://localhost:8096/health'
      }
      
      ;(fetch as jest.Mock).mockResolvedValue(mockResponse)

      // Make more than max history checks
      for (let i = 0; i < 105; i++) {
        await healthChecker.checkService('jellyfin', mockServices.jellyfin)
      }

      const history = healthChecker.getServiceHistory('jellyfin')
      expect(history.length).toBeLessThanOrEqual(100)
    })
  })
})