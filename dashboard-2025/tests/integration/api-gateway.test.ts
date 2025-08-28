/**
 * Integration Tests for API Gateway
 * Tests complete service integration, health checks, and API endpoints
 */

import request from 'supertest'
import { NextApiRequest, NextApiResponse } from 'next'
import { createMocks } from 'node-mocks-http'

// Mock the actual services
jest.mock('../../src/lib/services/ServiceConnectors')

describe('API Gateway Integration', () => {
  describe('Health Check Endpoints', () => {
    it('should return health status for all services', async () => {
      const { req, res } = createMocks<NextApiRequest, NextApiResponse>({
        method: 'GET',
        url: '/api/gateway?action=health'
      })

      // Mock successful responses for all services
      global.fetch = jest.fn()
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ status: 'running' }),
          url: 'http://localhost:8096/health'
        })
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ status: 'online' }),
          url: 'http://localhost:32400/identity'
        })

      // Import and execute the API route
      const { GET } = await import('../../src/app/api/gateway/route')
      const response = await GET(req as any)
      const data = await response.json()

      expect(response.status).toBe(200)
      expect(data.services).toBeDefined()
      expect(data.summary).toBeDefined()
      expect(data.summary.total).toBeGreaterThan(0)
    })

    it('should return health status for specific service', async () => {
      const { req, res } = createMocks<NextApiRequest, NextApiResponse>({
        method: 'GET',
        url: '/api/gateway?action=health&service=jellyfin'
      })

      global.fetch = jest.fn().mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ status: 'running' }),
        url: 'http://localhost:8096/health'
      })

      const { GET } = await import('../../src/app/api/gateway/route')
      const response = await GET(req as any)
      const data = await response.json()

      expect(response.status).toBe(200)
      expect(data.service).toBe('jellyfin')
      expect(data.status).toBe('online')
    })

    it('should handle service timeouts gracefully', async () => {
      const { req, res } = createMocks<NextApiRequest, NextApiResponse>({
        method: 'GET',
        url: '/api/gateway?action=health&service=jellyfin'
      })

      // Mock timeout
      global.fetch = jest.fn().mockImplementation(() => 
        new Promise((resolve) => setTimeout(resolve, 6000))
      )

      const { GET } = await import('../../src/app/api/gateway/route')
      const response = await GET(req as any)
      const data = await response.json()

      expect(response.status).toBe(200)
      expect(data.status).toBe('offline')
      expect(data.error).toContain('Timeout')
    })
  })

  describe('Service Proxy Endpoints', () => {
    it('should proxy requests to Jellyfin', async () => {
      const { req, res } = createMocks<NextApiRequest, NextApiResponse>({
        method: 'GET',
        url: '/api/gateway?action=proxy&service=jellyfin&endpoint=/Users'
      })

      global.fetch = jest.fn().mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve([
          { Id: '1', Name: 'Test User' }
        ])
      })

      const { GET } = await import('../../src/app/api/gateway/route')
      const response = await GET(req as any)
      const data = await response.json()

      expect(response.status).toBe(200)
      expect(Array.isArray(data)).toBe(true)
      expect(data[0].Name).toBe('Test User')
    })

    it('should handle invalid service proxy requests', async () => {
      const { req, res } = createMocks<NextApiRequest, NextApiResponse>({
        method: 'GET',
        url: '/api/gateway?action=proxy&service=invalid&endpoint=/test'
      })

      const { GET } = await import('../../src/app/api/gateway/route')
      const response = await GET(req as any)
      const data = await response.json()

      expect(response.status).toBe(404)
      expect(data.error).toBe('Service not found')
    })
  })

  describe('Service Commands', () => {
    it('should execute service commands', async () => {
      const { req, res } = createMocks<NextApiRequest, NextApiResponse>({
        method: 'POST',
        url: '/api/gateway?action=command&service=jellyfin',
        body: { command: 'scan-library' }
      })

      const { POST } = await import('../../src/app/api/gateway/route')
      const response = await POST(req as any)
      const data = await response.json()

      expect(response.status).toBe(200)
      expect(data.service).toBe('jellyfin')
      expect(data.command).toBe('scan-library')
      expect(data.status).toBe('success')
    })

    it('should handle search requests', async () => {
      const { req, res } = createMocks<NextApiRequest, NextApiResponse>({
        method: 'POST',
        url: '/api/gateway?action=search&service=prowlarr',
        body: { query: 'test movie', type: 'movie' }
      })

      const { POST } = await import('../../src/app/api/gateway/route')
      const response = await POST(req as any)
      const data = await response.json()

      expect(response.status).toBe(200)
      expect(data.service).toBe('prowlarr')
      expect(data.query).toBe('test movie')
      expect(Array.isArray(data.results)).toBe(true)
    })

    it('should handle download requests', async () => {
      const { req, res } = createMocks<NextApiRequest, NextApiResponse>({
        method: 'POST',
        url: '/api/gateway?action=download&service=qbittorrent',
        body: { title: 'Test Movie', magnetLink: 'magnet:?xt=...' }
      })

      const { POST } = await import('../../src/app/api/gateway/route')
      const response = await POST(req as any)
      const data = await response.json()

      expect(response.status).toBe(200)
      expect(data.service).toBe('qbittorrent')
      expect(data.title).toBe('Test Movie')
      expect(data.status).toBe('queued')
    })
  })

  describe('System Status', () => {
    it('should return system information', async () => {
      const { req, res } = createMocks<NextApiRequest, NextApiResponse>({
        method: 'GET',
        url: '/api/gateway?action=status'
      })

      const { GET } = await import('../../src/app/api/gateway/route')
      const response = await GET(req as any)
      const data = await response.json()

      expect(response.status).toBe(200)
      expect(data.version).toBe('2025.1.0')
      expect(data.features).toBeDefined()
      expect(data.features.webSockets).toBe(true)
      expect(data.features.authentication).toBe(true)
    })
  })

  describe('Error Handling', () => {
    it('should handle invalid action parameters', async () => {
      const { req, res } = createMocks<NextApiRequest, NextApiResponse>({
        method: 'GET',
        url: '/api/gateway?action=invalid'
      })

      const { GET } = await import('../../src/app/api/gateway/route')
      const response = await GET(req as any)
      const data = await response.json()

      expect(response.status).toBe(400)
      expect(data.error).toBe('Invalid action')
    })

    it('should handle missing required parameters', async () => {
      const { req, res } = createMocks<NextApiRequest, NextApiResponse>({
        method: 'GET',
        url: '/api/gateway?action=proxy&service=jellyfin'
      })

      const { GET } = await import('../../src/app/api/gateway/route')
      const response = await GET(req as any)
      const data = await response.json()

      expect(response.status).toBe(400)
      expect(data.error).toBe('Service and endpoint required')
    })

    it('should handle internal server errors', async () => {
      const { req, res } = createMocks<NextApiRequest, NextApiResponse>({
        method: 'GET',
        url: '/api/gateway?action=health'
      })

      // Mock fetch to throw an error
      global.fetch = jest.fn().mockRejectedValue(new Error('Internal error'))

      const { GET } = await import('../../src/app/api/gateway/route')
      const response = await GET(req as any)
      const data = await response.json()

      expect(response.status).toBe(500)
      expect(data.error).toBe('Internal server error')
    })
  })

  describe('Service Integration', () => {
    it('should integrate with multiple services simultaneously', async () => {
      const { req, res } = createMocks<NextApiRequest, NextApiResponse>({
        method: 'GET',
        url: '/api/gateway?action=health'
      })

      // Mock responses for multiple services
      global.fetch = jest.fn()
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ status: 'running' })
        })
        .mockResolvedValueOnce({
          ok: false,
          status: 503,
          json: () => Promise.resolve({})
        })
        .mockRejectedValueOnce(new Error('Connection refused'))

      const { GET } = await import('../../src/app/api/gateway/route')
      const response = await GET(req as any)
      const data = await response.json()

      expect(response.status).toBe(200)
      expect(data.summary.online).toBeGreaterThan(0)
      expect(data.summary.offline).toBeGreaterThan(0)
      expect(data.summary.error).toBeGreaterThan(0)
    })

    it('should handle partial service availability', async () => {
      const { req, res } = createMocks<NextApiRequest, NextApiResponse>({
        method: 'POST',
        url: '/api/gateway?action=search&service=prowlarr',
        body: { query: 'test' }
      })

      // Mock Prowlarr being unavailable but graceful fallback
      global.fetch = jest.fn().mockRejectedValue(new Error('Service unavailable'))

      const { POST } = await import('../../src/app/api/gateway/route')
      const response = await POST(req as any)
      const data = await response.json()

      expect(response.status).toBe(200)
      expect(data.results).toBeDefined()
      // Should return mock results even when service is down
    })
  })

  describe('Performance', () => {
    it('should handle concurrent requests efficiently', async () => {
      const requests = []
      
      for (let i = 0; i < 10; i++) {
        const { req } = createMocks<NextApiRequest, NextApiResponse>({
          method: 'GET',
          url: '/api/gateway?action=status'
        })
        
        requests.push(req)
      }

      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ status: 'running' })
      })

      const { GET } = await import('../../src/app/api/gateway/route')
      
      const start = Date.now()
      const responses = await Promise.all(
        requests.map(req => GET(req as any))
      )
      const duration = Date.now() - start

      expect(responses).toHaveLength(10)
      expect(duration).toBeLessThan(5000) // Should complete within 5 seconds
      responses.forEach(response => {
        expect(response.status).toBe(200)
      })
    })

    it('should respect timeout configurations', async () => {
      const { req, res } = createMocks<NextApiRequest, NextApiResponse>({
        method: 'GET',
        url: '/api/gateway?action=health&service=jellyfin'
      })

      // Mock a slow response
      global.fetch = jest.fn().mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ status: 'running' })
        }), 10000)) // 10 second delay
      )

      const { GET } = await import('../../src/app/api/gateway/route')
      
      const start = Date.now()
      const response = await GET(req as any)
      const duration = Date.now() - start
      const data = await response.json()

      expect(duration).toBeLessThan(6000) // Should timeout before 6 seconds
      expect(data.status).toBe('offline')
    })
  })
})