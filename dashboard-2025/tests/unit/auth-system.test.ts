/**
 * Unit Tests for AuthSystem
 * Tests user registration, login, JWT tokens, OAuth2, and Redis sessions
 */

import { AuthSystem } from '../../src/lib/auth/AuthSystem'
import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken'

// Mock dependencies
jest.mock('ioredis')
jest.mock('bcryptjs')
jest.mock('jsonwebtoken')

// Mock fetch for OAuth2 testing
global.fetch = jest.fn()

describe('AuthSystem', () => {
  let authSystem: AuthSystem
  let mockRedis: any

  beforeEach(() => {
    mockRedis = {
      get: jest.fn(),
      setex: jest.fn(),
      del: jest.fn()
    }

    const oauth2Config = {
      google: {
        clientId: 'test-google-client',
        clientSecret: 'test-google-secret',
        redirectUri: 'http://localhost:3000/auth/google/callback'
      },
      github: {
        clientId: 'test-github-client',
        clientSecret: 'test-github-secret',
        redirectUri: 'http://localhost:3000/auth/github/callback'
      }
    }

    authSystem = new AuthSystem('test-jwt-secret', oauth2Config)
    
    // Mock Redis instance
    ;(authSystem as any).redis = mockRedis

    jest.clearAllMocks()
  })

  describe('register', () => {
    it('should successfully register a new user', async () => {
      const registerRequest = {
        email: 'test@example.com',
        username: 'testuser',
        password: 'securepassword123',
        confirmPassword: 'securepassword123'
      }

      ;(bcrypt.hash as jest.Mock).mockResolvedValue('hashedpassword')
      ;(jwt.sign as jest.Mock).mockReturnValue('access-token')

      const result = await authSystem.register(registerRequest)

      expect(result.user.email).toBe('test@example.com')
      expect(result.user.username).toBe('testuser')
      expect(result.user.passwordHash).toBeUndefined() // Should be sanitized
      expect(result.tokens.accessToken).toBe('access-token')
      expect(bcrypt.hash).toHaveBeenCalledWith('securepassword123', 12)
    })

    it('should reject registration with mismatched passwords', async () => {
      const registerRequest = {
        email: 'test@example.com',
        username: 'testuser',
        password: 'password123',
        confirmPassword: 'different123'
      }

      await expect(authSystem.register(registerRequest)).rejects.toThrow('Passwords do not match')
    })

    it('should reject registration with weak password', async () => {
      const registerRequest = {
        email: 'test@example.com',
        username: 'testuser',
        password: '123',
        confirmPassword: '123'
      }

      await expect(authSystem.register(registerRequest)).rejects.toThrow('Password must be at least 8 characters long')
    })

    it('should reject registration for existing user', async () => {
      const registerRequest = {
        email: 'test@example.com',
        username: 'testuser',
        password: 'securepassword123',
        confirmPassword: 'securepassword123'
      }

      ;(bcrypt.hash as jest.Mock).mockResolvedValue('hashedpassword')
      ;(jwt.sign as jest.Mock).mockReturnValue('access-token')

      // Register first user
      await authSystem.register(registerRequest)

      // Try to register again
      await expect(authSystem.register(registerRequest)).rejects.toThrow('User already exists')
    })
  })

  describe('login', () => {
    beforeEach(async () => {
      // Create a test user
      ;(bcrypt.hash as jest.Mock).mockResolvedValue('hashedpassword')
      ;(jwt.sign as jest.Mock).mockReturnValue('access-token')
      
      await authSystem.register({
        email: 'test@example.com',
        username: 'testuser',
        password: 'securepassword123',
        confirmPassword: 'securepassword123'
      })
    })

    it('should successfully login with valid credentials', async () => {
      const loginRequest = {
        email: 'test@example.com',
        password: 'securepassword123'
      }

      ;(bcrypt.compare as jest.Mock).mockResolvedValue(true)
      ;(jwt.sign as jest.Mock).mockReturnValue('new-access-token')

      const result = await authSystem.login(loginRequest)

      expect(result.user.email).toBe('test@example.com')
      expect(result.tokens.accessToken).toBe('new-access-token')
      expect(bcrypt.compare).toHaveBeenCalledWith('securepassword123', 'hashedpassword')
    })

    it('should reject login with invalid email', async () => {
      const loginRequest = {
        email: 'nonexistent@example.com',
        password: 'securepassword123'
      }

      await expect(authSystem.login(loginRequest)).rejects.toThrow('Invalid credentials')
    })

    it('should reject login with invalid password', async () => {
      const loginRequest = {
        email: 'test@example.com',
        password: 'wrongpassword'
      }

      ;(bcrypt.compare as jest.Mock).mockResolvedValue(false)

      await expect(authSystem.login(loginRequest)).rejects.toThrow('Invalid credentials')
    })
  })

  describe('verifyToken', () => {
    it('should verify valid JWT token', async () => {
      const mockDecodedToken = {
        userId: 'user123',
        email: 'test@example.com',
        type: 'access'
      }

      ;(jwt.verify as jest.Mock).mockReturnValue(mockDecodedToken)
      mockRedis.get.mockResolvedValue(null) // Not blacklisted

      // Create a test user
      ;(bcrypt.hash as jest.Mock).mockResolvedValue('hashedpassword')
      ;(jwt.sign as jest.Mock).mockReturnValue('access-token')
      
      const registerResult = await authSystem.register({
        email: 'test@example.com',
        username: 'testuser',
        password: 'securepassword123',
        confirmPassword: 'securepassword123'
      })

      // Update user ID to match mock
      ;(authSystem as any).users.set('user123', {
        ...(authSystem as any).users.get(registerResult.user.id),
        id: 'user123'
      })
      ;(authSystem as any).users.delete(registerResult.user.id)

      const result = await authSystem.verifyToken('test-token')

      expect(result).toBeTruthy()
      expect(result?.email).toBe('test@example.com')
      expect(jwt.verify).toHaveBeenCalledWith('test-token', 'test-jwt-secret')
    })

    it('should reject blacklisted token', async () => {
      const mockDecodedToken = {
        userId: 'user123',
        type: 'access'
      }

      ;(jwt.verify as jest.Mock).mockReturnValue(mockDecodedToken)
      mockRedis.get.mockResolvedValue('true') // Blacklisted

      const result = await authSystem.verifyToken('blacklisted-token')

      expect(result).toBeNull()
    })

    it('should reject invalid token', async () => {
      ;(jwt.verify as jest.Mock).mockImplementation(() => {
        throw new Error('Invalid token')
      })

      const result = await authSystem.verifyToken('invalid-token')

      expect(result).toBeNull()
    })
  })

  describe('Google OAuth2', () => {
    it('should authenticate with Google successfully', async () => {
      const mockTokenResponse = {
        ok: true,
        json: () => Promise.resolve({ access_token: 'google-access-token' })
      }

      const mockUserResponse = {
        ok: true,
        json: () => Promise.resolve({
          id: 'google123',
          email: 'user@gmail.com',
          name: 'Test User',
          picture: 'https://example.com/avatar.jpg'
        })
      }

      ;(fetch as jest.Mock)
        .mockResolvedValueOnce(mockTokenResponse)
        .mockResolvedValueOnce(mockUserResponse)
      
      ;(jwt.sign as jest.Mock).mockReturnValue('jwt-token')

      const result = await authSystem.authenticateWithGoogle('auth-code')

      expect(result.user.email).toBe('user@gmail.com')
      expect(result.user.provider).toBe('google')
      expect(result.user.providerId).toBe('google123')
      expect(result.tokens.accessToken).toBe('jwt-token')
    })

    it('should handle Google OAuth2 errors', async () => {
      const mockErrorResponse = {
        ok: false,
        status: 400
      }

      ;(fetch as jest.Mock).mockResolvedValueOnce(mockErrorResponse)

      await expect(authSystem.authenticateWithGoogle('invalid-code')).rejects.toThrow('Google authentication failed')
    })
  })

  describe('GitHub OAuth2', () => {
    it('should authenticate with GitHub successfully', async () => {
      const mockTokenResponse = {
        ok: true,
        json: () => Promise.resolve({ access_token: 'github-access-token' })
      }

      const mockUserResponse = {
        ok: true,
        json: () => Promise.resolve({
          id: 12345,
          login: 'testuser',
          email: 'user@github.com',
          avatar_url: 'https://github.com/avatar.jpg',
          html_url: 'https://github.com/testuser'
        })
      }

      ;(fetch as jest.Mock)
        .mockResolvedValueOnce(mockTokenResponse)
        .mockResolvedValueOnce(mockUserResponse)
      
      ;(jwt.sign as jest.Mock).mockReturnValue('jwt-token')

      const result = await authSystem.authenticateWithGitHub('auth-code')

      expect(result.user.email).toBe('user@github.com')
      expect(result.user.provider).toBe('github')
      expect(result.user.username).toBe('testuser')
      expect(result.tokens.accessToken).toBe('jwt-token')
    })
  })

  describe('Session Management', () => {
    it('should create and retrieve session', async () => {
      const sessionData = { ip: '127.0.0.1', userAgent: 'test' }
      mockRedis.setex.mockResolvedValue('OK')
      mockRedis.get.mockResolvedValue(JSON.stringify({
        sessionId: 'session123',
        userId: 'user123',
        ...sessionData
      }))

      const sessionId = await authSystem.createSession('user123', sessionData)
      expect(sessionId).toContain('session_')

      const retrievedSession = await authSystem.getSession('user123')
      expect(retrievedSession.userId).toBe('user123')
      expect(retrievedSession.ip).toBe('127.0.0.1')
    })

    it('should delete session', async () => {
      mockRedis.del.mockResolvedValue(1)

      await authSystem.deleteSession('user123')

      expect(mockRedis.del).toHaveBeenCalledWith('session:user123')
    })
  })

  describe('Token Refresh', () => {
    it('should refresh valid refresh token', async () => {
      const mockDecodedToken = {
        userId: 'user123',
        type: 'refresh'
      }

      ;(jwt.verify as jest.Mock).mockReturnValue(mockDecodedToken)
      ;(jwt.sign as jest.Mock).mockReturnValue('new-access-token')
      mockRedis.get.mockResolvedValue(null) // Not blacklisted

      // Create a test user
      ;(bcrypt.hash as jest.Mock).mockResolvedValue('hashedpassword')
      
      const registerResult = await authSystem.register({
        email: 'test@example.com',
        username: 'testuser',
        password: 'securepassword123',
        confirmPassword: 'securepassword123'
      })

      // Update user ID to match mock
      ;(authSystem as any).users.set('user123', {
        ...(authSystem as any).users.get(registerResult.user.id),
        id: 'user123'
      })

      const result = await authSystem.refreshToken('refresh-token')

      expect(result.accessToken).toBe('new-access-token')
    })

    it('should reject invalid refresh token', async () => {
      ;(jwt.verify as jest.Mock).mockImplementation(() => {
        throw new Error('Invalid token')
      })

      await expect(authSystem.refreshToken('invalid-refresh-token')).rejects.toThrow('Invalid refresh token')
    })
  })
})