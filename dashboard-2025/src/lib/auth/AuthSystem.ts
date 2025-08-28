/**
 * Complete OAuth2/JWT Authentication System
 * Supports user registration, JWT tokens, OAuth2 providers, and Redis sessions
 */

import jwt from 'jsonwebtoken'
import bcrypt from 'bcryptjs'
import Redis from 'ioredis'

export interface User {
  id: string
  email: string
  username: string
  passwordHash?: string
  provider?: 'local' | 'google' | 'github'
  providerId?: string
  roles: string[]
  permissions: string[]
  createdAt: Date
  lastLogin?: Date
  isActive: boolean
  metadata?: Record<string, any>
}

export interface AuthToken {
  accessToken: string
  refreshToken: string
  expiresIn: number
  tokenType: 'Bearer'
}

export interface LoginRequest {
  email: string
  password: string
  rememberMe?: boolean
}

export interface RegisterRequest {
  email: string
  username: string
  password: string
  confirmPassword: string
}

export interface OAuth2Config {
  google: {
    clientId: string
    clientSecret: string
    redirectUri: string
  }
  github: {
    clientId: string
    clientSecret: string
    redirectUri: string
  }
}

export class AuthSystem {
  private redis: Redis
  private users = new Map<string, User>() // In-memory store for demo
  
  constructor(
    private jwtSecret: string,
    private oauth2Config: OAuth2Config,
    redisUrl?: string
  ) {
    this.redis = new Redis(redisUrl || 'redis://localhost:6379')
  }

  /**
   * Register a new user
   */
  async register(request: RegisterRequest): Promise<{ user: User; tokens: AuthToken }> {
    // Validate request
    if (request.password !== request.confirmPassword) {
      throw new Error('Passwords do not match')
    }

    if (request.password.length < 8) {
      throw new Error('Password must be at least 8 characters long')
    }

    // Check if user exists
    const existingUser = Array.from(this.users.values()).find(
      u => u.email === request.email || u.username === request.username
    )

    if (existingUser) {
      throw new Error('User already exists')
    }

    // Hash password
    const passwordHash = await bcrypt.hash(request.password, 12)

    // Create user
    const user: User = {
      id: `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      email: request.email,
      username: request.username,
      passwordHash,
      provider: 'local',
      roles: ['user'],
      permissions: ['media:read', 'downloads:read'],
      createdAt: new Date(),
      isActive: true
    }

    this.users.set(user.id, user)

    // Generate tokens
    const tokens = await this.generateTokens(user)

    return { user: this.sanitizeUser(user), tokens }
  }

  /**
   * Login with email/password
   */
  async login(request: LoginRequest): Promise<{ user: User; tokens: AuthToken }> {
    // Find user
    const user = Array.from(this.users.values()).find(u => u.email === request.email)
    
    if (!user || !user.passwordHash) {
      throw new Error('Invalid credentials')
    }

    if (!user.isActive) {
      throw new Error('Account is disabled')
    }

    // Verify password
    const isValidPassword = await bcrypt.compare(request.password, user.passwordHash)
    
    if (!isValidPassword) {
      throw new Error('Invalid credentials')
    }

    // Update last login
    user.lastLogin = new Date()
    this.users.set(user.id, user)

    // Generate tokens
    const tokens = await this.generateTokens(user, request.rememberMe)

    return { user: this.sanitizeUser(user), tokens }
  }

  /**
   * Refresh access token
   */
  async refreshToken(refreshToken: string): Promise<AuthToken> {
    try {
      const decoded = jwt.verify(refreshToken, this.jwtSecret) as any
      
      if (decoded.type !== 'refresh') {
        throw new Error('Invalid token type')
      }

      const user = this.users.get(decoded.userId)
      
      if (!user || !user.isActive) {
        throw new Error('User not found or inactive')
      }

      // Check if refresh token is blacklisted
      const isBlacklisted = await this.redis.get(`blacklist:${refreshToken}`)
      if (isBlacklisted) {
        throw new Error('Token has been revoked')
      }

      return await this.generateTokens(user)
    } catch (error) {
      throw new Error('Invalid refresh token')
    }
  }

  /**
   * Logout and invalidate tokens
   */
  async logout(accessToken: string, refreshToken: string): Promise<void> {
    try {
      const decoded = jwt.verify(accessToken, this.jwtSecret) as any
      
      // Blacklist both tokens
      await Promise.all([
        this.redis.setex(`blacklist:${accessToken}`, decoded.exp - Math.floor(Date.now() / 1000), 'true'),
        this.redis.setex(`blacklist:${refreshToken}`, decoded.exp - Math.floor(Date.now() / 1000), 'true')
      ])

      // Remove user session
      await this.redis.del(`session:${decoded.userId}`)
    } catch (error) {
      // Token might be expired, but still remove from blacklist
      console.warn('Error during logout:', error)
    }
  }

  /**
   * Verify and decode JWT token
   */
  async verifyToken(token: string): Promise<User | null> {
    try {
      const decoded = jwt.verify(token, this.jwtSecret) as any
      
      if (decoded.type !== 'access') {
        return null
      }

      // Check if token is blacklisted
      const isBlacklisted = await this.redis.get(`blacklist:${token}`)
      if (isBlacklisted) {
        return null
      }

      const user = this.users.get(decoded.userId)
      
      if (!user || !user.isActive) {
        return null
      }

      return this.sanitizeUser(user)
    } catch (error) {
      return null
    }
  }

  /**
   * Google OAuth2 authentication
   */
  async authenticateWithGoogle(authCode: string): Promise<{ user: User; tokens: AuthToken }> {
    try {
      // Exchange auth code for tokens
      const tokenResponse = await fetch('https://oauth2.googleapis.com/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          client_id: this.oauth2Config.google.clientId,
          client_secret: this.oauth2Config.google.clientSecret,
          code: authCode,
          grant_type: 'authorization_code',
          redirect_uri: this.oauth2Config.google.redirectUri
        })
      })

      if (!tokenResponse.ok) {
        throw new Error('Failed to exchange auth code')
      }

      const tokens = await tokenResponse.json()

      // Get user info
      const userResponse = await fetch('https://www.googleapis.com/oauth2/v2/userinfo', {
        headers: { Authorization: `Bearer ${tokens.access_token}` }
      })

      if (!userResponse.ok) {
        throw new Error('Failed to get user info')
      }

      const googleUser = await userResponse.json()

      // Find or create user
      let user = Array.from(this.users.values()).find(
        u => u.email === googleUser.email || (u.provider === 'google' && u.providerId === googleUser.id)
      )

      if (!user) {
        user = {
          id: `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          email: googleUser.email,
          username: googleUser.name || googleUser.email.split('@')[0],
          provider: 'google',
          providerId: googleUser.id,
          roles: ['user'],
          permissions: ['media:read', 'downloads:read'],
          createdAt: new Date(),
          isActive: true,
          metadata: {
            picture: googleUser.picture,
            locale: googleUser.locale
          }
        }
        this.users.set(user.id, user)
      }

      // Update last login
      user.lastLogin = new Date()
      this.users.set(user.id, user)

      // Generate our tokens
      const authTokens = await this.generateTokens(user)

      return { user: this.sanitizeUser(user), tokens: authTokens }
    } catch (error) {
      throw new Error(`Google authentication failed: ${error}`)
    }
  }

  /**
   * GitHub OAuth2 authentication
   */
  async authenticateWithGitHub(authCode: string): Promise<{ user: User; tokens: AuthToken }> {
    try {
      // Exchange auth code for tokens
      const tokenResponse = await fetch('https://github.com/login/oauth/access_token', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          client_id: this.oauth2Config.github.clientId,
          client_secret: this.oauth2Config.github.clientSecret,
          code: authCode
        })
      })

      if (!tokenResponse.ok) {
        throw new Error('Failed to exchange auth code')
      }

      const tokens = await tokenResponse.json()

      // Get user info
      const userResponse = await fetch('https://api.github.com/user', {
        headers: { 
          Authorization: `token ${tokens.access_token}`,
          'User-Agent': 'Ultimate-Media-Server-2025'
        }
      })

      if (!userResponse.ok) {
        throw new Error('Failed to get user info')
      }

      const githubUser = await userResponse.json()

      // Get user email (might be private)
      const emailResponse = await fetch('https://api.github.com/user/emails', {
        headers: { 
          Authorization: `token ${tokens.access_token}`,
          'User-Agent': 'Ultimate-Media-Server-2025'
        }
      })

      let email = githubUser.email
      if (!email && emailResponse.ok) {
        const emails = await emailResponse.json()
        const primaryEmail = emails.find((e: any) => e.primary)
        email = primaryEmail?.email || emails[0]?.email
      }

      if (!email) {
        throw new Error('Email address is required')
      }

      // Find or create user
      let user = Array.from(this.users.values()).find(
        u => u.email === email || (u.provider === 'github' && u.providerId === githubUser.id.toString())
      )

      if (!user) {
        user = {
          id: `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          email,
          username: githubUser.login,
          provider: 'github',
          providerId: githubUser.id.toString(),
          roles: ['user'],
          permissions: ['media:read', 'downloads:read'],
          createdAt: new Date(),
          isActive: true,
          metadata: {
            avatar: githubUser.avatar_url,
            githubProfile: githubUser.html_url,
            bio: githubUser.bio
          }
        }
        this.users.set(user.id, user)
      }

      // Update last login
      user.lastLogin = new Date()
      this.users.set(user.id, user)

      // Generate our tokens
      const authTokens = await this.generateTokens(user)

      return { user: this.sanitizeUser(user), tokens: authTokens }
    } catch (error) {
      throw new Error(`GitHub authentication failed: ${error}`)
    }
  }

  /**
   * Get user by ID
   */
  async getUserById(userId: string): Promise<User | null> {
    const user = this.users.get(userId)
    return user ? this.sanitizeUser(user) : null
  }

  /**
   * Update user permissions
   */
  async updateUserPermissions(userId: string, permissions: string[]): Promise<User | null> {
    const user = this.users.get(userId)
    if (!user) return null

    user.permissions = permissions
    this.users.set(userId, user)

    return this.sanitizeUser(user)
  }

  /**
   * Session management
   */
  async createSession(userId: string, data: any): Promise<string> {
    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    await this.redis.setex(`session:${userId}`, 86400, JSON.stringify({ // 24 hours
      sessionId,
      userId,
      createdAt: new Date().toISOString(),
      ...data
    }))
    return sessionId
  }

  async getSession(userId: string): Promise<any> {
    const sessionData = await this.redis.get(`session:${userId}`)
    return sessionData ? JSON.parse(sessionData) : null
  }

  async deleteSession(userId: string): Promise<void> {
    await this.redis.del(`session:${userId}`)
  }

  private async generateTokens(user: User, rememberMe = false): Promise<AuthToken> {
    const accessTokenExpiry = '15m'
    const refreshTokenExpiry = rememberMe ? '30d' : '7d'

    const accessToken = jwt.sign(
      {
        userId: user.id,
        email: user.email,
        roles: user.roles,
        permissions: user.permissions,
        type: 'access'
      },
      this.jwtSecret,
      { expiresIn: accessTokenExpiry }
    )

    const refreshToken = jwt.sign(
      {
        userId: user.id,
        type: 'refresh'
      },
      this.jwtSecret,
      { expiresIn: refreshTokenExpiry }
    )

    return {
      accessToken,
      refreshToken,
      expiresIn: 15 * 60, // 15 minutes in seconds
      tokenType: 'Bearer'
    }
  }

  private sanitizeUser(user: User): User {
    const { passwordHash, ...sanitized } = user
    return sanitized as User
  }
}