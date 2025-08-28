/**
 * WebSocket Real-Time Manager
 * Handles live updates for downloads, service status, notifications, and messaging
 */

import { Server as SocketIOServer } from 'socket.io'
import { createServer } from 'http'

export interface DownloadProgress {
  id: string
  title: string
  progress: number
  speed: string
  eta: string
  status: 'downloading' | 'paused' | 'completed' | 'error'
  service: string
  timestamp: Date
}

export interface ServiceStatus {
  name: string
  status: 'online' | 'offline' | 'degraded'
  responseTime: number
  lastUpdate: Date
  metadata?: Record<string, any>
}

export interface Notification {
  id: string
  type: 'info' | 'success' | 'warning' | 'error'
  title: string
  message: string
  timestamp: Date
  userId?: string
  actions?: Array<{
    label: string
    action: string
    data?: any
  }>
}

export interface ChatMessage {
  id: string
  userId: string
  username: string
  message: string
  timestamp: Date
  channel: string
  type: 'text' | 'system' | 'file' | 'media'
  metadata?: Record<string, any>
}

export class RealTimeManager {
  private io: SocketIOServer
  private downloads = new Map<string, DownloadProgress>()
  private serviceStatuses = new Map<string, ServiceStatus>()
  private notifications: Notification[] = []
  private chatMessages: ChatMessage[] = []
  private userSessions = new Map<string, Set<string>>() // userId -> socketIds

  constructor(server?: any) {
    this.io = new SocketIOServer(server || createServer(), {
      cors: {
        origin: process.env.NODE_ENV === 'development' ? ['http://localhost:3000'] : true,
        methods: ['GET', 'POST'],
        credentials: true
      },
      transports: ['websocket', 'polling']
    })

    this.setupEventHandlers()
  }

  private setupEventHandlers(): void {
    this.io.on('connection', (socket) => {
      console.log(`Socket connected: ${socket.id}`)

      // Authentication
      socket.on('authenticate', async (token: string) => {
        try {
          // Verify JWT token (integrate with AuthSystem)
          const user = await this.verifySocketToken(token)
          if (user) {
            socket.data.user = user
            this.addUserSession(user.id, socket.id)
            
            // Send initial data
            socket.emit('authenticated', { user })
            socket.emit('downloads:initial', Array.from(this.downloads.values()))
            socket.emit('services:initial', Array.from(this.serviceStatuses.values()))
            socket.emit('notifications:initial', this.getNotificationsForUser(user.id))
            
            socket.join(`user:${user.id}`)
          } else {
            socket.emit('auth:error', { message: 'Invalid token' })
          }
        } catch (error) {
          socket.emit('auth:error', { message: 'Authentication failed' })
        }
      })

      // Download management
      socket.on('downloads:subscribe', () => {
        socket.join('downloads')
      })

      socket.on('downloads:unsubscribe', () => {
        socket.leave('downloads')
      })

      socket.on('download:pause', (downloadId: string) => {
        this.pauseDownload(downloadId, socket.data.user?.id)
      })

      socket.on('download:resume', (downloadId: string) => {
        this.resumeDownload(downloadId, socket.data.user?.id)
      })

      socket.on('download:cancel', (downloadId: string) => {
        this.cancelDownload(downloadId, socket.data.user?.id)
      })

      // Service monitoring
      socket.on('services:subscribe', () => {
        socket.join('services')
      })

      socket.on('services:unsubscribe', () => {
        socket.leave('services')
      })

      socket.on('service:restart', (serviceName: string) => {
        this.restartService(serviceName, socket.data.user?.id)
      })

      // Notifications
      socket.on('notifications:subscribe', () => {
        if (socket.data.user) {
          socket.join(`notifications:${socket.data.user.id}`)
        }
      })

      socket.on('notification:dismiss', (notificationId: string) => {
        this.dismissNotification(notificationId, socket.data.user?.id)
      })

      socket.on('notification:action', (data: { notificationId: string; action: string; actionData?: any }) => {
        this.handleNotificationAction(data, socket.data.user?.id)
      })

      // Chat/Messaging
      socket.on('chat:join', (channel: string) => {
        socket.join(`chat:${channel}`)
        socket.emit('chat:history', this.getChatHistory(channel))
      })

      socket.on('chat:leave', (channel: string) => {
        socket.leave(`chat:${channel}`)
      })

      socket.on('chat:message', (data: { channel: string; message: string; type?: string }) => {
        this.handleChatMessage(data, socket.data.user)
      })

      socket.on('chat:typing', (data: { channel: string; isTyping: boolean }) => {
        socket.to(`chat:${data.channel}`).emit('chat:typing', {
          userId: socket.data.user?.id,
          username: socket.data.user?.username,
          isTyping: data.isTyping
        })
      })

      // System events
      socket.on('system:stats', () => {
        socket.emit('system:stats', this.getSystemStats())
      })

      // Disconnect handling
      socket.on('disconnect', () => {
        console.log(`Socket disconnected: ${socket.id}`)
        if (socket.data.user) {
          this.removeUserSession(socket.data.user.id, socket.id)
        }
      })
    })
  }

  // Download Progress Methods
  updateDownloadProgress(progress: DownloadProgress): void {
    this.downloads.set(progress.id, progress)
    this.io.to('downloads').emit('download:progress', progress)
  }

  addDownload(download: DownloadProgress): void {
    this.downloads.set(download.id, download)
    this.io.to('downloads').emit('download:added', download)
    
    // Send notification
    this.sendNotification({
      id: `download-${download.id}`,
      type: 'info',
      title: 'Download Started',
      message: `${download.title} has been added to the download queue`,
      timestamp: new Date()
    })
  }

  completeDownload(downloadId: string): void {
    const download = this.downloads.get(downloadId)
    if (download) {
      download.status = 'completed'
      download.progress = 100
      this.downloads.set(downloadId, download)
      
      this.io.to('downloads').emit('download:completed', download)
      
      // Send notification
      this.sendNotification({
        id: `download-complete-${downloadId}`,
        type: 'success',
        title: 'Download Complete',
        message: `${download.title} has finished downloading`,
        timestamp: new Date(),
        actions: [
          { label: 'View File', action: 'open-file', data: { downloadId } },
          { label: 'Play Now', action: 'play-media', data: { downloadId } }
        ]
      })
    }
  }

  private async pauseDownload(downloadId: string, userId?: string): Promise<void> {
    const download = this.downloads.get(downloadId)
    if (download && this.hasPermission(userId, 'downloads:control')) {
      download.status = 'paused'
      this.downloads.set(downloadId, download)
      this.io.to('downloads').emit('download:paused', download)
    }
  }

  private async resumeDownload(downloadId: string, userId?: string): Promise<void> {
    const download = this.downloads.get(downloadId)
    if (download && this.hasPermission(userId, 'downloads:control')) {
      download.status = 'downloading'
      this.downloads.set(downloadId, download)
      this.io.to('downloads').emit('download:resumed', download)
    }
  }

  private async cancelDownload(downloadId: string, userId?: string): Promise<void> {
    const download = this.downloads.get(downloadId)
    if (download && this.hasPermission(userId, 'downloads:control')) {
      this.downloads.delete(downloadId)
      this.io.to('downloads').emit('download:cancelled', { id: downloadId })
    }
  }

  // Service Status Methods
  updateServiceStatus(status: ServiceStatus): void {
    const previousStatus = this.serviceStatuses.get(status.name)
    this.serviceStatuses.set(status.name, status)
    
    this.io.to('services').emit('service:status', status)
    
    // Send notification on status change
    if (previousStatus && previousStatus.status !== status.status) {
      const notificationType = status.status === 'online' ? 'success' : 
                              status.status === 'degraded' ? 'warning' : 'error'
      
      this.sendNotification({
        id: `service-${status.name}-${Date.now()}`,
        type: notificationType,
        title: `Service ${status.status}`,
        message: `${status.name} is now ${status.status}`,
        timestamp: new Date()
      })
    }
  }

  private async restartService(serviceName: string, userId?: string): Promise<void> {
    if (this.hasPermission(userId, 'services:control')) {
      // Emit restart command
      this.io.to('services').emit('service:restarting', { name: serviceName })
      
      // Send notification
      this.sendNotification({
        id: `service-restart-${serviceName}-${Date.now()}`,
        type: 'info',
        title: 'Service Restart',
        message: `Restarting ${serviceName}...`,
        timestamp: new Date()
      })
    }
  }

  // Notification Methods
  sendNotification(notification: Notification): void {
    this.notifications.push(notification)
    
    // Keep only last 100 notifications
    if (this.notifications.length > 100) {
      this.notifications = this.notifications.slice(-100)
    }
    
    if (notification.userId) {
      this.io.to(`notifications:${notification.userId}`).emit('notification:new', notification)
    } else {
      this.io.emit('notification:new', notification)
    }
  }

  sendUserNotification(userId: string, notification: Omit<Notification, 'userId'>): void {
    this.sendNotification({ ...notification, userId })
  }

  private dismissNotification(notificationId: string, userId?: string): void {
    const index = this.notifications.findIndex(n => n.id === notificationId)
    if (index >= 0) {
      const notification = this.notifications[index]
      if (!notification.userId || notification.userId === userId) {
        this.notifications.splice(index, 1)
        this.io.emit('notification:dismissed', { id: notificationId })
      }
    }
  }

  private handleNotificationAction(data: { notificationId: string; action: string; actionData?: any }, userId?: string): void {
    const notification = this.notifications.find(n => n.id === data.notificationId)
    if (notification) {
      // Emit action event for handling by other parts of the system
      this.io.emit('notification:action', {
        ...data,
        userId,
        notification
      })
      
      // Dismiss notification after action
      this.dismissNotification(data.notificationId, userId)
    }
  }

  // Chat/Messaging Methods
  private handleChatMessage(data: { channel: string; message: string; type?: string }, user?: any): void {
    if (!user) return
    
    const message: ChatMessage = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      userId: user.id,
      username: user.username,
      message: data.message,
      timestamp: new Date(),
      channel: data.channel,
      type: (data.type as any) || 'text'
    }
    
    this.chatMessages.push(message)
    
    // Keep only last 1000 messages
    if (this.chatMessages.length > 1000) {
      this.chatMessages = this.chatMessages.slice(-1000)
    }
    
    this.io.to(`chat:${data.channel}`).emit('chat:message', message)
  }

  private getChatHistory(channel: string, limit = 50): ChatMessage[] {
    return this.chatMessages
      .filter(msg => msg.channel === channel)
      .slice(-limit)
  }

  // User Session Management
  private addUserSession(userId: string, socketId: string): void {
    if (!this.userSessions.has(userId)) {
      this.userSessions.set(userId, new Set())
    }
    this.userSessions.get(userId)!.add(socketId)
  }

  private removeUserSession(userId: string, socketId: string): void {
    const sessions = this.userSessions.get(userId)
    if (sessions) {
      sessions.delete(socketId)
      if (sessions.size === 0) {
        this.userSessions.delete(userId)
      }
    }
  }

  // Utility Methods
  private getNotificationsForUser(userId: string): Notification[] {
    return this.notifications.filter(n => !n.userId || n.userId === userId)
  }

  private getSystemStats(): any {
    return {
      connectedUsers: this.userSessions.size,
      activeDownloads: Array.from(this.downloads.values()).filter(d => d.status === 'downloading').length,
      totalDownloads: this.downloads.size,
      onlineServices: Array.from(this.serviceStatuses.values()).filter(s => s.status === 'online').length,
      totalServices: this.serviceStatuses.size,
      unreadNotifications: this.notifications.length,
      chatChannels: new Set(this.chatMessages.map(m => m.channel)).size
    }
  }

  private async verifySocketToken(token: string): Promise<any> {
    // This should integrate with your AuthSystem
    // For now, return a mock user
    try {
      // JWT verification logic here
      return {
        id: 'user123',
        username: 'demo_user',
        permissions: ['downloads:control', 'services:control']
      }
    } catch {
      return null
    }
  }

  private hasPermission(userId?: string, permission?: string): boolean {
    // Mock permission check - integrate with your auth system
    return true
  }

  // Public API Methods
  getServer(): SocketIOServer {
    return this.io
  }

  getConnectedUsers(): number {
    return this.userSessions.size
  }

  getActiveDownloads(): DownloadProgress[] {
    return Array.from(this.downloads.values()).filter(d => d.status === 'downloading')
  }

  getAllDownloads(): DownloadProgress[] {
    return Array.from(this.downloads.values())
  }

  getServiceStatuses(): ServiceStatus[] {
    return Array.from(this.serviceStatuses.values())
  }

  broadcastMessage(event: string, data: any): void {
    this.io.emit(event, data)
  }

  sendToUser(userId: string, event: string, data: any): void {
    this.io.to(`user:${userId}`).emit(event, data)
  }
}