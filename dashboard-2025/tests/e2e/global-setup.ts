/**
 * Playwright Global Setup
 * Initializes test environment and services
 */

import { chromium, FullConfig } from '@playwright/test'
import path from 'path'
import fs from 'fs'

async function globalSetup(config: FullConfig) {
  console.log('🚀 Starting Playwright global setup...')
  
  // Ensure output directories exist
  const authDir = path.join(__dirname, '../../test-results/auth')
  const screenshotsDir = path.join(__dirname, '../../test-results/screenshots')
  const videosDir = path.join(__dirname, '../../test-results/videos')
  
  if (!fs.existsSync(authDir)) {
    fs.mkdirSync(authDir, { recursive: true })
  }
  
  if (!fs.existsSync(screenshotsDir)) {
    fs.mkdirSync(screenshotsDir, { recursive: true })
  }
  
  if (!fs.existsSync(videosDir)) {
    fs.mkdirSync(videosDir, { recursive: true })
  }

  // Set up authentication
  const browser = await chromium.launch()
  const page = await browser.newPage()
  
  try {
    console.log('🔐 Setting up authentication...')
    
    // Navigate to app
    await page.goto(config.use?.baseURL || 'http://localhost:3000')
    
    // Check if we need to register/login
    const isLoginPage = await page.locator('[data-testid="login-form"]').isVisible().catch(() => false)
    
    if (isLoginPage) {
      // Try to login first
      await page.fill('[data-testid="email-input"]', 'test@example.com')
      await page.fill('[data-testid="password-input"]', 'securepassword123')
      await page.click('[data-testid="submit-login"]')
      
      // If login fails, register
      const isErrorVisible = await page.locator('[data-testid="error-message"]').isVisible().catch(() => false)
      
      if (isErrorVisible) {
        console.log('📝 Registering test user...')
        await page.click('[data-testid="register-button"]')
        await page.fill('[data-testid="email-input"]', 'test@example.com')
        await page.fill('[data-testid="username-input"]', 'testuser')
        await page.fill('[data-testid="password-input"]', 'securepassword123')
        await page.fill('[data-testid="confirm-password-input"]', 'securepassword123')
        await page.click('[data-testid="submit-registration"]')
      }
    }
    
    // Wait for successful authentication
    await page.waitForSelector('[data-testid="dashboard-main"]', { timeout: 10000 })
    console.log('✅ Authentication successful')
    
    // Save authenticated state
    await page.context().storageState({ path: path.join(authDir, 'user.json') })
    console.log('💾 Authentication state saved')
    
    // Set up test data
    console.log('📊 Setting up test data...')
    
    // Add some mock content to the recommendation engine
    await page.evaluate(() => {
      // Mock some viewing patterns
      const mockPatterns = [
        {
          contentId: 'movie123',
          contentType: 'movie',
          watchTime: 7200,
          totalDuration: 7800,
          completionRate: 0.92,
          genre: ['Action', 'Thriller'],
          rating: 8,
          device: 'smart-tv',
          timeOfDay: 'evening'
        },
        {
          contentId: 'show456',
          contentType: 'tv',
          watchTime: 2700,
          totalDuration: 3000,
          completionRate: 0.9,
          genre: ['Drama', 'Crime'],
          rating: 9,
          device: 'laptop',
          timeOfDay: 'night'
        }
      ]
      
      // Store in localStorage for tests
      localStorage.setItem('mockViewingPatterns', JSON.stringify(mockPatterns))
      
      // Mock service statuses
      const mockServices = [
        { name: 'jellyfin', status: 'online', responseTime: 150 },
        { name: 'plex', status: 'online', responseTime: 200 },
        { name: 'sonarr', status: 'online', responseTime: 100 },
        { name: 'radarr', status: 'online', responseTime: 120 },
        { name: 'qbittorrent', status: 'online', responseTime: 80 }
      ]
      
      localStorage.setItem('mockServiceStatuses', JSON.stringify(mockServices))
      
      // Mock downloads
      const mockDownloads = [
        {
          id: 'download123',
          title: 'Test Movie (2023)',
          progress: 65,
          status: 'downloading',
          speed: '2.5 MB/s',
          eta: '15m',
          size: '4.2 GB'
        },
        {
          id: 'download456',
          title: 'Test Series S01E01',
          progress: 100,
          status: 'completed',
          speed: '0 B/s',
          eta: '0m',
          size: '1.8 GB'
        }
      ]
      
      localStorage.setItem('mockDownloads', JSON.stringify(mockDownloads))
    })
    
    console.log('✅ Test data setup complete')
    
  } catch (error) {
    console.error('❌ Global setup failed:', error)
    throw error
  } finally {
    await browser.close()
  }
  
  console.log('🎉 Playwright global setup complete!')
}

export default globalSetup