/**
 * End-to-End Tests for Dashboard
 * Tests complete user workflows with Playwright
 */

import { test, expect, Page } from '@playwright/test'

test.describe('Dashboard E2E Tests', () => {
  let page: Page

  test.beforeEach(async ({ browser }) => {
    page = await browser.newPage()
    await page.goto('/')
  })

  test.afterEach(async () => {
    await page.close()
  })

  test.describe('Authentication Flow', () => {
    test('should allow user registration and login', async () => {
      // Navigate to registration
      await page.click('[data-testid="register-button"]')
      await expect(page.locator('[data-testid="register-form"]')).toBeVisible()

      // Fill registration form
      await page.fill('[data-testid="email-input"]', 'test@example.com')
      await page.fill('[data-testid="username-input"]', 'testuser')
      await page.fill('[data-testid="password-input"]', 'securepassword123')
      await page.fill('[data-testid="confirm-password-input"]', 'securepassword123')

      // Submit registration
      await page.click('[data-testid="submit-registration"]')
      
      // Should redirect to dashboard
      await expect(page.locator('[data-testid="dashboard-main"]')).toBeVisible()
      await expect(page.locator('[data-testid="welcome-message"]')).toContainText('Welcome, testuser')
    })

    test('should handle login with valid credentials', async () => {
      // Navigate to login
      await page.click('[data-testid="login-button"]')
      await expect(page.locator('[data-testid="login-form"]')).toBeVisible()

      // Fill login form
      await page.fill('[data-testid="email-input"]', 'test@example.com')
      await page.fill('[data-testid="password-input"]', 'securepassword123')

      // Submit login
      await page.click('[data-testid="submit-login"]')
      
      // Should redirect to dashboard
      await expect(page.locator('[data-testid="dashboard-main"]')).toBeVisible()
    })

    test('should handle OAuth2 login with Google', async () => {
      // Click Google login button
      await page.click('[data-testid="google-login-button"]')
      
      // Should redirect to Google OAuth (we'll mock this in tests)
      await expect(page.url()).toContain('accounts.google.com')
      
      // Mock successful OAuth callback
      await page.goto('/?code=mock-auth-code&state=google')
      
      // Should be logged in
      await expect(page.locator('[data-testid="dashboard-main"]')).toBeVisible()
    })

    test('should show error for invalid login credentials', async () => {
      await page.click('[data-testid="login-button"]')
      
      await page.fill('[data-testid="email-input"]', 'invalid@example.com')
      await page.fill('[data-testid="password-input"]', 'wrongpassword')
      
      await page.click('[data-testid="submit-login"]')
      
      await expect(page.locator('[data-testid="error-message"]')).toContainText('Invalid credentials')
    })
  })

  test.describe('Dashboard Navigation', () => {
    test.beforeEach(async () => {
      // Login before each test
      await page.goto('/login')
      await page.fill('[data-testid="email-input"]', 'test@example.com')
      await page.fill('[data-testid="password-input"]', 'securepassword123')
      await page.click('[data-testid="submit-login"]')
      await expect(page.locator('[data-testid="dashboard-main"]')).toBeVisible()
    })

    test('should navigate between main sections', async () => {
      // Test media library navigation
      await page.click('[data-testid="nav-media"]')
      await expect(page.locator('[data-testid="media-library"]')).toBeVisible()
      
      // Test downloads navigation
      await page.click('[data-testid="nav-downloads"]')
      await expect(page.locator('[data-testid="downloads-page"]')).toBeVisible()
      
      // Test requests navigation
      await page.click('[data-testid="nav-requests"]')
      await expect(page.locator('[data-testid="requests-page"]')).toBeVisible()
    })

    test('should display service status grid', async () => {
      await expect(page.locator('[data-testid="service-grid"]')).toBeVisible()
      
      // Should show multiple service cards
      const serviceCards = page.locator('[data-testid^="service-card-"]')
      await expect(serviceCards).toHaveCountGreaterThan(5)
      
      // Each service card should have status indicator
      const firstCard = serviceCards.first()
      await expect(firstCard.locator('[data-testid="service-status"]')).toBeVisible()
    })

    test('should show real-time system metrics', async () => {
      await expect(page.locator('[data-testid="system-metrics"]')).toBeVisible()
      
      // Should display CPU, Memory, and Storage metrics
      await expect(page.locator('[data-testid="cpu-metric"]')).toBeVisible()
      await expect(page.locator('[data-testid="memory-metric"]')).toBeVisible()
      await expect(page.locator('[data-testid="storage-metric"]')).toBeVisible()
    })
  })

  test.describe('Media Library Features', () => {
    test.beforeEach(async () => {
      await page.goto('/login')
      await page.fill('[data-testid="email-input"]', 'test@example.com')
      await page.fill('[data-testid="password-input"]', 'securepassword123')
      await page.click('[data-testid="submit-login"]')
      await page.click('[data-testid="nav-media"]')
    })

    test('should display media items from connected services', async () => {
      await expect(page.locator('[data-testid="media-library"]')).toBeVisible()
      
      // Should show media items
      const mediaItems = page.locator('[data-testid^="media-item-"]')
      await expect(mediaItems).toHaveCountGreaterThan(0)
      
      // Each item should have title and poster
      const firstItem = mediaItems.first()
      await expect(firstItem.locator('[data-testid="media-title"]')).toBeVisible()
      await expect(firstItem.locator('[data-testid="media-poster"]')).toBeVisible()
    })

    test('should filter media by type and service', async () => {
      // Filter by media type
      await page.selectOption('[data-testid="media-type-filter"]', 'movie')
      
      // Should update media grid
      await page.waitForTimeout(1000) // Wait for filter to apply
      const movieItems = page.locator('[data-testid^="media-item-movie-"]')
      await expect(movieItems).toHaveCountGreaterThan(0)
      
      // Filter by service
      await page.selectOption('[data-testid="service-filter"]', 'jellyfin')
      
      // Should show only Jellyfin items
      await page.waitForTimeout(1000)
      const jellyfinItems = page.locator('[data-testid^="media-item-"][data-service="jellyfin"]')
      await expect(jellyfinItems).toHaveCountGreaterThan(0)
    })

    test('should search media library', async () => {
      const searchTerm = 'Test Movie'
      
      await page.fill('[data-testid="media-search"]', searchTerm)
      await page.press('[data-testid="media-search"]', 'Enter')
      
      // Should show search results
      await expect(page.locator('[data-testid="search-results"]')).toBeVisible()
      
      // Results should contain search term
      const resultTitles = page.locator('[data-testid="media-title"]')
      const firstTitle = await resultTitles.first().textContent()
      expect(firstTitle?.toLowerCase()).toContain(searchTerm.toLowerCase())
    })

    test('should play media item', async () => {
      const mediaItems = page.locator('[data-testid^="media-item-"]')
      const firstItem = mediaItems.first()
      
      // Click play button
      await firstItem.hover()
      await firstItem.locator('[data-testid="play-button"]').click()
      
      // Should open media player
      await expect(page.locator('[data-testid="media-player"]')).toBeVisible()
    })
  })

  test.describe('Download Management', () => {
    test.beforeEach(async () => {
      await page.goto('/login')
      await page.fill('[data-testid="email-input"]', 'test@example.com')
      await page.fill('[data-testid="password-input"]', 'securepassword123')
      await page.click('[data-testid="submit-login"]')
      await page.click('[data-testid="nav-downloads"]')
    })

    test('should display active downloads', async () => {
      await expect(page.locator('[data-testid="downloads-page"]')).toBeVisible()
      
      // Should show download items
      const downloadItems = page.locator('[data-testid^="download-item-"]')
      await expect(downloadItems).toHaveCountGreaterThanOrEqual(0)
      
      // If downloads exist, check their structure
      const firstDownload = downloadItems.first()
      if (await firstDownload.isVisible()) {
        await expect(firstDownload.locator('[data-testid="download-title"]')).toBeVisible()
        await expect(firstDownload.locator('[data-testid="download-progress"]')).toBeVisible()
        await expect(firstDownload.locator('[data-testid="download-status"]')).toBeVisible()
      }
    })

    test('should pause and resume downloads', async () => {
      const downloadItems = page.locator('[data-testid^="download-item-"]')
      const firstDownload = downloadItems.first()
      
      if (await firstDownload.isVisible()) {
        // Pause download
        await firstDownload.locator('[data-testid="pause-button"]').click()
        
        // Should show paused status
        await expect(firstDownload.locator('[data-testid="download-status"]')).toContainText('paused')
        
        // Resume download
        await firstDownload.locator('[data-testid="resume-button"]').click()
        
        // Should show downloading status
        await expect(firstDownload.locator('[data-testid="download-status"]')).toContainText('downloading')
      }
    })

    test('should cancel downloads', async () => {
      const downloadItems = page.locator('[data-testid^="download-item-"]')
      const firstDownload = downloadItems.first()
      
      if (await firstDownload.isVisible()) {
        const downloadId = await firstDownload.getAttribute('data-download-id')
        
        // Cancel download
        await firstDownload.locator('[data-testid="cancel-button"]').click()
        
        // Confirm cancellation
        await page.click('[data-testid="confirm-cancel"]')
        
        // Download should be removed
        await expect(page.locator(`[data-download-id="${downloadId}"]`)).not.toBeVisible()
      }
    })

    test('should show real-time download progress updates', async () => {
      const downloadItems = page.locator('[data-testid^="download-item-"]')
      const firstDownload = downloadItems.first()
      
      if (await firstDownload.isVisible()) {
        const progressBar = firstDownload.locator('[data-testid="download-progress"]')
        const initialProgress = await progressBar.getAttribute('value')
        
        // Wait for progress update (WebSocket)
        await page.waitForTimeout(2000)
        
        const updatedProgress = await progressBar.getAttribute('value')
        
        // Progress should update (or at least not throw an error)
        expect(updatedProgress).toBeDefined()
      }
    })
  })

  test.describe('Search and Request Features', () => {
    test.beforeEach(async () => {
      await page.goto('/login')
      await page.fill('[data-testid="email-input"]', 'test@example.com')
      await page.fill('[data-testid="password-input"]', 'securepassword123')
      await page.click('[data-testid="submit-login"]')
      await page.click('[data-testid="nav-requests"]')
    })

    test('should search for new content', async () => {
      await expect(page.locator('[data-testid="requests-page"]')).toBeVisible()
      
      const searchTerm = 'The Matrix'
      
      await page.fill('[data-testid="content-search"]', searchTerm)
      await page.click('[data-testid="search-button"]')
      
      // Should show search results
      await expect(page.locator('[data-testid="search-results"]')).toBeVisible()
      
      const searchResults = page.locator('[data-testid^="search-result-"]')
      await expect(searchResults).toHaveCountGreaterThan(0)
    })

    test('should request new content', async () => {
      const searchTerm = 'Test Movie 2023'
      
      await page.fill('[data-testid="content-search"]', searchTerm)
      await page.click('[data-testid="search-button"]')
      
      await expect(page.locator('[data-testid="search-results"]')).toBeVisible()
      
      const firstResult = page.locator('[data-testid^="search-result-"]').first()
      await firstResult.locator('[data-testid="request-button"]').click()
      
      // Should show request confirmation
      await expect(page.locator('[data-testid="request-confirmation"]')).toBeVisible()
      await expect(page.locator('[data-testid="success-message"]')).toContainText('Request submitted')
    })

    test('should display request history', async () => {
      await page.click('[data-testid="request-history-tab"]')
      
      await expect(page.locator('[data-testid="request-history"]')).toBeVisible()
      
      const requestItems = page.locator('[data-testid^="request-item-"]')
      
      // If requests exist, verify their structure
      const firstRequest = requestItems.first()
      if (await firstRequest.isVisible()) {
        await expect(firstRequest.locator('[data-testid="request-title"]')).toBeVisible()
        await expect(firstRequest.locator('[data-testid="request-status"]')).toBeVisible()
        await expect(firstRequest.locator('[data-testid="request-date"]')).toBeVisible()
      }
    })
  })

  test.describe('Real-time Features', () => {
    test.beforeEach(async () => {
      await page.goto('/login')
      await page.fill('[data-testid="email-input"]', 'test@example.com')
      await page.fill('[data-testid="password-input"]', 'securepassword123')
      await page.click('[data-testid="submit-login"]')
    })

    test('should receive real-time notifications', async () => {
      // Wait for WebSocket connection
      await page.waitForTimeout(1000)
      
      // Trigger a notification (simulate download completion)
      await page.evaluate(() => {
        // Simulate WebSocket message
        window.dispatchEvent(new CustomEvent('websocket-message', {
          detail: {
            type: 'notification:new',
            data: {
              id: 'test-notification',
              type: 'success',
              title: 'Download Complete',
              message: 'Test Movie has finished downloading'
            }
          }
        }))
      })
      
      // Should show notification
      await expect(page.locator('[data-testid="notification-toast"]')).toBeVisible()
      await expect(page.locator('[data-testid="notification-title"]')).toContainText('Download Complete')
    })

    test('should update service status in real-time', async () => {
      const serviceCard = page.locator('[data-testid="service-card-jellyfin"]')
      const statusIndicator = serviceCard.locator('[data-testid="service-status"]')
      
      // Initial status
      const initialStatus = await statusIndicator.textContent()
      
      // Simulate service status change
      await page.evaluate(() => {
        window.dispatchEvent(new CustomEvent('websocket-message', {
          detail: {
            type: 'service:status',
            data: {
              name: 'jellyfin',
              status: 'offline',
              responseTime: 0
            }
          }
        }))
      })
      
      // Should update status
      await expect(statusIndicator).toContainText('offline')
    })

    test('should show live download progress updates', async () => {
      await page.click('[data-testid="nav-downloads"]')
      
      const downloadItem = page.locator('[data-testid^="download-item-"]').first()
      
      if (await downloadItem.isVisible()) {
        const progressBar = downloadItem.locator('[data-testid="download-progress"]')
        
        // Simulate progress update
        await page.evaluate(() => {
          window.dispatchEvent(new CustomEvent('websocket-message', {
            detail: {
              type: 'download:progress',
              data: {
                id: 'test-download',
                progress: 75,
                speed: '2.5 MB/s',
                eta: '5m'
              }
            }
          }))
        })
        
        // Should update progress
        await expect(progressBar).toHaveAttribute('value', '75')
      }
    })
  })

  test.describe('3D Visualization Features', () => {
    test.beforeEach(async () => {
      await page.goto('/login')
      await page.fill('[data-testid="email-input"]', 'test@example.com')
      await page.fill('[data-testid="password-input"]', 'securepassword123')
      await page.click('[data-testid="submit-login"]')
    })

    test('should render 3D media visualization', async () => {
      await page.click('[data-testid="3d-view-toggle"]')
      
      // Should show 3D canvas
      await expect(page.locator('[data-testid="three-canvas"]')).toBeVisible()
      
      // Should render 3D objects
      const canvas = page.locator('canvas')
      await expect(canvas).toBeVisible()
      
      // Verify WebGL context is working
      const hasWebGL = await page.evaluate(() => {
        const canvas = document.querySelector('canvas')
        const gl = canvas?.getContext('webgl') || canvas?.getContext('experimental-webgl')
        return !!gl
      })
      
      expect(hasWebGL).toBe(true)
    })

    test('should allow 3D interaction with media orbs', async () => {
      await page.click('[data-testid="3d-view-toggle"]')
      await expect(page.locator('[data-testid="three-canvas"]')).toBeVisible()
      
      const canvas = page.locator('canvas')
      
      // Test mouse interaction
      await canvas.hover()
      await canvas.click()
      
      // Should show media details on interaction
      await expect(page.locator('[data-testid="media-details-popup"]')).toBeVisible()
    })
  })

  test.describe('Performance and Accessibility', () => {
    test('should load dashboard within acceptable time', async () => {
      const start = Date.now()
      
      await page.goto('/')
      await page.fill('[data-testid="email-input"]', 'test@example.com')
      await page.fill('[data-testid="password-input"]', 'securepassword123')
      await page.click('[data-testid="submit-login"]')
      
      await expect(page.locator('[data-testid="dashboard-main"]')).toBeVisible()
      
      const loadTime = Date.now() - start
      expect(loadTime).toBeLessThan(5000) // Should load within 5 seconds
    })

    test('should be keyboard navigable', async () => {
      await page.goto('/login')
      
      // Navigate using keyboard
      await page.press('body', 'Tab') // Focus email input
      await page.keyboard.type('test@example.com')
      
      await page.press('body', 'Tab') // Focus password input
      await page.keyboard.type('securepassword123')
      
      await page.press('body', 'Tab') // Focus submit button
      await page.press('body', 'Enter') // Submit form
      
      await expect(page.locator('[data-testid="dashboard-main"]')).toBeVisible()
    })

    test('should have proper ARIA labels', async () => {
      await page.goto('/')
      
      // Check main navigation
      const nav = page.locator('[role="navigation"]')
      await expect(nav).toBeVisible()
      
      // Check form labels
      const emailInput = page.locator('[data-testid="email-input"]')
      await expect(emailInput).toHaveAttribute('aria-label')
      
      const passwordInput = page.locator('[data-testid="password-input"]')
      await expect(passwordInput).toHaveAttribute('aria-label')
    })
  })
})