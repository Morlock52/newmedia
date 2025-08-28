/**
 * Playwright Global Teardown
 * Cleans up test environment and resources
 */

import { FullConfig } from '@playwright/test'
import fs from 'fs'
import path from 'path'

async function globalTeardown(config: FullConfig) {
  console.log('🧹 Starting Playwright global teardown...')
  
  try {
    // Clean up auth files if needed
    const authDir = path.join(__dirname, '../../test-results/auth')
    
    if (fs.existsSync(authDir)) {
      const authFiles = fs.readdirSync(authDir)
      console.log(`🗑️  Found ${authFiles.length} auth files to clean up`)
      
      // Optionally keep auth state for debugging
      if (!process.env.KEEP_AUTH_STATE) {
        authFiles.forEach(file => {
          const filePath = path.join(authDir, file)
          fs.unlinkSync(filePath)
        })
        console.log('✅ Auth files cleaned up')
      } else {
        console.log('🔒 Keeping auth files for debugging')
      }
    }
    
    // Clean up temporary test files
    const tempDir = path.join(__dirname, '../../temp')
    if (fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true })
      console.log('✅ Temporary files cleaned up')
    }
    
    // Log test results summary
    const resultsDir = path.join(__dirname, '../../test-results')
    if (fs.existsSync(resultsDir)) {
      const resultFiles = fs.readdirSync(resultsDir)
      console.log(`📊 Generated ${resultFiles.length} result files`)
      
      // Check for specific result files
      const reportExists = resultFiles.includes('playwright-report')
      const jsonExists = resultFiles.some(f => f.endsWith('.json'))
      const xmlExists = resultFiles.some(f => f.endsWith('.xml'))
      
      if (reportExists) console.log('📈 HTML report generated')
      if (jsonExists) console.log('📄 JSON results generated')
      if (xmlExists) console.log('📋 XML results generated')
    }
    
    console.log('🎉 Playwright global teardown complete!')
    
  } catch (error) {
    console.error('❌ Global teardown failed:', error)
    // Don't throw to avoid masking test failures
  }
}

export default globalTeardown