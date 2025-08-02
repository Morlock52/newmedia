const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

module.exports = async () => {
  console.log('\n🚀 Starting test environment setup...\n');

  try {
    // Create reports directory
    const reportsDir = path.join(__dirname, '../reports');
    if (!fs.existsSync(reportsDir)) {
      fs.mkdirSync(reportsDir, { recursive: true });
    }

    // Check if Docker is running
    try {
      execSync('docker info', { stdio: 'ignore' });
    } catch (err) {
      throw new Error('Docker is not running. Please start Docker and try again.');
    }

    // Stop any existing test containers
    console.log('🧹 Cleaning up existing test containers...');
    try {
      execSync('docker-compose -f docker-compose.test.yml down -v', {
        cwd: path.join(__dirname, '..'),
        stdio: 'inherit'
      });
    } catch (err) {
      // Ignore errors if containers don't exist
    }

    // Start test environment
    console.log('🐳 Starting test containers...');
    execSync('docker-compose -f docker-compose.test.yml up -d test-postgres test-redis mock-tmdb mock-indexer', {
      cwd: path.join(__dirname, '..'),
      stdio: 'inherit'
    });

    // Wait for services to be healthy
    console.log('⏳ Waiting for services to be healthy...');
    execSync('docker-compose -f docker-compose.test.yml up -d --wait', {
      cwd: path.join(__dirname, '..'),
      stdio: 'inherit'
    });

    // Start main services for testing
    console.log('🚀 Starting application services...');
    execSync('docker-compose up -d jellyfin sonarr radarr prowlarr overseerr homepage grafana prometheus', {
      cwd: path.join(__dirname, '../..'),
      stdio: 'inherit'
    });

    // Store test start time
    global.__TEST_START_TIME__ = Date.now();

    console.log('\n✅ Test environment setup complete!\n');
  } catch (error) {
    console.error('\n❌ Test environment setup failed:', error.message);
    
    // Cleanup on failure
    try {
      execSync('docker-compose -f docker-compose.test.yml down -v', {
        cwd: path.join(__dirname, '..'),
        stdio: 'ignore'
      });
    } catch (cleanupError) {
      // Ignore cleanup errors
    }
    
    throw error;
  }
};