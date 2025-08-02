const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

module.exports = async () => {
  console.log('\n🧹 Starting test environment cleanup...\n');

  try {
    // Calculate test duration
    const duration = Date.now() - global.__TEST_START_TIME__;
    const minutes = Math.floor(duration / 60000);
    const seconds = Math.floor((duration % 60000) / 1000);
    console.log(`⏱️  Total test duration: ${minutes}m ${seconds}s`);

    // Collect container logs for debugging
    console.log('📋 Collecting container logs...');
    const services = ['jellyfin', 'sonarr', 'radarr', 'prowlarr', 'test-postgres', 'test-redis'];
    const logsDir = path.join(__dirname, '../reports/logs');
    
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
    }

    for (const service of services) {
      try {
        const logs = execSync(`docker logs ${service} 2>&1`, { encoding: 'utf-8' });
        fs.writeFileSync(path.join(logsDir, `${service}.log`), logs);
      } catch (err) {
        // Service might not exist, ignore
      }
    }

    // Generate test summary
    const summaryPath = path.join(__dirname, '../reports/test-summary.json');
    const summary = {
      timestamp: new Date().toISOString(),
      duration: `${minutes}m ${seconds}s`,
      environment: process.env.NODE_ENV || 'test',
      dockerVersion: execSync('docker --version', { encoding: 'utf-8' }).trim(),
      composeVersion: execSync('docker-compose --version', { encoding: 'utf-8' }).trim()
    };
    
    fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));

    // Stop test containers
    console.log('🛑 Stopping test containers...');
    execSync('docker-compose -f docker-compose.test.yml down', {
      cwd: path.join(__dirname, '..'),
      stdio: 'inherit'
    });

    // Optionally stop main services (comment out to keep them running)
    if (process.env.CLEANUP_MAIN_SERVICES === 'true') {
      console.log('🛑 Stopping main services...');
      execSync('docker-compose down', {
        cwd: path.join(__dirname, '../..'),
        stdio: 'inherit'
      });
    }

    console.log('\n✅ Test environment cleanup complete!\n');
    console.log('📊 Test reports available in:', path.join(__dirname, '../reports'));
  } catch (error) {
    console.error('\n❌ Test environment cleanup failed:', error.message);
    // Don't throw to ensure Jest exits properly
  }
};