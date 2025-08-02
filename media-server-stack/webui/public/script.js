// Media Server Stack Web UI JavaScript

let currentTab = 'setup';
let setupProgress = {
  prerequisites: true,
  environment: false,
  security: false,
  deployment: false
};

// Tab Management
function showTab(tabName) {
  // Hide all tab contents
  document.querySelectorAll('.tab-content').forEach(content => {
    content.classList.remove('active');
  });
  
  // Hide all tab buttons
  document.querySelectorAll('.tab').forEach(tab => {
    tab.classList.remove('active');
  });
  
  // Show selected tab
  document.getElementById(tabName).classList.add('active');
  event.target.classList.add('active');
  
  currentTab = tabName;
  
  // Load tab-specific data
  if (tabName === 'management') {
    loadManagementData();
  } else if (tabName === 'monitoring') {
    loadMonitoringData();
  } else if (tabName === 'setup') {
    loadSetupData();
  }
}

// Setup Functions
async function loadSetupData() {
  try {
    // Auto-detect system values
    const response = await fetch('/api/system-info');
    if (response.ok) {
      const systemInfo = await response.json();
      
      document.getElementById('puid').value = systemInfo.puid || '1000';
      document.getElementById('timezone').value = systemInfo.timezone || 'UTC';
      
      // Check if .env exists
      const envResponse = await fetch('/api/env-status');
      if (envResponse.ok) {
        const envStatus = await envResponse.json();
        if (envStatus.exists) {
          showAlert('Environment file detected. You can modify existing configuration.', 'warning');
          loadExistingEnvironment();
        }
      }
    }
  } catch (error) {
    console.error('Error loading setup data:', error);
  }
}

async function loadExistingEnvironment() {
  try {
    const response = await fetch('/api/env-config');
    if (response.ok) {
      const config = await response.json();
      
      // Populate form with existing values
      Object.keys(config).forEach(key => {
        const element = document.getElementById(key.toLowerCase());
        if (element) {
          element.value = config[key];
        }
      });
    }
  } catch (error) {
    console.error('Error loading environment:', error);
  }
}

function loadDefaults() {
  // Load sensible defaults
  document.getElementById('domain').value = 'localhost';
  document.getElementById('email').value = 'admin@localhost';
  document.getElementById('vpnProvider').value = 'pia';
  document.getElementById('vpnType').value = 'wireguard';
  document.getElementById('dataRoot').value = './data';
  document.getElementById('configRoot').value = './config';
  
  showAlert('Default values loaded. Modify as needed for your setup.', 'success');
}

async function validateSetup() {
  const button = event.target;
  const loading = button.querySelector('.loading');
  const originalText = button.textContent;
  
  button.disabled = true;
  loading.classList.remove('hidden');
  button.innerHTML = '<span class="loading"></span> Validating...';
  
  try {
    const formData = new FormData(document.getElementById('setupForm'));
    const config = {};
    
    formData.forEach((value, key) => {
      config[key] = value;
    });
    
    const response = await fetch('/api/validate-config', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config)
    });
    
    const result = await response.json();
    
    if (response.ok) {
      showAlert('✅ Configuration is valid! Ready to generate environment.', 'success');
      updateSetupProgress('environment', true);
    } else {
      showAlert(`❌ Validation failed: ${result.message}`, 'error');
      showOutput('setupOutput', result.details || result.message);
    }
    
  } catch (error) {
    showAlert(`❌ Validation error: ${error.message}`, 'error');
  } finally {
    button.disabled = false;
    button.textContent = originalText;
    loading.classList.add('hidden');
  }
}

async function runSetup() {
  const button = event.target;
  button.disabled = true;
  button.textContent = 'Generating...';
  
  try {
    const formData = new FormData(document.getElementById('setupForm'));
    const config = {};
    
    formData.forEach((value, key) => {
      config[key] = value;
    });
    
    const response = await fetch('/api/setup-environment', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config)
    });
    
    const result = await response.text();
    
    if (response.ok) {
      showAlert('✅ Environment configuration generated successfully!', 'success');
      showOutput('setupOutput', result);
      updateSetupProgress('environment', true);
      updateSetupProgress('security', true);
    } else {
      showAlert('❌ Setup failed. Check the output below.', 'error');
      showOutput('setupOutput', result);
    }
    
  } catch (error) {
    showAlert(`❌ Setup error: ${error.message}`, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Generate Environment';
  }
}

// Management Functions
async function loadManagementData() {
  checkEnvironment();
  checkDocker();
  fetchStatus();
}

async function checkEnvironment() {
  try {
    const response = await fetch('/api/env-status');
    const status = await response.json();
    
    const statusElement = document.getElementById('envStatus');
    if (status.exists) {
      statusElement.textContent = `✅ Environment configured (${status.variables} variables)`;
      statusElement.parentElement.classList.add('success');
    } else {
      statusElement.textContent = '❌ Environment not configured';
      statusElement.parentElement.classList.add('error');
    }
  } catch (error) {
    document.getElementById('envStatus').textContent = '❌ Error checking environment';
  }
}

async function checkDocker() {
  try {
    const response = await fetch('/api/docker-status');
    const status = await response.json();
    
    const statusElement = document.getElementById('dockerStatus');
    if (status.running) {
      statusElement.textContent = `✅ Docker running (${status.version})`;
      statusElement.parentElement.classList.add('success');
    } else {
      statusElement.textContent = '❌ Docker not running';
      statusElement.parentElement.classList.add('error');
    }
  } catch (error) {
    document.getElementById('dockerStatus').textContent = '❌ Error checking Docker';
  }
}

async function fetchStatus() {
  try {
    const response = await fetch('/api/status');
    const result = await response.text();
    
    // Parse service status and update UI
    const services = parseServiceStatus(result);
    updateServiceGrid(services);
    
    document.getElementById('serviceCount').textContent = 
      `${services.filter(s => s.status === 'running').length} of ${services.length} services running`;
    
    showOutput('managementOutput', result);
  } catch (error) {
    showAlert(`❌ Error fetching status: ${error.message}`, 'error');
  }
}

async function startStack() {
  const button = event.target;
  button.disabled = true;
  button.textContent = 'Starting...';
  
  try {
    const response = await fetch('/api/start', { method: 'POST' });
    const result = await response.text();
    
    if (response.ok) {
      showAlert('✅ Stack started successfully!', 'success');
      updateSetupProgress('deployment', true);
    } else {
      showAlert('❌ Failed to start stack', 'error');
    }
    
    showOutput('managementOutput', result);
    
    // Refresh status after starting
    setTimeout(fetchStatus, 3000);
    
  } catch (error) {
    showAlert(`❌ Start error: ${error.message}`, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Start All';
  }
}

async function stopStack() {
  const button = event.target;
  button.disabled = true;
  button.textContent = 'Stopping...';
  
  try {
    const response = await fetch('/api/stop', { method: 'POST' });
    const result = await response.text();
    
    if (response.ok) {
      showAlert('✅ Stack stopped successfully!', 'success');
    } else {
      showAlert('❌ Failed to stop stack', 'error');
    }
    
    showOutput('managementOutput', result);
    
    // Refresh status after stopping
    setTimeout(fetchStatus, 2000);
    
  } catch (error) {
    showAlert(`❌ Stop error: ${error.message}`, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Stop All';
  }
}

async function restartStack() {
  const button = event.target;
  button.disabled = true;
  button.textContent = 'Restarting...';
  
  try {
    await stopStack();
    await new Promise(resolve => setTimeout(resolve, 3000)); // Wait 3 seconds
    await startStack();
    
    showAlert('✅ Stack restarted successfully!', 'success');
  } catch (error) {
    showAlert(`❌ Restart error: ${error.message}`, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Restart Stack';
  }
}

// Deployment Functions
async function deployStack() {
  const button = event.target;
  button.disabled = true;
  button.textContent = 'Deploying...';
  
  try {
    showAlert('🚀 Starting stack deployment...', 'success');
    showOutput('managementOutput', 'Starting deployment...\n');
    
    const response = await fetch('/api/deploy', { method: 'POST' });
    const result = await response.text();
    
    if (response.ok) {
      showAlert('✅ Stack deployed successfully!', 'success');
      updateSetupProgress('deployment', true);
    } else {
      showAlert('❌ Deployment failed. Check output below.', 'error');
    }
    
    showOutput('managementOutput', result);
    
    // Refresh status after deployment
    setTimeout(fetchStatus, 5000);
    
  } catch (error) {
    showAlert(`❌ Deployment error: ${error.message}`, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Deploy Complete Stack';
  }
}

async function deployWithMonitoring() {
  const button = event.target;
  button.disabled = true;
  button.textContent = 'Deploying with Monitoring...';
  
  try {
    showAlert('🚀 Starting stack deployment with monitoring...', 'success');
    showOutput('managementOutput', 'Starting deployment with monitoring stack...\n');
    
    const response = await fetch('/api/deploy-monitoring', { method: 'POST' });
    const result = await response.text();
    
    if (response.ok) {
      showAlert('✅ Stack with monitoring deployed successfully!', 'success');
      updateSetupProgress('deployment', true);
    } else {
      showAlert('❌ Deployment failed. Check output below.', 'error');
    }
    
    showOutput('managementOutput', result);
    
    // Refresh status after deployment
    setTimeout(fetchStatus, 5000);
    
  } catch (error) {
    showAlert(`❌ Deployment error: ${error.message}`, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Deploy + Monitoring';
  }
}

async function updateStack() {
  const button = event.target;
  button.disabled = true;
  button.textContent = 'Updating...';
  
  try {
    const response = await fetch('/api/update', { method: 'POST' });
    const result = await response.text();
    
    if (response.ok) {
      showAlert('✅ Stack updated successfully!', 'success');
    } else {
      showAlert('❌ Failed to update stack', 'error');
    }
    
    showOutput('managementOutput', result);
    
  } catch (error) {
    showAlert(`❌ Update error: ${error.message}`, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Update Images';
  }
}

async function openHealthCheck() {
  try {
    const response = await fetch('/api/health');
    const health = await response.json();
    
    let healthReport = `Health Check Report\n`;
    healthReport += `===================\n`;
    healthReport += `Timestamp: ${health.timestamp}\n`;
    healthReport += `Environment: ${health.environment ? '✅ Ready' : '❌ Not configured'}\n`;
    healthReport += `Docker: ${health.docker ? '✅ Running' : '❌ Not running'}\n`;
    healthReport += `Services: ${health.services.length} services found\n\n`;
    
    if (health.services.length > 0) {
      healthReport += `Service Details:\n`;
      health.services.forEach(service => {
        healthReport += `- ${service.name}: ${service.status} (${service.health})\n`;
      });
    }
    
    showOutput('managementOutput', healthReport);
    
  } catch (error) {
    showAlert(`❌ Health check error: ${error.message}`, 'error');
  }
}

async function viewDeployLogs() {
  try {
    const response = await fetch('/api/logs/all');
    const logs = await response.text();
    
    showOutput('managementOutput', logs || 'No deployment logs available');
    
  } catch (error) {
    showAlert(`❌ Error fetching deployment logs: ${error.message}`, 'error');
  }
}

// Monitoring Functions
async function loadMonitoringData() {
  refreshMonitoring();
  loadServiceLinks();
}

async function refreshMonitoring() {
  try {
    const response = await fetch('/api/system-stats');
    const stats = await response.json();
    
    document.getElementById('cpuUsage').textContent = `${stats.cpu}%`;
    document.getElementById('memoryUsage').textContent = `${stats.memory}%`;
    document.getElementById('diskUsage').textContent = `${stats.disk}%`;
    document.getElementById('networkUsage').textContent = `${stats.network}`;
    
  } catch (error) {
    console.error('Error refreshing monitoring:', error);
    document.getElementById('cpuUsage').textContent = 'Error';
    document.getElementById('memoryUsage').textContent = 'Error';
    document.getElementById('diskUsage').textContent = 'Error';
    document.getElementById('networkUsage').textContent = 'Error';
  }
}

function loadServiceLinks() {
  const services = [
    { name: 'Jellyfin', url: 'https://jellyfin.localhost', description: 'Media Server' },
    { name: 'Sonarr', url: 'https://sonarr.localhost', description: 'TV Shows' },
    { name: 'Radarr', url: 'https://radarr.localhost', description: 'Movies' },
    { name: 'Prowlarr', url: 'https://prowlarr.localhost', description: 'Indexer Manager' },
    { name: 'Overseerr', url: 'https://overseerr.localhost', description: 'Request Management' },
    { name: 'qBittorrent', url: 'https://qbittorrent.localhost', description: 'Torrent Client' },
    { name: 'Grafana', url: 'https://grafana.localhost', description: 'Monitoring' }
  ];
  
  const serviceLinksContainer = document.getElementById('serviceLinks');
  serviceLinksContainer.innerHTML = services.map(service => `
    <div class="service-card">
      <h4>${service.name}</h4>
      <p>${service.description}</p>
      <a href="${service.url}" target="_blank" class="btn btn-primary">Open</a>
    </div>
  `).join('');
}

// Logs Functions
async function fetchLogs() {
  const service = document.getElementById('logService').value;
  const button = event.target;
  
  button.disabled = true;
  button.textContent = 'Fetching...';
  
  try {
    const response = await fetch(`/api/logs/${service}`);
    const logs = await response.text();
    
    showOutput('logsOutput', logs);
    
  } catch (error) {
    showAlert(`❌ Error fetching logs: ${error.message}`, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Fetch Logs';
  }
}

function clearLogs() {
  document.getElementById('logsOutput').textContent = '';
}

async function downloadLogs() {
  const service = document.getElementById('logService').value;
  
  try {
    const response = await fetch(`/api/logs/${service}?download=true`);
    const blob = await response.blob();
    
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${service}-logs-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    
    showAlert('✅ Logs downloaded successfully!', 'success');
  } catch (error) {
    showAlert(`❌ Error downloading logs: ${error.message}`, 'error');
  }
}

// Utility Functions
function showAlert(message, type) {
  // Remove existing alerts
  document.querySelectorAll('.alert').forEach(alert => alert.remove());
  
  const alert = document.createElement('div');
  alert.className = `alert alert-${type}`;
  alert.textContent = message;
  
  // Insert at the top of the current tab content
  const currentTabContent = document.querySelector('.tab-content.active');
  currentTabContent.insertBefore(alert, currentTabContent.firstChild);
  
  // Auto-remove after 5 seconds
  setTimeout(() => {
    if (alert.parentNode) {
      alert.remove();
    }
  }, 5000);
}

function showOutput(elementId, text) {
  const output = document.getElementById(elementId);
  output.textContent = text;
  output.classList.remove('hidden');
  output.scrollTop = output.scrollHeight;
}

function updateSetupProgress(step, completed) {
  setupProgress[step] = completed;
  
  const steps = document.querySelectorAll('.progress-step');
  const stepMap = {
    'prerequisites': 0,
    'environment': 1,
    'security': 2,
    'deployment': 3
  };
  
  const stepElement = steps[stepMap[step]];
  if (stepElement) {
    stepElement.classList.remove('active', 'pending');
    stepElement.classList.add(completed ? 'completed' : 'active');
    
    const icon = stepElement.querySelector('.step-icon');
    icon.textContent = completed ? '✓' : stepMap[step] + 1;
  }
}

function parseServiceStatus(statusText) {
  const lines = statusText.split('\n').filter(line => line.trim());
  const services = [];
  
  for (const line of lines) {
    if (line.includes('Up') || line.includes('Exit') || line.includes('Created')) {
      const parts = line.split(/\s+/);
      const name = parts[0];
      const status = line.includes('Up') ? 'running' : 
                   line.includes('Exit') ? 'stopped' : 'starting';
      
      services.push({ name, status, info: line });
    }
  }
  
  return services;
}

function updateServiceGrid(services) {
  const grid = document.getElementById('serviceGrid');
  if (!grid) return;
  
  grid.innerHTML = services.map(service => `
    <div class="service-card">
      <h4>${service.name}</h4>
      <div class="service-status ${service.status}">${service.status}</div>
      <div class="help-text">${service.info}</div>
    </div>
  `).join('');
}

function openHealthCheck() {
  window.open('/api/health', '_blank');
}

function updateStack() {
  showAlert('⚠️ Update functionality coming soon!', 'warning');
}

function exportLogs() {
  showAlert('⚠️ Export functionality coming soon!', 'warning');
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
  loadSetupData();
  
  // Auto-refresh status every 30 seconds when on management tab
  setInterval(() => {
    if (currentTab === 'management') {
      fetchStatus();
    } else if (currentTab === 'monitoring') {
      refreshMonitoring();
    }
  }, 30000);
});
