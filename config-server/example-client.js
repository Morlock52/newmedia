const WebSocket = require('ws');
const readline = require('readline');

// Configuration
const SERVER_URL = 'http://localhost:3000';
const WS_URL = 'ws://localhost:3000';

// Create readline interface for user input
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Helper function to make API requests
async function apiRequest(method, path, token, body = null) {
  const options = {
    method,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': token ? `Bearer ${token}` : undefined
    }
  };
  
  if (body) {
    options.body = JSON.stringify(body);
  }
  
  const response = await fetch(`${SERVER_URL}${path}`, options);
  const data = await response.json();
  
  if (!response.ok) {
    throw new Error(data.error || 'Request failed');
  }
  
  return data;
}

// Main client function
async function runClient() {
  console.log('Config Server Example Client');
  console.log('==========================\n');
  
  try {
    // Step 1: Login
    console.log('Logging in...');
    const loginData = await apiRequest('POST', '/api/auth/login', null, {
      username: 'admin',
      password: 'admin' // Change this to match your password
    });
    
    console.log('✓ Logged in successfully');
    const token = loginData.token;
    
    // Step 2: Connect WebSocket
    console.log('\nConnecting to WebSocket...');
    const ws = new WebSocket(`${WS_URL}?token=${token}`);
    
    ws.on('open', () => {
      console.log('✓ WebSocket connected');
      console.log('\nListening for real-time events...\n');
    });
    
    ws.on('message', (data) => {
      const event = JSON.parse(data);
      console.log(`[EVENT] ${event.type}:`, event);
    });
    
    ws.on('error', (error) => {
      console.error('WebSocket error:', error);
    });
    
    ws.on('close', () => {
      console.log('WebSocket disconnected');
      process.exit(0);
    });
    
    // Step 3: Interactive menu
    const showMenu = () => {
      console.log('\nAvailable Commands:');
      console.log('1. List .env files');
      console.log('2. Read .env file');
      console.log('3. Update variable');
      console.log('4. List Docker services');
      console.log('5. Start Docker service');
      console.log('6. Stop Docker service');
      console.log('7. Watch .env file');
      console.log('8. Exit');
      console.log('');
    };
    
    const handleCommand = async (command) => {
      try {
        switch (command) {
          case '1':
            const files = await apiRequest('GET', '/api/env/files', token);
            console.log('Environment files:', files);
            break;
            
          case '2':
            rl.question('Enter filename (e.g., .env): ', async (filename) => {
              const file = await apiRequest('GET', `/api/env/files/${filename}`, token);
              console.log(`\nFile: ${filename}`);
              console.log('Variables:', file.parsed);
              showMenu();
            });
            return;
            
          case '3':
            rl.question('Enter filename (e.g., .env): ', (filename) => {
              rl.question('Enter variable name: ', (key) => {
                rl.question('Enter new value: ', async (value) => {
                  const result = await apiRequest(
                    'PATCH',
                    `/api/env/files/${filename}/variables/${key}`,
                    token,
                    { value }
                  );
                  console.log('✓ Variable updated:', result);
                  showMenu();
                });
              });
            });
            return;
            
          case '4':
            const services = await apiRequest('GET', '/api/docker/services', token);
            console.log('Docker services:', services);
            break;
            
          case '5':
            rl.question('Enter service name (or press Enter for all): ', async (service) => {
              const body = service ? { services: [service] } : {};
              const result = await apiRequest('POST', '/api/docker/services/start', token, body);
              console.log('✓ Services started:', result);
              showMenu();
            });
            return;
            
          case '6':
            rl.question('Enter service name (or press Enter for all): ', async (service) => {
              const body = service ? { services: [service] } : {};
              const result = await apiRequest('POST', '/api/docker/services/stop', token, body);
              console.log('✓ Services stopped:', result);
              showMenu();
            });
            return;
            
          case '7':
            rl.question('Enter filename to watch (e.g., .env): ', async (filename) => {
              const result = await apiRequest('POST', `/api/env/files/${filename}/watch`, token);
              console.log('✓ Watching file:', result);
              console.log('(File changes will appear as WebSocket events)');
              showMenu();
            });
            return;
            
          case '8':
            console.log('Goodbye!');
            ws.close();
            rl.close();
            process.exit(0);
            
          default:
            console.log('Invalid command');
        }
        
        showMenu();
      } catch (error) {
        console.error('Error:', error.message);
        showMenu();
      }
    };
    
    // Show initial menu
    showMenu();
    
    // Handle user input
    rl.on('line', (input) => {
      handleCommand(input.trim());
    });
    
  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

// Run the client
runClient().catch(console.error);