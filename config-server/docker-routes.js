const express = require('express');
const { exec } = require('child_process');
const { promisify } = require('util');
const Joi = require('joi');
const path = require('path');
const fs = require('fs').promises;

const router = express.Router();
const execAsync = promisify(exec);

// Validation schemas
const serviceNameSchema = Joi.string().alphanum().min(1).max(50);
const dockerComposePathSchema = Joi.string().max(500);

// Docker compose file path (configurable via environment)
const DOCKER_COMPOSE_PATH = process.env.DOCKER_COMPOSE_PATH || path.join(process.cwd(), 'docker-compose.yml');

// Helper function to execute docker-compose commands
const executeDockerCompose = async (command, options = {}) => {
  const dockerComposeCmd = `docker-compose -f "${DOCKER_COMPOSE_PATH}" ${command}`;
  
  try {
    const { stdout, stderr } = await execAsync(dockerComposeCmd, {
      ...options,
      timeout: 30000 // 30 second timeout
    });
    
    return { success: true, stdout: stdout.trim(), stderr: stderr.trim() };
  } catch (error) {
    return {
      success: false,
      error: error.message,
      stdout: error.stdout?.trim(),
      stderr: error.stderr?.trim(),
      code: error.code
    };
  }
};

// Validate docker-compose file exists
const validateDockerComposeFile = async () => {
  try {
    await fs.access(DOCKER_COMPOSE_PATH);
    return true;
  } catch {
    return false;
  }
};

// Get all services status
router.get('/services', async (req, res) => {
  try {
    if (!await validateDockerComposeFile()) {
      return res.status(404).json({ error: 'Docker compose file not found' });
    }

    const result = await executeDockerCompose('ps --format json');
    
    if (!result.success) {
      return res.status(500).json({
        error: 'Failed to get services status',
        details: result.stderr || result.error
      });
    }

    // Parse the JSON output (if docker-compose supports it)
    let services = [];
    try {
      if (result.stdout) {
        // Handle different docker-compose versions
        const lines = result.stdout.split('\n').filter(line => line.trim());
        services = lines.map(line => {
          try {
            return JSON.parse(line);
          } catch {
            // Fallback for non-JSON output
            return { raw: line };
          }
        });
      }
    } catch (parseError) {
      console.error('Error parsing docker-compose output:', parseError);
      services = [{ raw: result.stdout }];
    }

    // Broadcast status update
    const broadcast = req.app.get('broadcast');
    broadcast({
      type: 'docker:services:status',
      data: services,
      timestamp: new Date().toISOString()
    });

    res.json({
      services,
      raw: result.stdout
    });
  } catch (error) {
    console.error('Error getting services:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get specific service status
router.get('/services/:name', async (req, res) => {
  try {
    const { error } = serviceNameSchema.validate(req.params.name);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    if (!await validateDockerComposeFile()) {
      return res.status(404).json({ error: 'Docker compose file not found' });
    }

    const result = await executeDockerCompose(`ps ${req.params.name}`);
    
    if (!result.success) {
      return res.status(500).json({
        error: 'Failed to get service status',
        details: result.stderr || result.error
      });
    }

    res.json({
      service: req.params.name,
      status: result.stdout,
      running: result.stdout.includes('Up')
    });
  } catch (error) {
    console.error('Error getting service status:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Start service(s)
router.post('/services/start', async (req, res) => {
  try {
    const { services } = req.body;
    
    if (!await validateDockerComposeFile()) {
      return res.status(404).json({ error: 'Docker compose file not found' });
    }

    // If specific services provided, validate them
    let command = 'up -d';
    if (services && Array.isArray(services)) {
      for (const service of services) {
        const { error } = serviceNameSchema.validate(service);
        if (error) {
          return res.status(400).json({ 
            error: `Invalid service name: ${service}`,
            details: error.details[0].message 
          });
        }
      }
      command += ' ' + services.join(' ');
    }

    const result = await executeDockerCompose(command);
    
    if (!result.success) {
      return res.status(500).json({
        error: 'Failed to start services',
        details: result.stderr || result.error
      });
    }

    // Broadcast event
    const broadcast = req.app.get('broadcast');
    broadcast({
      type: 'docker:services:started',
      services: services || 'all',
      timestamp: new Date().toISOString()
    });

    res.json({
      message: 'Services started successfully',
      services: services || 'all',
      output: result.stdout
    });
  } catch (error) {
    console.error('Error starting services:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Stop service(s)
router.post('/services/stop', async (req, res) => {
  try {
    const { services } = req.body;
    
    if (!await validateDockerComposeFile()) {
      return res.status(404).json({ error: 'Docker compose file not found' });
    }

    let command = 'stop';
    if (services && Array.isArray(services)) {
      for (const service of services) {
        const { error } = serviceNameSchema.validate(service);
        if (error) {
          return res.status(400).json({ 
            error: `Invalid service name: ${service}`,
            details: error.details[0].message 
          });
        }
      }
      command += ' ' + services.join(' ');
    }

    const result = await executeDockerCompose(command);
    
    if (!result.success) {
      return res.status(500).json({
        error: 'Failed to stop services',
        details: result.stderr || result.error
      });
    }

    // Broadcast event
    const broadcast = req.app.get('broadcast');
    broadcast({
      type: 'docker:services:stopped',
      services: services || 'all',
      timestamp: new Date().toISOString()
    });

    res.json({
      message: 'Services stopped successfully',
      services: services || 'all',
      output: result.stdout
    });
  } catch (error) {
    console.error('Error stopping services:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Restart service(s)
router.post('/services/restart', async (req, res) => {
  try {
    const { services } = req.body;
    
    if (!await validateDockerComposeFile()) {
      return res.status(404).json({ error: 'Docker compose file not found' });
    }

    let command = 'restart';
    if (services && Array.isArray(services)) {
      for (const service of services) {
        const { error } = serviceNameSchema.validate(service);
        if (error) {
          return res.status(400).json({ 
            error: `Invalid service name: ${service}`,
            details: error.details[0].message 
          });
        }
      }
      command += ' ' + services.join(' ');
    }

    const result = await executeDockerCompose(command);
    
    if (!result.success) {
      return res.status(500).json({
        error: 'Failed to restart services',
        details: result.stderr || result.error
      });
    }

    // Broadcast event
    const broadcast = req.app.get('broadcast');
    broadcast({
      type: 'docker:services:restarted',
      services: services || 'all',
      timestamp: new Date().toISOString()
    });

    res.json({
      message: 'Services restarted successfully',
      services: services || 'all',
      output: result.stdout
    });
  } catch (error) {
    console.error('Error restarting services:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get service logs
router.get('/services/:name/logs', async (req, res) => {
  try {
    const { error } = serviceNameSchema.validate(req.params.name);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    if (!await validateDockerComposeFile()) {
      return res.status(404).json({ error: 'Docker compose file not found' });
    }

    const tail = req.query.tail || '100';
    const follow = req.query.follow === 'true';

    const command = `logs ${follow ? '-f' : ''} --tail=${tail} ${req.params.name}`;
    
    if (follow) {
      // For streaming logs, we need to handle it differently
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      });

      const { spawn } = require('child_process');
      const dockerProcess = spawn('docker-compose', 
        ['-f', DOCKER_COMPOSE_PATH, 'logs', '-f', `--tail=${tail}`, req.params.name]
      );

      dockerProcess.stdout.on('data', (data) => {
        res.write(`data: ${data.toString()}\n\n`);
      });

      dockerProcess.stderr.on('data', (data) => {
        res.write(`data: [ERROR] ${data.toString()}\n\n`);
      });

      req.on('close', () => {
        dockerProcess.kill();
      });

      return;
    }

    const result = await executeDockerCompose(command);
    
    if (!result.success) {
      return res.status(500).json({
        error: 'Failed to get service logs',
        details: result.stderr || result.error
      });
    }

    res.json({
      service: req.params.name,
      logs: result.stdout,
      tail: tail
    });
  } catch (error) {
    console.error('Error getting service logs:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Pull latest images
router.post('/services/pull', async (req, res) => {
  try {
    const { services } = req.body;
    
    if (!await validateDockerComposeFile()) {
      return res.status(404).json({ error: 'Docker compose file not found' });
    }

    let command = 'pull';
    if (services && Array.isArray(services)) {
      for (const service of services) {
        const { error } = serviceNameSchema.validate(service);
        if (error) {
          return res.status(400).json({ 
            error: `Invalid service name: ${service}`,
            details: error.details[0].message 
          });
        }
      }
      command += ' ' + services.join(' ');
    }

    const result = await executeDockerCompose(command);
    
    if (!result.success) {
      return res.status(500).json({
        error: 'Failed to pull images',
        details: result.stderr || result.error
      });
    }

    // Broadcast event
    const broadcast = req.app.get('broadcast');
    broadcast({
      type: 'docker:images:pulled',
      services: services || 'all',
      timestamp: new Date().toISOString()
    });

    res.json({
      message: 'Images pulled successfully',
      services: services || 'all',
      output: result.stdout
    });
  } catch (error) {
    console.error('Error pulling images:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;