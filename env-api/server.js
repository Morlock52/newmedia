const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const rateLimit = require('express-rate-limit');

const app = express();
const PORT = 3456;

// Paths
const ENV_PATH = path.join(__dirname, '..', '.env');
const BACKUP_DIR = path.join(__dirname, 'backups');

// Ensure backup directory exists
if (!fs.existsSync(BACKUP_DIR)) {
  fs.mkdirSync(BACKUP_DIR, { recursive: true });
}

// Middleware
app.use(cors());
app.use(express.json());

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
app.use('/api/', limiter);

// Helper function to parse .env file preserving structure
function parseEnvFile(content) {
  const lines = content.split('\n');
  const result = {
    variables: {},
    raw: content,
    lines: []
  };

  lines.forEach((line, index) => {
    const trimmed = line.trim();
    
    // Parse line structure
    const lineInfo = {
      number: index + 1,
      content: line,
      type: 'empty'
    };

    if (trimmed.startsWith('#')) {
      lineInfo.type = 'comment';
    } else if (trimmed.includes('=')) {
      lineInfo.type = 'variable';
      const [key, ...valueParts] = line.split('=');
      const value = valueParts.join('=').trim();
      // Remove quotes if present
      const cleanValue = value.replace(/^["']|["']$/g, '');
      result.variables[key.trim()] = cleanValue;
      lineInfo.key = key.trim();
      lineInfo.value = cleanValue;
    }

    result.lines.push(lineInfo);
  });

  return result;
}

// Helper function to reconstruct .env content
function reconstructEnvContent(parsedData, updates = {}) {
  const lines = parsedData.lines.map(line => {
    if (line.type === 'variable' && updates[line.key] !== undefined) {
      // Update the value while preserving formatting
      const value = updates[line.key];
      // Add quotes if value contains spaces or special characters
      const quotedValue = /[\s#]/.test(value) ? `"${value}"` : value;
      return `${line.key}=${quotedValue}`;
    }
    return line.content;
  });

  return lines.join('\n');
}

// Create backup
function createBackup() {
  try {
    if (fs.existsSync(ENV_PATH)) {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const backupPath = path.join(BACKUP_DIR, `env-backup-${timestamp}.env`);
      fs.copyFileSync(ENV_PATH, backupPath);
      
      // Keep only last 10 backups
      const backups = fs.readdirSync(BACKUP_DIR)
        .filter(f => f.startsWith('env-backup-'))
        .sort()
        .reverse();
      
      if (backups.length > 10) {
        backups.slice(10).forEach(backup => {
          fs.unlinkSync(path.join(BACKUP_DIR, backup));
        });
      }
      
      return backupPath;
    }
  } catch (error) {
    console.error('Backup creation failed:', error);
  }
  return null;
}

// GET endpoint - Read current .env file
app.get('/api/env', (req, res) => {
  try {
    if (!fs.existsSync(ENV_PATH)) {
      return res.json({
        variables: {},
        raw: '',
        exists: false,
        message: '.env file not found'
      });
    }

    const content = fs.readFileSync(ENV_PATH, 'utf8');
    const parsed = parseEnvFile(content);
    
    res.json({
      variables: parsed.variables,
      raw: parsed.raw,
      exists: true,
      lastModified: fs.statSync(ENV_PATH).mtime
    });
  } catch (error) {
    console.error('Error reading .env file:', error);
    res.status(500).json({
      error: 'Failed to read .env file',
      message: error.message
    });
  }
});

// PUT endpoint - Update .env file
app.put('/api/env', (req, res) => {
  try {
    const updates = req.body;
    
    // Validate input
    if (!updates || typeof updates !== 'object') {
      return res.status(400).json({
        error: 'Invalid request body',
        message: 'Expected object with key-value pairs'
      });
    }

    // Validate keys and values
    for (const [key, value] of Object.entries(updates)) {
      if (!/^[A-Z_][A-Z0-9_]*$/i.test(key)) {
        return res.status(400).json({
          error: 'Invalid variable name',
          message: `Variable name '${key}' must contain only letters, numbers, and underscores`
        });
      }
      if (typeof value !== 'string') {
        return res.status(400).json({
          error: 'Invalid value type',
          message: `Value for '${key}' must be a string`
        });
      }
    }

    // Read current content
    let currentContent = '';
    let parsed;
    
    if (fs.existsSync(ENV_PATH)) {
      currentContent = fs.readFileSync(ENV_PATH, 'utf8');
      parsed = parseEnvFile(currentContent);
    } else {
      // Create new .env structure
      parsed = {
        variables: {},
        raw: '',
        lines: []
      };
    }

    // Create backup before making changes
    const backupPath = createBackup();

    // Apply updates
    const newContent = reconstructEnvContent(parsed, updates);
    
    // Write updated content
    fs.writeFileSync(ENV_PATH, newContent, 'utf8');

    // Read back to confirm
    const updatedContent = fs.readFileSync(ENV_PATH, 'utf8');
    const updatedParsed = parseEnvFile(updatedContent);

    res.json({
      success: true,
      variables: updatedParsed.variables,
      backup: backupPath,
      message: 'Environment variables updated successfully'
    });
  } catch (error) {
    console.error('Error updating .env file:', error);
    res.status(500).json({
      error: 'Failed to update .env file',
      message: error.message
    });
  }
});

// GET endpoint - List backups
app.get('/api/backups', (req, res) => {
  try {
    const backups = fs.readdirSync(BACKUP_DIR)
      .filter(f => f.startsWith('env-backup-'))
      .sort()
      .reverse()
      .map(filename => {
        const stats = fs.statSync(path.join(BACKUP_DIR, filename));
        return {
          filename,
          created: stats.mtime,
          size: stats.size
        };
      });

    res.json({ backups });
  } catch (error) {
    console.error('Error listing backups:', error);
    res.status(500).json({
      error: 'Failed to list backups',
      message: error.message
    });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    envFileExists: fs.existsSync(ENV_PATH)
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'An error occurred'
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Environment API server running on http://localhost:${PORT}`);
  console.log(`Reading .env file from: ${ENV_PATH}`);
  console.log(`Storing backups in: ${BACKUP_DIR}`);
  
  if (!fs.existsSync(ENV_PATH)) {
    console.warn('Warning: .env file not found at expected location');
  }
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully');
  process.exit(0);
});