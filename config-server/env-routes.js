const express = require('express');
const path = require('path');
const fs = require('fs').promises;
const Joi = require('joi');
const chokidar = require('chokidar');

const router = express.Router();

// Validation schemas
const envKeySchema = Joi.string().pattern(/^[A-Z][A-Z0-9_]*$/).max(100);
const envValueSchema = Joi.string().max(10000).allow('');
const envFileNameSchema = Joi.string().pattern(/^\.env(\.[a-zA-Z0-9_-]+)?$/).max(50);

// Default .env file path
const DEFAULT_ENV_PATH = process.env.ENV_FILES_PATH || process.cwd();

// File watcher for .env files
const watchers = new Map();

// Helper to safely parse .env file content
const parseEnvFile = (content) => {
  const lines = content.split('\n');
  const env = {};
  
  lines.forEach((line) => {
    // Skip empty lines and comments
    if (!line.trim() || line.trim().startsWith('#')) {
      return;
    }
    
    // Parse key=value pairs
    const separatorIndex = line.indexOf('=');
    if (separatorIndex > 0) {
      const key = line.substring(0, separatorIndex).trim();
      let value = line.substring(separatorIndex + 1).trim();
      
      // Remove surrounding quotes if present
      if ((value.startsWith('"') && value.endsWith('"')) || 
          (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }
      
      env[key] = value;
    }
  });
  
  return env;
};

// Helper to format env object back to string
const formatEnvFile = (env, preserveComments = true, originalContent = '') => {
  const lines = [];
  const processedKeys = new Set();
  
  if (preserveComments && originalContent) {
    // Preserve comments and ordering from original file
    const originalLines = originalContent.split('\n');
    
    originalLines.forEach((line) => {
      if (!line.trim() || line.trim().startsWith('#')) {
        lines.push(line);
      } else {
        const separatorIndex = line.indexOf('=');
        if (separatorIndex > 0) {
          const key = line.substring(0, separatorIndex).trim();
          if (env.hasOwnProperty(key)) {
            // Update existing key with new value
            lines.push(`${key}=${env[key]}`);
            processedKeys.add(key);
          }
          // Skip deleted keys
        }
      }
    });
    
    // Add new keys at the end
    Object.keys(env).forEach((key) => {
      if (!processedKeys.has(key)) {
        lines.push(`${key}=${env[key]}`);
      }
    });
  } else {
    // Simple format without preserving comments
    Object.entries(env).forEach(([key, value]) => {
      lines.push(`${key}=${value}`);
    });
  }
  
  return lines.join('\n');
};

// Get list of .env files
router.get('/files', async (req, res) => {
  try {
    const files = await fs.readdir(DEFAULT_ENV_PATH);
    const envFiles = files.filter(file => file.match(/^\.env(\.[a-zA-Z0-9_-]+)?$/));
    
    const fileDetails = await Promise.all(
      envFiles.map(async (file) => {
        const filePath = path.join(DEFAULT_ENV_PATH, file);
        const stats = await fs.stat(filePath);
        return {
          name: file,
          path: filePath,
          size: stats.size,
          modified: stats.mtime,
          watching: watchers.has(filePath)
        };
      })
    );
    
    res.json({ files: fileDetails });
  } catch (error) {
    console.error('Error listing env files:', error);
    res.status(500).json({ error: 'Failed to list env files' });
  }
});

// Read specific .env file
router.get('/files/:filename', async (req, res) => {
  try {
    const { error } = envFileNameSchema.validate(req.params.filename);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }
    
    const filePath = path.join(DEFAULT_ENV_PATH, req.params.filename);
    
    // Security check - ensure we're not reading outside the allowed directory
    const resolvedPath = path.resolve(filePath);
    const resolvedBase = path.resolve(DEFAULT_ENV_PATH);
    if (!resolvedPath.startsWith(resolvedBase)) {
      return res.status(403).json({ error: 'Access denied' });
    }
    
    const content = await fs.readFile(filePath, 'utf8');
    const parsed = parseEnvFile(content);
    
    res.json({
      filename: req.params.filename,
      path: filePath,
      content: content,
      parsed: parsed,
      variables: Object.keys(parsed).length
    });
  } catch (error) {
    if (error.code === 'ENOENT') {
      return res.status(404).json({ error: 'File not found' });
    }
    console.error('Error reading env file:', error);
    res.status(500).json({ error: 'Failed to read env file' });
  }
});

// Create or update entire .env file
router.put('/files/:filename', async (req, res) => {
  try {
    const { error: filenameError } = envFileNameSchema.validate(req.params.filename);
    if (filenameError) {
      return res.status(400).json({ error: filenameError.details[0].message });
    }
    
    const { content, variables } = req.body;
    
    if (!content && !variables) {
      return res.status(400).json({ error: 'Either content or variables must be provided' });
    }
    
    const filePath = path.join(DEFAULT_ENV_PATH, req.params.filename);
    
    // Security check
    const resolvedPath = path.resolve(filePath);
    const resolvedBase = path.resolve(DEFAULT_ENV_PATH);
    if (!resolvedPath.startsWith(resolvedBase)) {
      return res.status(403).json({ error: 'Access denied' });
    }
    
    let finalContent;
    
    if (content !== undefined) {
      // Direct content update
      finalContent = content;
    } else if (variables) {
      // Update from variables object
      // Validate all keys and values
      for (const [key, value] of Object.entries(variables)) {
        const keyError = envKeySchema.validate(key).error;
        if (keyError) {
          return res.status(400).json({ 
            error: `Invalid variable key: ${key}`,
            details: keyError.details[0].message 
          });
        }
        
        const valueError = envValueSchema.validate(value).error;
        if (valueError) {
          return res.status(400).json({ 
            error: `Invalid value for ${key}`,
            details: valueError.details[0].message 
          });
        }
      }
      
      // Try to preserve comments if file exists
      let originalContent = '';
      try {
        originalContent = await fs.readFile(filePath, 'utf8');
      } catch (error) {
        // File doesn't exist, that's okay
      }
      
      finalContent = formatEnvFile(variables, true, originalContent);
    }
    
    // Create backup before writing
    try {
      const backupPath = `${filePath}.backup`;
      const existingContent = await fs.readFile(filePath, 'utf8');
      await fs.writeFile(backupPath, existingContent);
    } catch (error) {
      // No existing file to backup
    }
    
    // Write the file
    await fs.writeFile(filePath, finalContent, 'utf8');
    
    // Broadcast update
    const broadcast = req.app.get('broadcast');
    broadcast({
      type: 'env:file:updated',
      filename: req.params.filename,
      timestamp: new Date().toISOString()
    });
    
    res.json({
      message: 'File updated successfully',
      filename: req.params.filename,
      path: filePath,
      size: Buffer.byteLength(finalContent, 'utf8')
    });
  } catch (error) {
    console.error('Error updating env file:', error);
    res.status(500).json({ error: 'Failed to update env file' });
  }
});

// Update specific variable in .env file
router.patch('/files/:filename/variables/:key', async (req, res) => {
  try {
    const { error: filenameError } = envFileNameSchema.validate(req.params.filename);
    if (filenameError) {
      return res.status(400).json({ error: filenameError.details[0].message });
    }
    
    const { error: keyError } = envKeySchema.validate(req.params.key);
    if (keyError) {
      return res.status(400).json({ error: keyError.details[0].message });
    }
    
    const { value } = req.body;
    if (value === undefined) {
      return res.status(400).json({ error: 'Value is required' });
    }
    
    const { error: valueError } = envValueSchema.validate(value);
    if (valueError) {
      return res.status(400).json({ error: valueError.details[0].message });
    }
    
    const filePath = path.join(DEFAULT_ENV_PATH, req.params.filename);
    
    // Security check
    const resolvedPath = path.resolve(filePath);
    const resolvedBase = path.resolve(DEFAULT_ENV_PATH);
    if (!resolvedPath.startsWith(resolvedBase)) {
      return res.status(403).json({ error: 'Access denied' });
    }
    
    // Read existing file
    let content = '';
    let env = {};
    
    try {
      content = await fs.readFile(filePath, 'utf8');
      env = parseEnvFile(content);
    } catch (error) {
      if (error.code !== 'ENOENT') {
        throw error;
      }
      // File doesn't exist, we'll create it
    }
    
    // Update the variable
    env[req.params.key] = value;
    
    // Format and write back
    const newContent = formatEnvFile(env, true, content);
    await fs.writeFile(filePath, newContent, 'utf8');
    
    // Broadcast update
    const broadcast = req.app.get('broadcast');
    broadcast({
      type: 'env:variable:updated',
      filename: req.params.filename,
      key: req.params.key,
      timestamp: new Date().toISOString()
    });
    
    res.json({
      message: 'Variable updated successfully',
      filename: req.params.filename,
      key: req.params.key,
      value: value
    });
  } catch (error) {
    console.error('Error updating variable:', error);
    res.status(500).json({ error: 'Failed to update variable' });
  }
});

// Delete specific variable from .env file
router.delete('/files/:filename/variables/:key', async (req, res) => {
  try {
    const { error: filenameError } = envFileNameSchema.validate(req.params.filename);
    if (filenameError) {
      return res.status(400).json({ error: filenameError.details[0].message });
    }
    
    const { error: keyError } = envKeySchema.validate(req.params.key);
    if (keyError) {
      return res.status(400).json({ error: keyError.details[0].message });
    }
    
    const filePath = path.join(DEFAULT_ENV_PATH, req.params.filename);
    
    // Security check
    const resolvedPath = path.resolve(filePath);
    const resolvedBase = path.resolve(DEFAULT_ENV_PATH);
    if (!resolvedPath.startsWith(resolvedBase)) {
      return res.status(403).json({ error: 'Access denied' });
    }
    
    // Read existing file
    const content = await fs.readFile(filePath, 'utf8');
    const env = parseEnvFile(content);
    
    if (!env.hasOwnProperty(req.params.key)) {
      return res.status(404).json({ error: 'Variable not found' });
    }
    
    // Delete the variable
    delete env[req.params.key];
    
    // Format and write back
    const newContent = formatEnvFile(env, true, content);
    await fs.writeFile(filePath, newContent, 'utf8');
    
    // Broadcast update
    const broadcast = req.app.get('broadcast');
    broadcast({
      type: 'env:variable:deleted',
      filename: req.params.filename,
      key: req.params.key,
      timestamp: new Date().toISOString()
    });
    
    res.json({
      message: 'Variable deleted successfully',
      filename: req.params.filename,
      key: req.params.key
    });
  } catch (error) {
    if (error.code === 'ENOENT') {
      return res.status(404).json({ error: 'File not found' });
    }
    console.error('Error deleting variable:', error);
    res.status(500).json({ error: 'Failed to delete variable' });
  }
});

// Watch .env file for changes
router.post('/files/:filename/watch', async (req, res) => {
  try {
    const { error } = envFileNameSchema.validate(req.params.filename);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }
    
    const filePath = path.join(DEFAULT_ENV_PATH, req.params.filename);
    
    // Security check
    const resolvedPath = path.resolve(filePath);
    const resolvedBase = path.resolve(DEFAULT_ENV_PATH);
    if (!resolvedPath.startsWith(resolvedBase)) {
      return res.status(403).json({ error: 'Access denied' });
    }
    
    // Check if already watching
    if (watchers.has(filePath)) {
      return res.json({
        message: 'Already watching file',
        filename: req.params.filename
      });
    }
    
    // Create watcher
    const watcher = chokidar.watch(filePath, {
      persistent: true,
      ignoreInitial: true
    });
    
    const broadcast = req.app.get('broadcast');
    
    watcher.on('change', async () => {
      try {
        const content = await fs.readFile(filePath, 'utf8');
        const parsed = parseEnvFile(content);
        
        broadcast({
          type: 'env:file:changed',
          filename: req.params.filename,
          parsed: parsed,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        console.error('Error reading changed file:', error);
      }
    });
    
    watcher.on('unlink', () => {
      broadcast({
        type: 'env:file:deleted',
        filename: req.params.filename,
        timestamp: new Date().toISOString()
      });
      
      // Stop watching if file is deleted
      watcher.close();
      watchers.delete(filePath);
    });
    
    watchers.set(filePath, watcher);
    
    res.json({
      message: 'Started watching file',
      filename: req.params.filename
    });
  } catch (error) {
    console.error('Error setting up file watch:', error);
    res.status(500).json({ error: 'Failed to watch file' });
  }
});

// Stop watching .env file
router.delete('/files/:filename/watch', (req, res) => {
  try {
    const { error } = envFileNameSchema.validate(req.params.filename);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }
    
    const filePath = path.join(DEFAULT_ENV_PATH, req.params.filename);
    
    const watcher = watchers.get(filePath);
    if (!watcher) {
      return res.status(404).json({ error: 'Not watching this file' });
    }
    
    watcher.close();
    watchers.delete(filePath);
    
    res.json({
      message: 'Stopped watching file',
      filename: req.params.filename
    });
  } catch (error) {
    console.error('Error stopping file watch:', error);
    res.status(500).json({ error: 'Failed to stop watching file' });
  }
});

// Cleanup on module unload
process.on('beforeExit', () => {
  watchers.forEach((watcher) => {
    watcher.close();
  });
  watchers.clear();
});

module.exports = router;