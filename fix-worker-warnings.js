#!/usr/bin/env node

/**
 * Fix for MaxListenersExceededWarning in Worker threads
 * 
 * This script sets up proper configuration for Worker threads
 * to prevent memory leak warnings.
 */

const { Worker } = require('worker_threads');
const EventEmitter = require('events');

// Option 1: Increase the default max listeners globally
EventEmitter.defaultMaxListeners = 20;

// Option 2: Set max listeners for specific Worker instances
function createWorkerWithMaxListeners(scriptPath, options = {}) {
    const worker = new Worker(scriptPath, options);
    
    // Increase max listeners for this specific worker
    worker.setMaxListeners(20);
    
    // Ensure proper cleanup when worker exits
    worker.on('exit', (code) => {
        // Remove all listeners to prevent memory leaks
        worker.removeAllListeners();
    });
    
    return worker;
}

// Option 3: Environment variable to suppress warnings (temporary fix)
// Add this to your .env or shell:
// NODE_OPTIONS="--max-old-space-size=4096 --max-listeners=20"

console.log('Worker listener configuration applied.');
console.log('Default max listeners set to:', EventEmitter.defaultMaxListeners);

module.exports = {
    createWorkerWithMaxListeners,
    defaultMaxListeners: EventEmitter.defaultMaxListeners
};