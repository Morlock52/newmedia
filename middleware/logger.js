const winston = require('winston');

const formatJson = process.env.LOG_FORMAT === 'json';

const transports = [new winston.transports.Console()];

// Add file transport if LOG_FILE is provided (rotating style via maxsize/maxFiles)
const logFile = process.env.LOG_FILE || process.env.LOG_PATH || '/var/log/api/app.log';
if (logFile) {
  try {
    transports.push(new winston.transports.File({ filename: logFile, maxsize: 10 * 1024 * 1024, maxFiles: 5 }));
  } catch (e) {
    // ignore if cannot create file transport
  }
}

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    formatJson ? winston.format.json() : winston.format.simple()
  ),
  transports
});

module.exports = logger;
