const winston = require('winston');

const formatJson = process.env.LOG_FORMAT === 'json';

const transports = [new winston.transports.Console()];

// Add rotating file transport if requested
const logFile = process.env.LOG_FILE || process.env.LOG_PATH || '/var/log/api/app.log';
const enableDailyRotate = process.env.LOG_DAILY_ROTATE === 'true';
if (logFile) {
  try {
    if (enableDailyRotate) {
      // use winston-daily-rotate-file if available
      try {
        const DailyRotateFile = require('winston-daily-rotate-file');
        transports.push(new DailyRotateFile({
          filename: logFile.replace(/\.log$/, '') + '-%DATE%.log',
          datePattern: 'YYYY-MM-DD',
          zippedArchive: true,
          maxSize: '20m',
          maxFiles: '14d'
        }));
      } catch (e) {
        // fallback to simple file transport
        transports.push(new winston.transports.File({ filename: logFile, maxsize: 10 * 1024 * 1024, maxFiles: 5 }));
      }
    } else {
      transports.push(new winston.transports.File({ filename: logFile, maxsize: 10 * 1024 * 1024, maxFiles: 5 }));
    }
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
