import winston from 'winston';

const formatJson = process.env.LOG_FORMAT === 'json';

const transports = [new winston.transports.Console()];
const logFile = process.env.LOG_FILE || process.env.LOG_PATH || '/var/log/voice-server/app.log';
if (logFile) {
  try {
    transports.push(new winston.transports.File({ filename: logFile, maxsize: 10 * 1024 * 1024, maxFiles: 5 }));
  } catch (e) {}
}

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    formatJson ? winston.format.json() : winston.format.simple()
  ),
  transports
});

export default logger;
