/**
 * Security Monitoring Server
 * Real-time security monitoring and alerting system for media server infrastructure
 */

const express = require('express');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const slowDown = require('express-slow-down');
const cors = require('cors');
const compression = require('compression');
const morgan = require('morgan');
const winston = require('winston');
const nodemailer = require('nodemailer');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const cron = require('node-cron');
const Redis = require('ioredis');
const geoip = require('geoip-lite');
const UAParser = require('ua-parser-js');

const app = express();
const PORT = process.env.PORT || 3011;

// Logger configuration
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: '/var/log/monitor/error.log', level: 'error' }),
    new winston.transports.File({ filename: '/var/log/monitor/combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});

// Redis client for rate limiting and caching
const redis = new Redis({
  host: 'rate-limiter-redis',
  port: 6379,
  retryDelayOnFailover: 100,
  maxRetriesPerRequest: 3
});

// Email transporter
const emailTransporter = nodemailer.createTransporter({
  host: process.env.SMTP_HOST,
  port: process.env.SMTP_PORT || 587,
  secure: false,
  auth: {
    user: process.env.SMTP_USER,
    pass: process.env.SMTP_PASSWORD
  }
});

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      connectSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
      fontSrc: ["'self'"],
      objectSrc: ["'none'"],
      mediaSrc: ["'self'"],
      frameSrc: ["'none'"]
    }
  }
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
  store: new rateLimit.MemoryStore()
});

app.use(limiter);

// Slow down middleware
app.use(slowDown({
  windowMs: 15 * 60 * 1000,
  delayAfter: 50,
  delayMs: 500
}));

app.use(cors({
  origin: process.env.ALLOWED_ORIGINS ? process.env.ALLOWED_ORIGINS.split(',') : ['http://localhost:3000'],
  credentials: true
}));

app.use(compression());
app.use(morgan('combined', { stream: { write: message => logger.info(message.trim()) } }));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Security event tracking
let securityEvents = [];
let anomalyCount = 0;
let alertsSent = 0;

// Service endpoints to monitor
const MONITORED_SERVICES = {
  jellyfin: 'http://jellyfin:8096/health',
  sonarr: 'http://sonarr:8989/ping',
  radarr: 'http://radarr:7878/ping',
  prowlarr: 'http://prowlarr:9696/ping',
  qbittorrent: 'http://qbittorrent:8080/api/v2/app/version',
  grafana: 'http://grafana:3000/api/health'
};

/**
 * Analyze security logs for anomalies
 */
async function analyzeSecurityLogs() {
  try {
    const logFiles = [
      '/var/log/monitor/nginx-access.log',
      '/var/log/monitor/nginx-error.log',
      '/var/log/monitor/fail2ban.log'
    ];

    for (const logFile of logFiles) {
      try {
        const logContent = await fs.readFile(logFile, 'utf8');
        const lines = logContent.split('\n').filter(line => line.trim());
        
        // Analyze recent entries (last 1000 lines)
        const recentLines = lines.slice(-1000);
        await analyzeLogEntries(recentLines, logFile);
      } catch (error) {
        logger.warn(`Could not read log file ${logFile}: ${error.message}`);
      }
    }
  } catch (error) {
    logger.error('Error analyzing security logs:', error);
  }
}

/**
 * Analyze individual log entries for security threats
 */
async function analyzeLogEntries(lines, logFile) {
  const suspiciousPatterns = [
    /(\d+\.\d+\.\d+\.\d+).*"(GET|POST|PUT|DELETE).*".*40[1-4]/,
    /(\d+\.\d+\.\d+\.\d+).*".*".*429/,
    /(\d+\.\d+\.\d+\.\d+).*".*".*50[0-5]/,
    /(\d+\.\d+\.\d+\.\d+).*(sqlmap|nikto|nmap|masscan|zmap)/i,
    /(\d+\.\d+\.\d+\.\d+).*(\.php|\.asp|\.jsp|wp-admin|phpmyadmin)/i,
    /(\d+\.\d+\.\d+\.\d+).*".*".*".*bot.*"/i
  ];

  for (const line of lines) {
    for (const pattern of suspiciousPatterns) {
      const match = line.match(pattern);
      if (match) {
        const ip = match[1];
        await recordSecurityEvent({
          type: 'suspicious_activity',
          ip,
          pattern: pattern.source,
          logFile,
          timestamp: new Date().toISOString(),
          details: line.substring(0, 500) // Limit line length
        });
      }
    }
  }
}

/**
 * Record security events and trigger alerts if necessary
 */
async function recordSecurityEvent(event) {
  securityEvents.push(event);
  
  // Keep only last 10000 events
  if (securityEvents.length > 10000) {
    securityEvents = securityEvents.slice(-10000);
  }

  // Enrich event with geolocation data
  if (event.ip) {
    const geoData = geoip.lookup(event.ip);
    if (geoData) {
      event.location = {
        country: geoData.country,
        region: geoData.region,
        city: geoData.city,
        timezone: geoData.timezone
      };
    }
  }

  logger.warn('Security event recorded:', event);

  // Check if this IP has multiple recent events
  const recentEvents = securityEvents.filter(e => 
    e.ip === event.ip && 
    new Date(e.timestamp) > new Date(Date.now() - 3600000) // Last hour
  );

  if (recentEvents.length >= 5) {
    await sendSecurityAlert(event, recentEvents.length);
  }
}

/**
 * Send security alert via email and/or Slack
 */
async function sendSecurityAlert(event, eventCount) {
  try {
    const alertMessage = `
Security Alert: Suspicious Activity Detected

IP Address: ${event.ip}
Location: ${event.location ? `${event.location.city}, ${event.location.country}` : 'Unknown'}
Event Type: ${event.type}
Event Count (last hour): ${eventCount}
Pattern Matched: ${event.pattern}
Timestamp: ${event.timestamp}

Details: ${event.details}

Please review your security logs and consider blocking this IP if the activity continues.
    `;

    // Send email alert
    if (process.env.SECURITY_ALERT_EMAIL) {
      await emailTransporter.sendMail({
        from: process.env.SMTP_USER,
        to: process.env.SECURITY_ALERT_EMAIL,
        subject: `Security Alert: Suspicious Activity from ${event.ip}`,
        text: alertMessage
      });
    }

    // Send Slack alert
    if (process.env.SECURITY_SLACK_WEBHOOK) {
      await axios.post(process.env.SECURITY_SLACK_WEBHOOK, {
        text: `🚨 Security Alert: Suspicious activity detected from IP ${event.ip}`,
        attachments: [{
          color: 'danger',
          fields: [
            { title: 'IP Address', value: event.ip, short: true },
            { title: 'Location', value: event.location ? `${event.location.city}, ${event.location.country}` : 'Unknown', short: true },
            { title: 'Event Count', value: eventCount.toString(), short: true },
            { title: 'Type', value: event.type, short: true }
          ]
        }]
      });
    }

    alertsSent++;
    logger.info(`Security alert sent for IP ${event.ip}`);
  } catch (error) {
    logger.error('Error sending security alert:', error);
  }
}

/**
 * Monitor service health and detect anomalies
 */
async function monitorServiceHealth() {
  const healthResults = {};

  for (const [service, url] of Object.entries(MONITORED_SERVICES)) {
    try {
      const startTime = Date.now();
      const response = await axios.get(url, { timeout: 5000 });
      const responseTime = Date.now() - startTime;

      healthResults[service] = {
        status: 'healthy',
        responseTime,
        statusCode: response.status,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      healthResults[service] = {
        status: 'unhealthy',
        error: error.message,
        timestamp: new Date().toISOString()
      };

      // Record as security event if service is down
      await recordSecurityEvent({
        type: 'service_unavailable',
        service,
        error: error.message,
        timestamp: new Date().toISOString()
      });
    }
  }

  // Store health results in Redis
  await redis.set('service_health', JSON.stringify(healthResults), 'EX', 300);

  return healthResults;
}

/**
 * Detect rate limiting anomalies
 */
async function detectRateLimitingAnomalies() {
  try {
    const rateLimitData = await redis.get('rate_limit_stats');
    if (rateLimitData) {
      const stats = JSON.parse(rateLimitData);
      
      // Check for unusual spikes in rate limiting
      const totalBlocked = Object.values(stats).reduce((sum, count) => sum + count, 0);
      
      if (totalBlocked > 100) { // Threshold for anomaly
        await recordSecurityEvent({
          type: 'rate_limit_anomaly',
          details: `High number of rate-limited requests: ${totalBlocked}`,
          stats,
          timestamp: new Date().toISOString()
        });
      }
    }
  } catch (error) {
    logger.error('Error detecting rate limiting anomalies:', error);
  }
}

// API Routes

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Security dashboard endpoint
app.get('/dashboard', async (req, res) => {
  try {
    const serviceHealth = await redis.get('service_health');
    const recentEvents = securityEvents.slice(-50);

    res.json({
      serviceHealth: serviceHealth ? JSON.parse(serviceHealth) : {},
      recentSecurityEvents: recentEvents,
      statistics: {
        totalEvents: securityEvents.length,
        alertsSent,
        anomalyCount,
        uptime: process.uptime()
      }
    });
  } catch (error) {
    logger.error('Error generating dashboard data:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Security events API
app.get('/api/events', (req, res) => {
  const limit = parseInt(req.query.limit) || 100;
  const offset = parseInt(req.query.offset) || 0;
  
  const events = securityEvents
    .slice(-limit - offset, -offset || undefined)
    .reverse();

  res.json({
    events,
    total: securityEvents.length,
    limit,
    offset
  });
});

// Manual security scan endpoint
app.post('/api/scan', async (req, res) => {
  try {
    await analyzeSecurityLogs();
    await detectRateLimitingAnomalies();
    
    res.json({ 
      message: 'Security scan completed',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Error during manual security scan:', error);
    res.status(500).json({ error: 'Scan failed' });
  }
});

// Service health endpoint
app.get('/api/health', async (req, res) => {
  try {
    const healthResults = await monitorServiceHealth();
    res.json(healthResults);
  } catch (error) {
    logger.error('Error checking service health:', error);
    res.status(500).json({ error: 'Health check failed' });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  logger.error('Unhandled error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

// Start scheduled monitoring
cron.schedule('*/5 * * * *', () => {
  analyzeSecurityLogs().catch(error => logger.error('Scheduled log analysis failed:', error));
});

cron.schedule('*/2 * * * *', () => {
  monitorServiceHealth().catch(error => logger.error('Scheduled health check failed:', error));
});

cron.schedule('*/10 * * * *', () => {
  detectRateLimitingAnomalies().catch(error => logger.error('Scheduled anomaly detection failed:', error));
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  logger.info(`Security monitoring server running on port ${PORT}`);
  
  // Perform initial checks
  monitorServiceHealth();
  analyzeSecurityLogs();
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  process.exit(0);
});