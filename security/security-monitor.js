const EventEmitter = require('events');
const winston = require('winston');
const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');

// Security monitoring logger
const monitorLogger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: './logs/security-monitor.log' }),
    new winston.transports.File({ filename: './logs/security-incidents.log', level: 'warn' }),
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
});

class SecurityMonitor extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.options = {
      alertThresholds: {
        failedLoginAttempts: 5,
        requestsPerMinute: 1000,
        errorRate: 0.1, // 10%
        responseTimeMs: 5000
      },
      monitoringInterval: options.monitoringInterval || 60000, // 1 minute
      retentionPeriod: options.retentionPeriod || 30 * 24 * 60 * 60 * 1000, // 30 days
      anomalyDetection: options.anomalyDetection !== false,
      realTimeAlerts: options.realTimeAlerts !== false,
      ...options
    };

    // Monitoring data storage
    this.metrics = {
      requests: new Map(),
      errors: new Map(),
      loginAttempts: new Map(),
      securityEvents: [],
      systemMetrics: new Map()
    };

    // Anomaly detection baselines
    this.baselines = {
      requestRate: { mean: 0, stddev: 0, samples: [] },
      errorRate: { mean: 0, stddev: 0, samples: [] },
      responseTime: { mean: 0, stddev: 0, samples: [] }
    };

    // Active incidents
    this.activeIncidents = new Map();
    this.incidentCounter = 0;

    // Start monitoring
    this.startMonitoring();
  }

  // Real-time request monitoring
  monitorRequest(req, res, next) {
    const startTime = Date.now();
    const requestId = crypto.randomUUID();
    
    const requestData = {
      id: requestId,
      method: req.method,
      path: req.path,
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      timestamp: new Date(),
      user: req.user?.userId || 'anonymous'
    };

    // Track request
    this.recordRequest(requestData);

    // Monitor response
    res.on('finish', () => {
      const responseTime = Date.now() - startTime;
      
      const responseData = {
        ...requestData,
        statusCode: res.statusCode,
        responseTime,
        success: res.statusCode < 400
      };

      this.recordResponse(responseData);
      
      // Check for security events
      this.analyzeRequest(responseData);
    });

    next();
  }

  recordRequest(requestData) {
    const minute = Math.floor(Date.now() / 60000);
    const current = this.metrics.requests.get(minute) || {
      count: 0,
      ips: new Set(),
      paths: new Map(),
      methods: new Map()
    };

    current.count++;
    current.ips.add(requestData.ip);
    current.paths.set(requestData.path, (current.paths.get(requestData.path) || 0) + 1);
    current.methods.set(requestData.method, (current.methods.get(requestData.method) || 0) + 1);

    this.metrics.requests.set(minute, current);
  }

  recordResponse(responseData) {
    const minute = Math.floor(Date.now() / 60000);
    
    if (!responseData.success) {
      const errorData = this.metrics.errors.get(minute) || {
        count: 0,
        by_status: new Map(),
        by_ip: new Map(),
        by_path: new Map()
      };

      errorData.count++;
      errorData.by_status.set(responseData.statusCode, 
        (errorData.by_status.get(responseData.statusCode) || 0) + 1);
      errorData.by_ip.set(responseData.ip, 
        (errorData.by_ip.get(responseData.ip) || 0) + 1);
      errorData.by_path.set(responseData.path, 
        (errorData.by_path.get(responseData.path) || 0) + 1);

      this.metrics.errors.set(minute, errorData);
    }

    // Update response time baseline
    this.updateBaseline('responseTime', responseData.responseTime);
  }

  // Security event analysis
  analyzeRequest(responseData) {
    const securityChecks = [
      () => this.checkBruteForce(responseData),
      () => this.checkAnomalousActivity(responseData),
      () => this.checkSuspiciousPatterns(responseData),
      () => this.checkRateLimits(responseData),
      () => this.checkErrorRates(responseData)
    ];

    for (const check of securityChecks) {
      try {
        check();
      } catch (error) {
        monitorLogger.error('Security check failed', {
          error: error.message,
          requestId: responseData.id
        });
      }
    }
  }

  checkBruteForce(responseData) {
    if (responseData.statusCode === 401 || responseData.statusCode === 403) {
      const key = `${responseData.ip}:${responseData.path}`;
      const attempts = this.metrics.loginAttempts.get(key) || { count: 0, firstAttempt: Date.now() };
      
      attempts.count++;
      this.metrics.loginAttempts.set(key, attempts);

      if (attempts.count >= this.options.alertThresholds.failedLoginAttempts) {
        this.createSecurityIncident('BRUTE_FORCE_ATTACK', {
          severity: 'HIGH',
          description: `Multiple failed login attempts from ${responseData.ip}`,
          details: {
            ip: responseData.ip,
            path: responseData.path,
            attempts: attempts.count,
            timespan: Date.now() - attempts.firstAttempt
          },
          recommendations: [
            'Block IP address temporarily',
            'Enable CAPTCHA for this IP',
            'Notify user if account exists',
            'Monitor for continued attempts'
          ]
        });
      }
    }
  }

  checkAnomalousActivity(responseData) {
    if (!this.options.anomalyDetection) return;

    // Check if response time is anomalous
    const responseTimeAnomaly = this.isAnomalous('responseTime', responseData.responseTime);
    if (responseTimeAnomaly.isAnomaly && responseTimeAnomaly.severity === 'HIGH') {
      this.createSecurityIncident('PERFORMANCE_ANOMALY', {
        severity: 'MEDIUM',
        description: `Unusual response time detected: ${responseData.responseTime}ms`,
        details: {
          responseTime: responseData.responseTime,
          baseline: this.baselines.responseTime.mean,
          path: responseData.path,
          ip: responseData.ip
        }
      });
    }
  }

  checkSuspiciousPatterns(responseData) {
    const suspiciousPatterns = [
      {
        pattern: /\.(php|asp|jsp)$/i,
        severity: 'MEDIUM',
        description: 'Request for non-existent script files'
      },
      {
        pattern: /(admin|wp-admin|phpmyadmin)/i,
        severity: 'MEDIUM',
        description: 'Administrative path probe'
      },
      {
        pattern: /\.\./,
        severity: 'HIGH',
        description: 'Path traversal attempt'
      },
      {
        pattern: /(union|select|insert|delete|drop|script)/i,
        severity: 'HIGH',
        description: 'SQL injection or XSS attempt'
      },
      {
        pattern: /\/\.env|\/config|\/backup/i,
        severity: 'HIGH',
        description: 'Sensitive file access attempt'
      }
    ];

    for (const pattern of suspiciousPatterns) {
      if (pattern.pattern.test(responseData.path)) {
        this.createSecurityIncident('SUSPICIOUS_REQUEST', {
          severity: pattern.severity,
          description: pattern.description,
          details: {
            path: responseData.path,
            ip: responseData.ip,
            userAgent: responseData.userAgent,
            pattern: pattern.pattern.toString()
          },
          recommendations: [
            'Investigate source IP',
            'Check for similar patterns',
            'Consider blocking IP if pattern continues'
          ]
        });
        break;
      }
    }
  }

  checkRateLimits(responseData) {
    const minute = Math.floor(Date.now() / 60000);
    const requestData = this.metrics.requests.get(minute);
    
    if (requestData && requestData.count > this.options.alertThresholds.requestsPerMinute) {
      this.createSecurityIncident('RATE_LIMIT_EXCEEDED', {
        severity: 'MEDIUM',
        description: `High request rate detected: ${requestData.count} requests/minute`,
        details: {
          requestsPerMinute: requestData.count,
          threshold: this.options.alertThresholds.requestsPerMinute,
          uniqueIPs: requestData.ips.size,
          topPaths: Array.from(requestData.paths.entries())
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5)
        }
      });
    }
  }

  checkErrorRates(responseData) {
    const minute = Math.floor(Date.now() / 60000);
    const errorData = this.metrics.errors.get(minute);
    const requestData = this.metrics.requests.get(minute);
    
    if (errorData && requestData) {
      const errorRate = errorData.count / requestData.count;
      
      if (errorRate > this.options.alertThresholds.errorRate) {
        this.createSecurityIncident('HIGH_ERROR_RATE', {
          severity: 'MEDIUM',
          description: `High error rate detected: ${(errorRate * 100).toFixed(1)}%`,
          details: {
            errorRate: errorRate,
            errorCount: errorData.count,
            totalRequests: requestData.count,
            topErrors: Array.from(errorData.by_status.entries())
              .sort(([,a], [,b]) => b - a)
              .slice(0, 5)
          }
        });
      }
    }
  }

  // Anomaly detection
  updateBaseline(metric, value) {
    const baseline = this.baselines[metric];
    baseline.samples.push(value);
    
    // Keep only recent samples (sliding window)
    if (baseline.samples.length > 1000) {
      baseline.samples = baseline.samples.slice(-1000);
    }
    
    // Update statistical measures
    baseline.mean = baseline.samples.reduce((a, b) => a + b) / baseline.samples.length;
    const variance = baseline.samples.reduce((sum, val) => sum + Math.pow(val - baseline.mean, 2), 0) / baseline.samples.length;
    baseline.stddev = Math.sqrt(variance);
  }

  isAnomalous(metric, value) {
    const baseline = this.baselines[metric];
    
    if (baseline.samples.length < 10) {
      return { isAnomaly: false, severity: 'LOW' };
    }
    
    const zscore = Math.abs((value - baseline.mean) / baseline.stddev);
    
    if (zscore > 3) {
      return { isAnomaly: true, severity: 'HIGH', zscore };
    } else if (zscore > 2) {
      return { isAnomaly: true, severity: 'MEDIUM', zscore };
    } else if (zscore > 1.5) {
      return { isAnomaly: true, severity: 'LOW', zscore };
    }
    
    return { isAnomaly: false, severity: 'LOW', zscore };
  }

  // Incident management
  createSecurityIncident(type, details) {
    const incidentId = `INC-${Date.now()}-${++this.incidentCounter}`;
    
    const incident = {
      id: incidentId,
      type,
      severity: details.severity,
      description: details.description,
      details: details.details || {},
      recommendations: details.recommendations || [],
      timestamp: new Date(),
      status: 'OPEN',
      actions: []
    };

    // Check if similar incident already exists
    const existingSimilar = Array.from(this.activeIncidents.values())
      .find(inc => inc.type === type && inc.status === 'OPEN' && 
                   this.areSimilarIncidents(inc, incident));

    if (existingSimilar) {
      // Update existing incident
      existingSimilar.details.count = (existingSimilar.details.count || 1) + 1;
      existingSimilar.details.lastOccurrence = new Date();
      
      monitorLogger.warn('Security incident updated', {
        incidentId: existingSimilar.id,
        type,
        count: existingSimilar.details.count
      });
    } else {
      // Create new incident
      this.activeIncidents.set(incidentId, incident);
      
      monitorLogger.error('Security incident created', {
        incidentId,
        type,
        severity: details.severity,
        description: details.description
      });

      // Emit event for external handlers
      this.emit('securityIncident', incident);

      // Auto-response for certain incident types
      this.triggerAutoResponse(incident);
    }

    return incident;
  }

  areSimilarIncidents(incident1, incident2) {
    // Simple similarity check - can be enhanced
    return incident1.type === incident2.type &&
           JSON.stringify(incident1.details?.ip) === JSON.stringify(incident2.details?.ip);
  }

  triggerAutoResponse(incident) {
    const autoResponses = {
      'BRUTE_FORCE_ATTACK': () => this.handleBruteForceAttack(incident),
      'RATE_LIMIT_EXCEEDED': () => this.handleRateLimitExceeded(incident),
      'SUSPICIOUS_REQUEST': () => this.handleSuspiciousRequest(incident)
    };

    const handler = autoResponses[incident.type];
    if (handler && this.options.autoResponse) {
      try {
        handler();
        incident.actions.push({
          type: 'AUTO_RESPONSE',
          timestamp: new Date(),
          description: `Automated response triggered for ${incident.type}`
        });
      } catch (error) {
        monitorLogger.error('Auto-response failed', {
          incidentId: incident.id,
          error: error.message
        });
      }
    }
  }

  handleBruteForceAttack(incident) {
    const ip = incident.details.ip;
    
    // In a real implementation, this would:
    // - Add IP to temporary block list
    // - Increase rate limiting for this IP
    // - Send alert to security team
    
    monitorLogger.warn('Auto-blocking IP for brute force attack', {
      ip,
      incidentId: incident.id
    });
  }

  handleRateLimitExceeded(incident) {
    // In a real implementation, this would:
    // - Temporarily reduce rate limits
    // - Enable additional protection measures
    // - Scale up resources if needed
    
    monitorLogger.warn('Enhanced rate limiting activated', {
      incidentId: incident.id
    });
  }

  handleSuspiciousRequest(incident) {
    const ip = incident.details.ip;
    
    // In a real implementation, this would:
    // - Flag IP for enhanced monitoring
    // - Log additional request details
    // - Potentially challenge future requests
    
    monitorLogger.warn('Enhanced monitoring activated for suspicious IP', {
      ip,
      incidentId: incident.id
    });
  }

  // System monitoring
  startMonitoring() {
    monitorLogger.info('Starting security monitoring', {
      interval: this.options.monitoringInterval,
      anomalyDetection: this.options.anomalyDetection,
      realTimeAlerts: this.options.realTimeAlerts
    });

    // Clean up old data periodically
    setInterval(() => {
      this.cleanupOldData();
    }, this.options.monitoringInterval);

    // Generate periodic reports
    setInterval(() => {
      this.generateSecurityReport();
    }, 15 * 60 * 1000); // Every 15 minutes
  }

  cleanupOldData() {
    const cutoff = Date.now() - this.options.retentionPeriod;
    const cutoffMinute = Math.floor(cutoff / 60000);

    // Clean up metrics
    for (const [minute] of this.metrics.requests) {
      if (minute < cutoffMinute) {
        this.metrics.requests.delete(minute);
      }
    }

    for (const [minute] of this.metrics.errors) {
      if (minute < cutoffMinute) {
        this.metrics.errors.delete(minute);
      }
    }

    // Clean up login attempts
    for (const [key, attempts] of this.metrics.loginAttempts) {
      if (attempts.firstAttempt < cutoff) {
        this.metrics.loginAttempts.delete(key);
      }
    }

    // Close old incidents
    for (const [id, incident] of this.activeIncidents) {
      if (incident.timestamp.getTime() < cutoff && incident.status === 'OPEN') {
        incident.status = 'AUTO_CLOSED';
        incident.actions.push({
          type: 'AUTO_CLOSE',
          timestamp: new Date(),
          description: 'Automatically closed due to age'
        });
        
        this.activeIncidents.delete(id);
      }
    }
  }

  generateSecurityReport() {
    const now = Date.now();
    const currentMinute = Math.floor(now / 60000);
    const last15Minutes = Array.from({ length: 15 }, (_, i) => currentMinute - i);
    
    // Aggregate metrics
    const totalRequests = last15Minutes.reduce((sum, minute) => {
      const data = this.metrics.requests.get(minute);
      return sum + (data?.count || 0);
    }, 0);

    const totalErrors = last15Minutes.reduce((sum, minute) => {
      const data = this.metrics.errors.get(minute);
      return sum + (data?.count || 0);
    }, 0);

    const errorRate = totalRequests > 0 ? totalErrors / totalRequests : 0;
    const uniqueIPs = new Set();
    
    last15Minutes.forEach(minute => {
      const data = this.metrics.requests.get(minute);
      if (data) {
        data.ips.forEach(ip => uniqueIPs.add(ip));
      }
    });

    const report = {
      timestamp: new Date(),
      period: '15 minutes',
      metrics: {
        totalRequests,
        totalErrors,
        errorRate: Math.round(errorRate * 100 * 100) / 100, // Round to 2 decimal places
        uniqueIPs: uniqueIPs.size,
        activeIncidents: this.activeIncidents.size
      },
      topIPs: this.getTopIPs(last15Minutes),
      recentIncidents: Array.from(this.activeIncidents.values())
        .filter(inc => now - inc.timestamp.getTime() < 15 * 60 * 1000)
        .map(inc => ({
          id: inc.id,
          type: inc.type,
          severity: inc.severity,
          description: inc.description
        }))
    };

    monitorLogger.info('Security monitoring report', report);

    // Save detailed report
    this.saveSecurityReport(report);

    return report;
  }

  getTopIPs(minutes) {
    const ipCounts = new Map();
    
    minutes.forEach(minute => {
      const data = this.metrics.requests.get(minute);
      if (data) {
        data.ips.forEach(ip => {
          ipCounts.set(ip, (ipCounts.get(ip) || 0) + 1);
        });
      }
    });

    return Array.from(ipCounts.entries())
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .map(([ip, count]) => ({ ip, requests: count }));
  }

  async saveSecurityReport(report) {
    try {
      const filename = `security-report-${Date.now()}.json`;
      const filepath = path.join('./logs', filename);
      
      await fs.writeFile(filepath, JSON.stringify(report, null, 2));
      
      monitorLogger.debug('Security report saved', { filename });
    } catch (error) {
      monitorLogger.error('Failed to save security report', {
        error: error.message
      });
    }
  }

  // Public API
  getMetrics() {
    return {
      requests: Object.fromEntries(this.metrics.requests),
      errors: Object.fromEntries(this.metrics.errors),
      loginAttempts: Object.fromEntries(this.metrics.loginAttempts),
      activeIncidents: Object.fromEntries(this.activeIncidents),
      baselines: this.baselines
    };
  }

  getIncidents(status = null) {
    let incidents = Array.from(this.activeIncidents.values());
    
    if (status) {
      incidents = incidents.filter(inc => inc.status === status);
    }
    
    return incidents.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  resolveIncident(incidentId, resolution) {
    const incident = this.activeIncidents.get(incidentId);
    
    if (incident) {
      incident.status = 'RESOLVED';
      incident.resolution = resolution;
      incident.resolvedAt = new Date();
      incident.actions.push({
        type: 'MANUAL_RESOLVE',
        timestamp: new Date(),
        description: resolution
      });

      monitorLogger.info('Security incident resolved', {
        incidentId,
        resolution
      });

      return incident;
    }
    
    return null;
  }
}

module.exports = SecurityMonitor;