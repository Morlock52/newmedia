/**
 * Quantum Security Monitoring System
 * Real-time monitoring and threat detection for quantum-resistant systems
 */

const EventEmitter = require('events');
const crypto = require('crypto');
const { promisify } = require('util');

class SecurityMonitor extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.metrics = {
      quantumOperations: new Map(),
      threats: new Map(),
      performance: new Map(),
      errors: new Map()
    };
    
    this.thresholds = {
      maxAuthAttempts: options.maxAuthAttempts || 5,
      maxTokensPerMinute: options.maxTokensPerMinute || 100,
      maxQuantumOpsPerSecond: options.maxQuantumOpsPerSecond || 1000,
      anomalyThreshold: options.anomalyThreshold || 0.85
    };
    
    this.alerts = [];
    this.startTime = Date.now();
    
    this.initializeMonitoring();
  }

  /**
   * Initialize monitoring systems
   */
  initializeMonitoring() {
    // Set up metric collection intervals
    this.metricsInterval = setInterval(() => {
      this.collectMetrics();
    }, 5000); // Every 5 seconds

    // Set up anomaly detection
    this.anomalyInterval = setInterval(() => {
      this.detectAnomalies();
    }, 30000); // Every 30 seconds

    // Set up threat analysis
    this.threatInterval = setInterval(() => {
      this.analyzeThreatPatterns();
    }, 60000); // Every minute

    console.log('Security monitoring initialized');
  }

  /**
   * Record quantum operation
   */
  recordQuantumOperation(operation, details = {}) {
    const timestamp = Date.now();
    const op = {
      type: operation,
      timestamp,
      duration: details.duration || 0,
      success: details.success !== false,
      algorithm: details.algorithm,
      size: details.size,
      ...details
    };

    // Store in time-bucketed map
    const bucket = Math.floor(timestamp / 60000); // 1-minute buckets
    if (!this.metrics.quantumOperations.has(bucket)) {
      this.metrics.quantumOperations.set(bucket, []);
    }
    
    this.metrics.quantumOperations.get(bucket).push(op);

    // Check for immediate threats
    this.checkOperationThreats(op);

    // Emit event for real-time monitoring
    this.emit('quantum-operation', op);
  }

  /**
   * Record authentication attempt
   */
  recordAuthAttempt(userId, success, details = {}) {
    const timestamp = Date.now();
    const attempt = {
      userId,
      success,
      timestamp,
      ip: details.ip,
      method: details.method,
      ...details
    };

    const bucket = Math.floor(timestamp / 60000);
    if (!this.metrics.threats.has(bucket)) {
      this.metrics.threats.set(bucket, { authAttempts: [] });
    }
    
    this.metrics.threats.get(bucket).authAttempts.push(attempt);

    // Check for brute force attempts
    this.checkBruteForce(userId);

    this.emit('auth-attempt', attempt);
  }

  /**
   * Record performance metrics
   */
  recordPerformance(metric, value, unit = 'ms') {
    const timestamp = Date.now();
    const perf = {
      metric,
      value,
      unit,
      timestamp
    };

    const bucket = Math.floor(timestamp / 60000);
    if (!this.metrics.performance.has(bucket)) {
      this.metrics.performance.set(bucket, []);
    }
    
    this.metrics.performance.get(bucket).push(perf);

    // Check for performance degradation
    this.checkPerformanceDegradation(metric, value);

    this.emit('performance-metric', perf);
  }

  /**
   * Record error
   */
  recordError(error, context = {}) {
    const timestamp = Date.now();
    const errorRecord = {
      message: error.message,
      stack: error.stack,
      context,
      timestamp
    };

    const bucket = Math.floor(timestamp / 60000);
    if (!this.metrics.errors.has(bucket)) {
      this.metrics.errors.set(bucket, []);
    }
    
    this.metrics.errors.get(bucket).push(errorRecord);

    // Check for error patterns
    this.checkErrorPatterns(errorRecord);

    this.emit('error', errorRecord);
  }

  /**
   * Check for operation-based threats
   */
  checkOperationThreats(operation) {
    // Check for unusual operation patterns
    const recentOps = this.getRecentOperations(60000); // Last minute
    
    // Check operation rate
    const opsPerSecond = recentOps.length / 60;
    if (opsPerSecond > this.thresholds.maxQuantumOpsPerSecond) {
      this.raiseAlert('HIGH_OPERATION_RATE', {
        current: opsPerSecond,
        threshold: this.thresholds.maxQuantumOpsPerSecond,
        operation: operation.type
      });
    }

    // Check for timing attacks
    if (operation.type === 'ml-dsa-verify' || operation.type === 'ml-kem-decapsulate') {
      const similarOps = recentOps.filter(op => op.type === operation.type);
      if (similarOps.length > 10) {
        const timings = similarOps.map(op => op.duration);
        const avgTiming = timings.reduce((a, b) => a + b, 0) / timings.length;
        const variance = this.calculateVariance(timings, avgTiming);
        
        if (variance < 0.1) { // Very consistent timing might indicate attack
          this.raiseAlert('POSSIBLE_TIMING_ATTACK', {
            operation: operation.type,
            avgTiming,
            variance
          });
        }
      }
    }
  }

  /**
   * Check for brute force attacks
   */
  checkBruteForce(userId) {
    const recentAttempts = this.getRecentAuthAttempts(300000); // Last 5 minutes
    const userAttempts = recentAttempts.filter(a => a.userId === userId);
    const failedAttempts = userAttempts.filter(a => !a.success);

    if (failedAttempts.length >= this.thresholds.maxAuthAttempts) {
      this.raiseAlert('BRUTE_FORCE_DETECTED', {
        userId,
        attempts: failedAttempts.length,
        timeWindow: '5 minutes'
      });
    }

    // Check for distributed attacks
    const uniqueIPs = new Set(recentAttempts.map(a => a.ip)).size;
    if (uniqueIPs > 50 && failedAttempts.length > 100) {
      this.raiseAlert('DISTRIBUTED_ATTACK', {
        uniqueIPs,
        totalAttempts: recentAttempts.length,
        failedAttempts: failedAttempts.length
      });
    }
  }

  /**
   * Check for performance degradation
   */
  checkPerformanceDegradation(metric, value) {
    const historicalValues = this.getHistoricalPerformance(metric, 3600000); // Last hour
    
    if (historicalValues.length > 10) {
      const avg = historicalValues.reduce((a, b) => a + b.value, 0) / historicalValues.length;
      const stdDev = Math.sqrt(this.calculateVariance(historicalValues.map(h => h.value), avg));
      
      // Alert if current value is 3 standard deviations above average
      if (value > avg + (3 * stdDev)) {
        this.raiseAlert('PERFORMANCE_DEGRADATION', {
          metric,
          currentValue: value,
          avgValue: avg,
          stdDev
        });
      }
    }
  }

  /**
   * Check for error patterns
   */
  checkErrorPatterns(error) {
    const recentErrors = this.getRecentErrors(300000); // Last 5 minutes
    
    // Check for error rate spike
    if (recentErrors.length > 50) {
      this.raiseAlert('HIGH_ERROR_RATE', {
        errorCount: recentErrors.length,
        timeWindow: '5 minutes',
        commonErrors: this.findCommonErrors(recentErrors)
      });
    }

    // Check for specific quantum-related errors
    if (error.message.includes('quantum') || error.message.includes('signature')) {
      const quantumErrors = recentErrors.filter(e => 
        e.message.includes('quantum') || e.message.includes('signature')
      );
      
      if (quantumErrors.length > 10) {
        this.raiseAlert('QUANTUM_SYSTEM_FAILURE', {
          errorCount: quantumErrors.length,
          errors: quantumErrors.slice(0, 5)
        });
      }
    }
  }

  /**
   * Detect anomalies using statistical analysis
   */
  detectAnomalies() {
    // Analyze quantum operation patterns
    const operations = this.getRecentOperations(300000); // Last 5 minutes
    
    if (operations.length > 100) {
      // Group by operation type
      const opTypes = {};
      operations.forEach(op => {
        opTypes[op.type] = (opTypes[op.type] || 0) + 1;
      });

      // Check for unusual distribution
      const totalOps = operations.length;
      Object.entries(opTypes).forEach(([type, count]) => {
        const ratio = count / totalOps;
        
        // Alert if one operation type dominates (potential attack)
        if (ratio > this.thresholds.anomalyThreshold) {
          this.raiseAlert('ANOMALOUS_OPERATION_PATTERN', {
            operationType: type,
            ratio,
            count,
            totalOperations: totalOps
          });
        }
      });
    }

    // Analyze timing patterns
    const timingData = operations.map(op => ({
      type: op.type,
      duration: op.duration,
      timestamp: op.timestamp
    }));

    this.analyzeTimingAnomalies(timingData);
  }

  /**
   * Analyze timing anomalies
   */
  analyzeTimingAnomalies(timingData) {
    // Group by operation type
    const timingByType = {};
    
    timingData.forEach(data => {
      if (!timingByType[data.type]) {
        timingByType[data.type] = [];
      }
      timingByType[data.type].push(data.duration);
    });

    // Check each operation type
    Object.entries(timingByType).forEach(([type, durations]) => {
      if (durations.length > 20) {
        const avg = durations.reduce((a, b) => a + b, 0) / durations.length;
        const variance = this.calculateVariance(durations, avg);
        const stdDev = Math.sqrt(variance);
        
        // Detect outliers
        const outliers = durations.filter(d => Math.abs(d - avg) > 3 * stdDev);
        
        if (outliers.length > durations.length * 0.1) { // More than 10% outliers
          this.raiseAlert('TIMING_ANOMALY', {
            operationType: type,
            outlierRatio: outliers.length / durations.length,
            avgDuration: avg,
            stdDev
          });
        }
      }
    });
  }

  /**
   * Analyze threat patterns
   */
  analyzeThreatPatterns() {
    const threats = this.getRecentThreats(3600000); // Last hour
    
    if (threats.length === 0) return;

    // Analyze threat types
    const threatTypes = {};
    threats.forEach(threat => {
      threatTypes[threat.type] = (threatTypes[threat.type] || 0) + 1;
    });

    // Generate threat report
    const report = {
      timestamp: Date.now(),
      timeWindow: '1 hour',
      totalThreats: threats.length,
      threatTypes,
      severityBreakdown: this.analyzeSeverity(threats),
      recommendations: this.generateRecommendations(threatTypes)
    };

    this.emit('threat-report', report);

    // Store report
    this.lastThreatReport = report;
  }

  /**
   * Raise security alert
   */
  raiseAlert(type, details = {}) {
    const alert = {
      id: crypto.randomUUID(),
      type,
      severity: this.determineSeverity(type),
      timestamp: Date.now(),
      details,
      resolved: false
    };

    this.alerts.push(alert);
    
    // Emit alert for real-time handling
    this.emit('security-alert', alert);

    // Log alert
    console.error(`SECURITY ALERT [${alert.severity}]: ${type}`, details);

    // Auto-remediate if possible
    this.attemptAutoRemediation(alert);
  }

  /**
   * Attempt automatic remediation
   */
  attemptAutoRemediation(alert) {
    switch (alert.type) {
      case 'BRUTE_FORCE_DETECTED':
        // Block user temporarily
        this.emit('block-user', {
          userId: alert.details.userId,
          duration: 3600000 // 1 hour
        });
        break;
        
      case 'HIGH_OPERATION_RATE':
        // Apply rate limiting
        this.emit('apply-rate-limit', {
          operation: alert.details.operation,
          limit: this.thresholds.maxQuantumOpsPerSecond * 0.8
        });
        break;
        
      case 'DISTRIBUTED_ATTACK':
        // Enable enhanced security mode
        this.emit('enable-enhanced-security');
        break;
    }
  }

  /**
   * Get security report
   */
  getSecurityReport() {
    const uptime = Date.now() - this.startTime;
    const recentOps = this.getRecentOperations(3600000);
    const recentThreats = this.getRecentThreats(3600000);
    const activeAlerts = this.alerts.filter(a => !a.resolved);

    return {
      status: activeAlerts.length === 0 ? 'SECURE' : 'THREATS_DETECTED',
      uptime,
      metrics: {
        totalOperations: recentOps.length,
        operationsPerSecond: recentOps.length / 3600,
        activeThreats: activeAlerts.length,
        resolvedThreats: this.alerts.filter(a => a.resolved).length,
        errorRate: this.getRecentErrors(3600000).length / 3600
      },
      activeAlerts,
      lastThreatReport: this.lastThreatReport,
      recommendations: this.generateSystemRecommendations()
    };
  }

  /**
   * Helper methods
   */

  getRecentOperations(timeWindow) {
    const cutoff = Date.now() - timeWindow;
    const operations = [];
    
    for (const [bucket, ops] of this.metrics.quantumOperations.entries()) {
      if (bucket * 60000 > cutoff) {
        operations.push(...ops.filter(op => op.timestamp > cutoff));
      }
    }
    
    return operations;
  }

  getRecentAuthAttempts(timeWindow) {
    const cutoff = Date.now() - timeWindow;
    const attempts = [];
    
    for (const [bucket, threats] of this.metrics.threats.entries()) {
      if (bucket * 60000 > cutoff && threats.authAttempts) {
        attempts.push(...threats.authAttempts.filter(a => a.timestamp > cutoff));
      }
    }
    
    return attempts;
  }

  getRecentErrors(timeWindow) {
    const cutoff = Date.now() - timeWindow;
    const errors = [];
    
    for (const [bucket, errs] of this.metrics.errors.entries()) {
      if (bucket * 60000 > cutoff) {
        errors.push(...errs.filter(e => e.timestamp > cutoff));
      }
    }
    
    return errors;
  }

  getRecentThreats(timeWindow) {
    return this.alerts.filter(a => a.timestamp > Date.now() - timeWindow);
  }

  getHistoricalPerformance(metric, timeWindow) {
    const cutoff = Date.now() - timeWindow;
    const values = [];
    
    for (const [bucket, perfs] of this.metrics.performance.entries()) {
      if (bucket * 60000 > cutoff) {
        values.push(...perfs.filter(p => p.metric === metric && p.timestamp > cutoff));
      }
    }
    
    return values;
  }

  calculateVariance(values, mean) {
    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
    return squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
  }

  findCommonErrors(errors) {
    const errorCounts = {};
    errors.forEach(e => {
      const key = e.message.split('\n')[0]; // First line of error
      errorCounts[key] = (errorCounts[key] || 0) + 1;
    });
    
    return Object.entries(errorCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([error, count]) => ({ error, count }));
  }

  determineSeverity(alertType) {
    const severityMap = {
      'BRUTE_FORCE_DETECTED': 'HIGH',
      'DISTRIBUTED_ATTACK': 'CRITICAL',
      'QUANTUM_SYSTEM_FAILURE': 'CRITICAL',
      'HIGH_OPERATION_RATE': 'MEDIUM',
      'TIMING_ANOMALY': 'HIGH',
      'POSSIBLE_TIMING_ATTACK': 'HIGH',
      'PERFORMANCE_DEGRADATION': 'MEDIUM',
      'HIGH_ERROR_RATE': 'HIGH',
      'ANOMALOUS_OPERATION_PATTERN': 'MEDIUM'
    };
    
    return severityMap[alertType] || 'LOW';
  }

  analyzeSeverity(threats) {
    const severity = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
    threats.forEach(threat => {
      severity[threat.severity]++;
    });
    return severity;
  }

  generateRecommendations(threatTypes) {
    const recommendations = [];
    
    if (threatTypes['BRUTE_FORCE_DETECTED'] > 5) {
      recommendations.push('Implement stronger rate limiting on authentication endpoints');
    }
    
    if (threatTypes['TIMING_ANOMALY'] > 3) {
      recommendations.push('Review quantum operation implementations for timing attack vulnerabilities');
    }
    
    if (threatTypes['HIGH_OPERATION_RATE'] > 10) {
      recommendations.push('Scale quantum processing infrastructure or implement request queuing');
    }
    
    return recommendations;
  }

  generateSystemRecommendations() {
    const recommendations = [];
    const report = this.getSecurityReport();
    
    if (report.metrics.operationsPerSecond > this.thresholds.maxQuantumOpsPerSecond * 0.8) {
      recommendations.push({
        type: 'SCALING',
        message: 'Quantum operation rate approaching threshold. Consider scaling infrastructure.',
        priority: 'HIGH'
      });
    }
    
    if (report.activeAlerts.filter(a => a.severity === 'CRITICAL').length > 0) {
      recommendations.push({
        type: 'SECURITY',
        message: 'Critical security alerts active. Immediate investigation required.',
        priority: 'CRITICAL'
      });
    }
    
    if (report.metrics.errorRate > 0.01) { // More than 1% error rate
      recommendations.push({
        type: 'RELIABILITY',
        message: 'High error rate detected. Review system logs and quantum operation implementations.',
        priority: 'MEDIUM'
      });
    }
    
    return recommendations;
  }

  /**
   * Cleanup and shutdown
   */
  shutdown() {
    clearInterval(this.metricsInterval);
    clearInterval(this.anomalyInterval);
    clearInterval(this.threatInterval);
    
    // Clean up old metrics
    const cutoff = Date.now() - 3600000; // Keep last hour
    for (const [bucket] of this.metrics.quantumOperations.entries()) {
      if (bucket * 60000 < cutoff) {
        this.metrics.quantumOperations.delete(bucket);
      }
    }
    
    console.log('Security monitoring shutdown complete');
  }
}

module.exports = SecurityMonitor;