#!/usr/bin/env python3
"""
Intelligent Alerting System
===========================

Advanced alerting system with ML-based anomaly detection, smart correlation,
escalation management, and multi-channel notifications for comprehensive
monitoring infrastructure.
"""

import asyncio
import json
import logging
import time
import smtplib
import aiohttp
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import sqlite3
import yaml
import statistics
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, deque
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Individual alert"""
    id: str
    timestamp: float
    source: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str  # critical, high, medium, low, info
    status: str  # active, acknowledged, resolved, suppressed
    title: str
    description: str
    tags: Dict[str, str]
    correlation_group: Optional[str] = None
    escalation_level: int = 0
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    suppressed_until: Optional[float] = None
    notification_channels: List[str] = None
    context_data: Dict[str, Any] = None

@dataclass
class AlertRule:
    """Alert rule configuration"""
    id: str
    name: str
    metric_pattern: str  # Regex pattern to match metrics
    condition: str  # >, <, ==, !=, contains, etc.
    threshold: float
    severity: str
    duration_seconds: int  # How long condition must persist
    cooldown_seconds: int  # Minimum time between alerts
    enabled: bool = True
    tags: Dict[str, str] = None
    notification_channels: List[str] = None
    escalation_rules: List[Dict[str, Any]] = None
    context_queries: List[str] = None  # Additional queries for context

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    id: str
    name: str
    type: str  # email, slack, discord, webhook, sms
    enabled: bool
    config: Dict[str, Any]
    rate_limit_per_hour: int = 60
    severity_filter: List[str] = None  # Only send these severities

@dataclass
class EscalationPolicy:
    """Escalation policy configuration"""
    id: str
    name: str
    steps: List[Dict[str, Any]]  # [{delay_minutes: 15, channels: [...], conditions: [...]}]
    repeat_count: int = 3
    repeat_interval_minutes: int = 60

class AnomalyDetector:
    """ML-based anomaly detection for metrics"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.models = {}
        self.scalers = {}
        self.training_data = defaultdict(deque)
        self.min_samples = 50
        
    def add_sample(self, metric_name: str, value: float, timestamp: float):
        """Add a metric sample for training"""
        self.training_data[metric_name].append((timestamp, value))
        
        # Keep only recent samples (last 24 hours)
        cutoff_time = timestamp - 86400  # 24 hours
        while (self.training_data[metric_name] and 
               self.training_data[metric_name][0][0] < cutoff_time):
            self.training_data[metric_name].popleft()
            
    def train_model(self, metric_name: str) -> bool:
        """Train anomaly detection model for a metric"""
        if metric_name not in self.training_data:
            return False
            
        data = list(self.training_data[metric_name])
        if len(data) < self.min_samples:
            return False
            
        # Prepare features (value, hour_of_day, day_of_week)
        features = []
        for timestamp, value in data:
            dt = datetime.fromtimestamp(timestamp)
            features.append([
                value,
                dt.hour,
                dt.weekday(),
                timestamp % 3600,  # Second in hour (seasonality)
                timestamp % 86400  # Second in day (daily pattern)
            ])
            
        features = np.array(features)
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Train isolation forest
        model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        model.fit(features_scaled)
        
        # Store model and scaler
        self.models[metric_name] = model
        self.scalers[metric_name] = scaler
        
        logger.info(f"Trained anomaly detection model for {metric_name} with {len(data)} samples")
        return True
        
    def detect_anomaly(self, metric_name: str, value: float, timestamp: float) -> Dict[str, Any]:
        """Detect if a value is anomalous"""
        if metric_name not in self.models:
            return {'is_anomaly': False, 'confidence': 0.0, 'reason': 'No model available'}
            
        model = self.models[metric_name]
        scaler = self.scalers[metric_name]
        
        # Prepare features
        dt = datetime.fromtimestamp(timestamp)
        features = np.array([[
            value,
            dt.hour,
            dt.weekday(),
            timestamp % 3600,
            timestamp % 86400
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Detect anomaly
        prediction = model.predict(features_scaled)[0]
        anomaly_score = model.decision_function(features_scaled)[0]
        
        is_anomaly = prediction == -1
        confidence = abs(anomaly_score)
        
        # Get additional context
        recent_values = [v for _, v in list(self.training_data[metric_name])[-10:]]
        if recent_values:
            recent_mean = statistics.mean(recent_values)
            recent_std = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
            deviation = abs(value - recent_mean) / max(recent_std, 0.001)
        else:
            deviation = 0.0
            
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'anomaly_score': anomaly_score,
            'deviation_from_recent': deviation,
            'reason': f'ML model detected anomaly (score: {anomaly_score:.3f})' if is_anomaly else 'Normal behavior'
        }

class AlertCorrelator:
    """Correlate related alerts to reduce noise"""
    
    def __init__(self):
        self.correlation_rules = [
            self._correlate_by_host,
            self._correlate_by_service,
            self._correlate_by_time_window,
            self._correlate_by_metric_type
        ]
        self.active_groups = {}
        
    def correlate_alert(self, alert: Alert, existing_alerts: List[Alert]) -> Optional[str]:
        """Find correlation group for an alert"""
        for rule in self.correlation_rules:
            group_id = rule(alert, existing_alerts)
            if group_id:
                return group_id
                
        return None
        
    def _correlate_by_host(self, alert: Alert, existing_alerts: List[Alert]) -> Optional[str]:
        """Correlate alerts from the same host"""
        alert_host = alert.tags.get('host')
        if not alert_host:
            return None
            
        for existing in existing_alerts:
            if (existing.status == 'active' and 
                existing.tags.get('host') == alert_host and
                abs(alert.timestamp - existing.timestamp) < 300):  # 5 minutes
                return f"host_{alert_host}"
                
        return None
        
    def _correlate_by_service(self, alert: Alert, existing_alerts: List[Alert]) -> Optional[str]:
        """Correlate alerts from the same service"""
        alert_service = alert.tags.get('service')
        if not alert_service:
            return None
            
        for existing in existing_alerts:
            if (existing.status == 'active' and 
                existing.tags.get('service') == alert_service and
                abs(alert.timestamp - existing.timestamp) < 600):  # 10 minutes
                return f"service_{alert_service}"
                
        return None
        
    def _correlate_by_time_window(self, alert: Alert, existing_alerts: List[Alert]) -> Optional[str]:
        """Correlate alerts within a time window"""
        time_window = 180  # 3 minutes
        
        recent_alerts = [
            a for a in existing_alerts
            if a.status == 'active' and abs(alert.timestamp - a.timestamp) < time_window
        ]
        
        if len(recent_alerts) >= 3:  # Multiple alerts in short time
            return f"burst_{int(alert.timestamp // time_window)}"
            
        return None
        
    def _correlate_by_metric_type(self, alert: Alert, existing_alerts: List[Alert]) -> Optional[str]:
        """Correlate alerts of the same metric type"""
        metric_category = alert.metric_name.split('_')[0]  # e.g., 'cpu' from 'cpu_usage'
        
        for existing in existing_alerts:
            if (existing.status == 'active' and 
                existing.metric_name.startswith(metric_category) and
                abs(alert.timestamp - existing.timestamp) < 900):  # 15 minutes
                return f"metric_type_{metric_category}"
                
        return None

class RateLimiter:
    """Rate limiting for notifications"""
    
    def __init__(self):
        self.channel_counters = defaultdict(deque)
        
    def is_allowed(self, channel_id: str, rate_limit: int) -> bool:
        """Check if notification is allowed within rate limit"""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        # Remove old entries
        while (self.channel_counters[channel_id] and 
               self.channel_counters[channel_id][0] < hour_ago):
            self.channel_counters[channel_id].popleft()
            
        # Check rate limit
        if len(self.channel_counters[channel_id]) >= rate_limit:
            return False
            
        # Add current notification
        self.channel_counters[channel_id].append(current_time)
        return True

class NotificationManager:
    """Manage notifications across multiple channels"""
    
    def __init__(self, channels: List[NotificationChannel]):
        self.channels = {ch.id: ch for ch in channels}
        self.rate_limiter = RateLimiter()
        
    async def send_notification(self, alert: Alert, channel_ids: List[str]) -> Dict[str, bool]:
        """Send notification to specified channels"""
        results = {}
        
        for channel_id in channel_ids:
            if channel_id not in self.channels:
                results[channel_id] = False
                continue
                
            channel = self.channels[channel_id]
            
            # Check if channel is enabled
            if not channel.enabled:
                results[channel_id] = False
                continue
                
            # Check severity filter
            if (channel.severity_filter and 
                alert.severity not in channel.severity_filter):
                results[channel_id] = False
                continue
                
            # Check rate limit
            if not self.rate_limiter.is_allowed(channel_id, channel.rate_limit_per_hour):
                logger.warning(f"Rate limit exceeded for channel {channel_id}")
                results[channel_id] = False
                continue
                
            # Send notification
            try:
                success = await self._send_to_channel(alert, channel)
                results[channel_id] = success
            except Exception as e:
                logger.error(f"Failed to send notification to {channel_id}: {e}")
                results[channel_id] = False
                
        return results
        
    async def _send_to_channel(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send notification to a specific channel"""
        if channel.type == 'email':
            return await self._send_email(alert, channel)
        elif channel.type == 'slack':
            return await self._send_slack(alert, channel)
        elif channel.type == 'discord':
            return await self._send_discord(alert, channel)
        elif channel.type == 'webhook':
            return await self._send_webhook(alert, channel)
        else:
            logger.error(f"Unknown channel type: {channel.type}")
            return False
            
    async def _send_email(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send email notification"""
        try:
            config = channel.config
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = config.get('from_email', 'monitoring@localhost')
            msg['To'] = ', '.join(config.get('to_emails', []))
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.title}"
            
            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(config.get('smtp_server', 'localhost'), config.get('smtp_port', 587))
            
            if config.get('use_tls', True):
                server.starttls()
                
            if config.get('username') and config.get('password'):
                server.login(config['username'], config['password'])
                
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
            
    async def _send_slack(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send Slack notification"""
        try:
            config = channel.config
            webhook_url = config.get('webhook_url')
            
            if not webhook_url:
                logger.error("Slack webhook URL not configured")
                return False
                
            # Create Slack message
            color = {
                'critical': '#FF0000',
                'high': '#FF8C00',
                'medium': '#FFD700',
                'low': '#32CD32',
                'info': '#1E90FF'
            }.get(alert.severity, '#808080')
            
            payload = {
                'channel': config.get('channel', '#alerts'),
                'username': config.get('username', 'Monitoring Bot'),
                'icon_emoji': config.get('icon', ':warning:'),
                'attachments': [{
                    'color': color,
                    'title': alert.title,
                    'text': alert.description,
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity.upper(), 'short': True},
                        {'title': 'Source', 'value': alert.source, 'short': True},
                        {'title': 'Metric', 'value': alert.metric_name, 'short': True},
                        {'title': 'Value', 'value': str(alert.current_value), 'short': True},
                        {'title': 'Threshold', 'value': str(alert.threshold_value), 'short': True},
                        {'title': 'Time', 'value': datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S'), 'short': True}
                    ],
                    'footer': 'Performance Monitoring System',
                    'ts': int(alert.timestamp)
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent for alert {alert.id}")
                        return True
                    else:
                        logger.error(f"Slack notification failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
            
    async def _send_discord(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send Discord notification"""
        try:
            config = channel.config
            webhook_url = config.get('webhook_url')
            
            if not webhook_url:
                logger.error("Discord webhook URL not configured")
                return False
                
            # Create Discord embed
            color = {
                'critical': 0xFF0000,
                'high': 0xFF8C00,
                'medium': 0xFFD700,
                'low': 0x32CD32,
                'info': 0x1E90FF
            }.get(alert.severity, 0x808080)
            
            embed = {
                'title': alert.title,
                'description': alert.description,
                'color': color,
                'fields': [
                    {'name': 'Severity', 'value': alert.severity.upper(), 'inline': True},
                    {'name': 'Source', 'value': alert.source, 'inline': True},
                    {'name': 'Metric', 'value': alert.metric_name, 'inline': True},
                    {'name': 'Current Value', 'value': str(alert.current_value), 'inline': True},
                    {'name': 'Threshold', 'value': str(alert.threshold_value), 'inline': True},
                    {'name': 'Time', 'value': datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S'), 'inline': True}
                ],
                'footer': {'text': 'Performance Monitoring System'},
                'timestamp': datetime.fromtimestamp(alert.timestamp).isoformat()
            }
            
            payload = {
                'username': config.get('username', 'Monitoring Bot'),
                'avatar_url': config.get('avatar_url'),
                'embeds': [embed]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 204:
                        logger.info(f"Discord notification sent for alert {alert.id}")
                        return True
                    else:
                        logger.error(f"Discord notification failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False
            
    async def _send_webhook(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send webhook notification"""
        try:
            config = channel.config
            webhook_url = config.get('url')
            
            if not webhook_url:
                logger.error("Webhook URL not configured")
                return False
                
            # Create webhook payload
            payload = {
                'alert': asdict(alert),
                'timestamp': alert.timestamp,
                'event_type': 'alert_created'
            }
            
            headers = config.get('headers', {})
            headers.setdefault('Content-Type', 'application/json')
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, headers=headers) as response:
                    if 200 <= response.status < 300:
                        logger.info(f"Webhook notification sent for alert {alert.id}")
                        return True
                    else:
                        logger.error(f"Webhook notification failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False
            
    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body"""
        severity_colors = {
            'critical': '#FF0000',
            'high': '#FF8C00',
            'medium': '#FFD700',
            'low': '#32CD32',
            'info': '#1E90FF'
        }
        
        color = severity_colors.get(alert.severity, '#808080')
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5;">
            <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="background-color: {color}; color: white; padding: 20px; text-align: center;">
                    <h1 style="margin: 0; font-size: 24px;">{alert.severity.upper()} ALERT</h1>
                </div>
                <div style="padding: 20px;">
                    <h2 style="color: #333; margin-top: 0;">{alert.title}</h2>
                    <p style="color: #666; font-size: 16px; line-height: 1.5;">{alert.description}</p>
                    
                    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 12px; border: 1px solid #dee2e6; font-weight: bold;">Source</td>
                            <td style="padding: 12px; border: 1px solid #dee2e6;">{alert.source}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border: 1px solid #dee2e6; font-weight: bold;">Metric</td>
                            <td style="padding: 12px; border: 1px solid #dee2e6;">{alert.metric_name}</td>
                        </tr>
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 12px; border: 1px solid #dee2e6; font-weight: bold;">Current Value</td>
                            <td style="padding: 12px; border: 1px solid #dee2e6;">{alert.current_value}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border: 1px solid #dee2e6; font-weight: bold;">Threshold</td>
                            <td style="padding: 12px; border: 1px solid #dee2e6;">{alert.threshold_value}</td>
                        </tr>
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 12px; border: 1px solid #dee2e6; font-weight: bold;">Time</td>
                            <td style="padding: 12px; border: 1px solid #dee2e6;">{datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}</td>
                        </tr>
                    </table>
                    
                    {self._format_context_data(alert.context_data) if alert.context_data else ''}
                </div>
                <div style="background-color: #f8f9fa; padding: 15px; text-align: center; color: #666; font-size: 12px;">
                    Performance Monitoring System • Alert ID: {alert.id}
                </div>
            </div>
        </body>
        </html>
        """
        
    def _format_context_data(self, context_data: Dict[str, Any]) -> str:
        """Format context data for email"""
        if not context_data:
            return ""
            
        rows = ""
        for key, value in context_data.items():
            rows += f"""
                <tr>
                    <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">{key}</td>
                    <td style="padding: 8px; border: 1px solid #dee2e6;">{value}</td>
                </tr>
            """
            
        return f"""
            <h3 style="color: #333; margin-top: 20px;">Additional Context</h3>
            <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
                {rows}
            </table>
        """

class EscalationManager:
    """Manage alert escalation policies"""
    
    def __init__(self, policies: List[EscalationPolicy]):
        self.policies = {p.id: p for p in policies}
        self.active_escalations = {}
        
    def start_escalation(self, alert: Alert, policy_id: str) -> bool:
        """Start escalation for an alert"""
        if policy_id not in self.policies:
            logger.error(f"Escalation policy {policy_id} not found")
            return False
            
        policy = self.policies[policy_id]
        
        escalation = {
            'alert_id': alert.id,
            'policy_id': policy_id,
            'current_step': 0,
            'started_at': time.time(),
            'last_escalation': time.time(),
            'repeat_count': 0
        }
        
        self.active_escalations[alert.id] = escalation
        logger.info(f"Started escalation for alert {alert.id} with policy {policy_id}")
        return True
        
    def check_escalations(self) -> List[Dict[str, Any]]:
        """Check for escalations that need to be triggered"""
        escalations_due = []
        current_time = time.time()
        
        for alert_id, escalation in list(self.active_escalations.items()):
            policy = self.policies[escalation['policy_id']]
            
            # Check if next step should be triggered
            step_index = escalation['current_step']
            
            if step_index < len(policy.steps):
                step = policy.steps[step_index]
                delay_seconds = step.get('delay_minutes', 0) * 60
                
                if current_time - escalation['last_escalation'] >= delay_seconds:
                    escalations_due.append({
                        'alert_id': alert_id,
                        'policy_id': escalation['policy_id'],
                        'step': step,
                        'step_index': step_index
                    })
                    
                    # Update escalation state
                    escalation['current_step'] += 1
                    escalation['last_escalation'] = current_time
                    
            elif escalation['repeat_count'] < policy.repeat_count:
                # Check for repeat escalation
                repeat_delay = policy.repeat_interval_minutes * 60
                
                if current_time - escalation['last_escalation'] >= repeat_delay:
                    # Repeat from first step
                    escalations_due.append({
                        'alert_id': alert_id,
                        'policy_id': escalation['policy_id'],
                        'step': policy.steps[0],
                        'step_index': 0
                    })
                    
                    escalation['current_step'] = 1
                    escalation['last_escalation'] = current_time
                    escalation['repeat_count'] += 1
                    
        return escalations_due
        
    def stop_escalation(self, alert_id: str):
        """Stop escalation for an alert"""
        if alert_id in self.active_escalations:
            del self.active_escalations[alert_id]
            logger.info(f"Stopped escalation for alert {alert_id}")

class IntelligentAlertingSystem:
    """Main intelligent alerting system"""
    
    def __init__(self, config_path: str = "config/alerting.yml"):
        self.config = self._load_config(config_path)
        self.db_path = "/tmp/alerting.db"
        self.init_database()
        
        # Initialize components
        self.anomaly_detector = AnomalyDetector()
        self.correlator = AlertCorrelator()
        self.notification_manager = NotificationManager(self._load_notification_channels())
        self.escalation_manager = EscalationManager(self._load_escalation_policies())
        
        # Load alert rules
        self.alert_rules = self._load_alert_rules()
        
        # State management
        self.active_alerts = {}
        self.rule_states = defaultdict(lambda: {'last_triggered': 0, 'condition_start': None})
        
        # Background tasks
        self.running = False
        self.task_queue = queue.Queue()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'anomaly_detection': {
                'enabled': True,
                'contamination': 0.1,
                'min_samples': 50
            },
            'correlation': {
                'enabled': True,
                'time_window_seconds': 300
            },
            'escalation': {
                'enabled': True,
                'default_policy': 'standard'
            }
        }
        
    def init_database(self):
        """Initialize alerting database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    source TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    tags TEXT,
                    correlation_group TEXT,
                    escalation_level INTEGER DEFAULT 0,
                    acknowledged_by TEXT,
                    acknowledged_at REAL,
                    resolved_at REAL,
                    suppressed_until REAL,
                    notification_channels TEXT,
                    context_data TEXT
                );
                
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    action TEXT NOT NULL,
                    user_id TEXT,
                    details TEXT
                );
                
                CREATE TABLE IF NOT EXISTS notification_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
                CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
                CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
            """)
            
    def _load_alert_rules(self) -> List[AlertRule]:
        """Load alert rules from configuration"""
        rules = []
        
        for rule_config in self.config.get('alert_rules', []):
            rule = AlertRule(**rule_config)
            rules.append(rule)
            logger.info(f"Loaded alert rule: {rule.name}")
            
        return rules
        
    def _load_notification_channels(self) -> List[NotificationChannel]:
        """Load notification channels from configuration"""
        channels = []
        
        for channel_config in self.config.get('notification_channels', []):
            channel = NotificationChannel(**channel_config)
            channels.append(channel)
            logger.info(f"Loaded notification channel: {channel.name}")
            
        return channels
        
    def _load_escalation_policies(self) -> List[EscalationPolicy]:
        """Load escalation policies from configuration"""
        policies = []
        
        for policy_config in self.config.get('escalation_policies', []):
            policy = EscalationPolicy(**policy_config)
            policies.append(policy)
            logger.info(f"Loaded escalation policy: {policy.name}")
            
        return policies
        
    async def process_metric(self, source: str, metric_name: str, value: float, 
                           timestamp: float, tags: Dict[str, str] = None) -> List[Alert]:
        """Process a metric and generate alerts if needed"""
        alerts_generated = []
        
        # Add sample for anomaly detection
        if self.config.get('anomaly_detection', {}).get('enabled', True):
            self.anomaly_detector.add_sample(metric_name, value, timestamp)
            
            # Train model if we have enough samples
            if len(self.anomaly_detector.training_data[metric_name]) >= self.anomaly_detector.min_samples:
                self.anomaly_detector.train_model(metric_name)
                
        # Check alert rules
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            # Check if metric matches rule pattern
            import re
            if not re.match(rule.metric_pattern, metric_name):
                continue
                
            # Evaluate condition
            alert = await self._evaluate_rule(rule, source, metric_name, value, timestamp, tags)
            if alert:
                alerts_generated.append(alert)
                
        return alerts_generated
        
    async def _evaluate_rule(self, rule: AlertRule, source: str, metric_name: str, 
                           value: float, timestamp: float, tags: Dict[str, str] = None) -> Optional[Alert]:
        """Evaluate an alert rule"""
        # Check condition
        condition_met = self._check_condition(rule.condition, value, rule.threshold)
        
        rule_key = f"{rule.id}_{source}_{metric_name}"
        rule_state = self.rule_states[rule_key]
        
        if condition_met:
            # Condition is met
            if rule_state['condition_start'] is None:
                # First time condition is met
                rule_state['condition_start'] = timestamp
                
            # Check if condition has persisted long enough
            if timestamp - rule_state['condition_start'] >= rule.duration_seconds:
                # Check cooldown
                if timestamp - rule_state['last_triggered'] >= rule.cooldown_seconds:
                    # Generate alert
                    alert = await self._create_alert(rule, source, metric_name, value, timestamp, tags)
                    rule_state['last_triggered'] = timestamp
                    return alert
        else:
            # Condition not met, reset state
            rule_state['condition_start'] = None
            
        return None
        
    def _check_condition(self, condition: str, value: float, threshold: float) -> bool:
        """Check if condition is met"""
        if condition == '>':
            return value > threshold
        elif condition == '<':
            return value < threshold
        elif condition == '>=':
            return value >= threshold
        elif condition == '<=':
            return value <= threshold
        elif condition == '==':
            return value == threshold
        elif condition == '!=':
            return value != threshold
        else:
            logger.error(f"Unknown condition: {condition}")
            return False
            
    async def _create_alert(self, rule: AlertRule, source: str, metric_name: str, 
                          value: float, timestamp: float, tags: Dict[str, str] = None) -> Alert:
        """Create an alert"""
        alert_id = f"{rule.id}_{source}_{metric_name}_{int(timestamp)}"
        
        # Get additional context
        context_data = await self._get_alert_context(rule, source, metric_name, value, timestamp)
        
        # Check for anomalies
        anomaly_info = None
        if self.config.get('anomaly_detection', {}).get('enabled', True):
            anomaly_info = self.anomaly_detector.detect_anomaly(metric_name, value, timestamp)
            if anomaly_info['is_anomaly']:
                context_data['anomaly_detection'] = anomaly_info
                
        alert = Alert(
            id=alert_id,
            timestamp=timestamp,
            source=source,
            metric_name=metric_name,
            current_value=value,
            threshold_value=rule.threshold,
            severity=rule.severity,
            status='active',
            title=f"{rule.name}: {metric_name} {rule.condition} {rule.threshold}",
            description=f"Metric {metric_name} from {source} has value {value}, which {rule.condition} threshold {rule.threshold}",
            tags=tags or {},
            notification_channels=rule.notification_channels or [],
            context_data=context_data
        )
        
        # Correlate with existing alerts
        if self.config.get('correlation', {}).get('enabled', True):
            existing_alerts = list(self.active_alerts.values())
            correlation_group = self.correlator.correlate_alert(alert, existing_alerts)
            if correlation_group:
                alert.correlation_group = correlation_group
                
        # Store alert
        await self._store_alert(alert)
        self.active_alerts[alert.id] = alert
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        # Start escalation if configured
        escalation_rules = rule.escalation_rules or []
        if escalation_rules and self.config.get('escalation', {}).get('enabled', True):
            policy_id = escalation_rules[0].get('policy', 'standard')
            self.escalation_manager.start_escalation(alert, policy_id)
            
        logger.info(f"Created alert: {alert.id} (severity: {alert.severity})")
        return alert
        
    async def _get_alert_context(self, rule: AlertRule, source: str, metric_name: str, 
                                value: float, timestamp: float) -> Dict[str, Any]:
        """Get additional context for alert"""
        context = {
            'rule_id': rule.id,
            'rule_name': rule.name,
            'source': source,
            'metric': metric_name,
            'timestamp': timestamp,
            'human_time': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Execute context queries if configured
        if rule.context_queries:
            for query in rule.context_queries:
                try:
                    # This would execute additional queries to gather context
                    # For now, just add placeholder
                    context[f'context_query_{len(context)}'] = f"Query: {query}"
                except Exception as e:
                    logger.error(f"Error executing context query: {e}")
                    
        return context
        
    async def _send_alert_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        if not alert.notification_channels:
            return
            
        try:
            results = await self.notification_manager.send_notification(
                alert, alert.notification_channels
            )
            
            # Log notification results
            for channel_id, success in results.items():
                await self._log_notification(alert.id, channel_id, success)
                
            success_count = sum(1 for success in results.values() if success)
            logger.info(f"Sent notifications for alert {alert.id}: {success_count}/{len(results)} successful")
            
        except Exception as e:
            logger.error(f"Error sending notifications for alert {alert.id}: {e}")
            
    async def _store_alert(self, alert: Alert):
        """Store alert in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alerts 
                (id, timestamp, source, metric_name, current_value, threshold_value,
                 severity, status, title, description, tags, correlation_group,
                 escalation_level, notification_channels, context_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id, alert.timestamp, alert.source, alert.metric_name,
                alert.current_value, alert.threshold_value, alert.severity,
                alert.status, alert.title, alert.description,
                json.dumps(alert.tags), alert.correlation_group,
                alert.escalation_level, json.dumps(alert.notification_channels),
                json.dumps(alert.context_data)
            ))
            
    async def _log_notification(self, alert_id: str, channel_id: str, success: bool, error_message: str = None):
        """Log notification attempt"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO notification_log (alert_id, channel_id, timestamp, success, error_message)
                VALUES (?, ?, ?, ?, ?)
            """, (alert_id, channel_id, time.time(), success, error_message))
            
    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id not in self.active_alerts:
            return False
            
        alert = self.active_alerts[alert_id]
        alert.status = 'acknowledged'
        alert.acknowledged_by = user_id
        alert.acknowledged_at = time.time()
        
        # Stop escalation
        self.escalation_manager.stop_escalation(alert_id)
        
        # Update database
        await self._store_alert(alert)
        
        # Log action
        await self._log_alert_action(alert_id, 'acknowledged', user_id)
        
        logger.info(f"Alert {alert_id} acknowledged by {user_id}")
        return True
        
    async def resolve_alert(self, alert_id: str, user_id: str = None) -> bool:
        """Resolve an alert"""
        if alert_id not in self.active_alerts:
            return False
            
        alert = self.active_alerts[alert_id]
        alert.status = 'resolved'
        alert.resolved_at = time.time()
        
        # Stop escalation
        self.escalation_manager.stop_escalation(alert_id)
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        # Update database
        await self._store_alert(alert)
        
        # Log action
        await self._log_alert_action(alert_id, 'resolved', user_id)
        
        logger.info(f"Alert {alert_id} resolved")
        return True
        
    async def _log_alert_action(self, alert_id: str, action: str, user_id: str = None, details: str = None):
        """Log alert action"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO alert_history (alert_id, timestamp, action, user_id, details)
                VALUES (?, ?, ?, ?, ?)
            """, (alert_id, time.time(), action, user_id, details))
            
    async def start_background_tasks(self):
        """Start background processing tasks"""
        self.running = True
        
        # Start escalation checker
        asyncio.create_task(self._escalation_checker())
        
        # Start alert cleaner
        asyncio.create_task(self._alert_cleaner())
        
        logger.info("Started background alerting tasks")
        
    async def _escalation_checker(self):
        """Check for escalations that need to be triggered"""
        while self.running:
            try:
                escalations_due = self.escalation_manager.check_escalations()
                
                for escalation in escalations_due:
                    alert_id = escalation['alert_id']
                    step = escalation['step']
                    
                    if alert_id in self.active_alerts:
                        alert = self.active_alerts[alert_id]
                        channels = step.get('channels', [])
                        
                        if channels:
                            await self.notification_manager.send_notification(alert, channels)
                            logger.info(f"Escalated alert {alert_id} to step {escalation['step_index']}")
                            
            except Exception as e:
                logger.error(f"Error in escalation checker: {e}")
                
            await asyncio.sleep(60)  # Check every minute
            
    async def _alert_cleaner(self):
        """Clean up old resolved alerts"""
        while self.running:
            try:
                cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        DELETE FROM alerts 
                        WHERE status = 'resolved' AND resolved_at < ?
                    """, (cutoff_time,))
                    
                logger.debug("Cleaned up old resolved alerts")
                
            except Exception as e:
                logger.error(f"Error in alert cleaner: {e}")
                
            await asyncio.sleep(3600)  # Clean every hour
            
    def stop_background_tasks(self):
        """Stop background processing tasks"""
        self.running = False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Alerting System")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--test-alert", action="store_true", help="Send test alert")
    
    args = parser.parse_args()
    
    async def main():
        config_path = args.config or "config/alerting.yml"
        alerting_system = IntelligentAlertingSystem(config_path)
        
        if args.test_alert:
            # Send test alert
            alerts = await alerting_system.process_metric(
                source='test_system',
                metric_name='cpu_usage',
                value=95.0,
                timestamp=time.time(),
                tags={'host': 'test-host', 'service': 'test-service'}
            )
            
            print(f"Generated {len(alerts)} test alerts")
            for alert in alerts:
                print(f"  • {alert.title} (severity: {alert.severity})")
        else:
            # Start alerting system
            await alerting_system.start_background_tasks()
            
            print("Intelligent Alerting System started")
            print("Press Ctrl+C to stop")
            
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                alerting_system.stop_background_tasks()
                print("Alerting system stopped")
                
    asyncio.run(main())