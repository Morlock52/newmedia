#!/usr/bin/env python3
"""
Intelligent Alerting System with AI-powered Alert Management
Smart alert correlation, deduplication, and escalation
"""

import asyncio
import json
import logging
import sqlite3
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.text import MIMEText, MIMEMultipart
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

class AlertSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"

@dataclass
class Alert:
    """Enhanced alert structure with AI capabilities"""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    source: str
    component: str
    metric: str
    value: float
    threshold: float
    tags: Dict[str, str]
    fingerprint: str
    correlation_id: Optional[str] = None
    escalation_level: int = 0
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    suppressed_until: Optional[datetime] = None
    notification_count: int = 0
    last_notification: Optional[datetime] = None
    runbook_url: Optional[str] = None
    impact_score: float = 0.0
    business_impact: Optional[str] = None
    related_alerts: List[str] = None

    def __post_init__(self):
        if self.related_alerts is None:
            self.related_alerts = []
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()

    def _generate_fingerprint(self) -> str:
        """Generate unique fingerprint for alert deduplication"""
        content = f"{self.source}:{self.component}:{self.metric}:{self.title}"
        return hashlib.md5(content.encode()).hexdigest()

@dataclass
class EscalationRule:
    """Escalation rule configuration"""
    severity: AlertSeverity
    initial_delay: int  # minutes
    escalation_delay: int  # minutes
    max_escalations: int
    channels: List[str]
    conditions: Dict[str, Any]

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    name: str
    type: str  # email, slack, discord, webhook, etc.
    config: Dict[str, Any]
    enabled: bool = True
    rate_limit: Optional[int] = None  # messages per hour

class AlertCorrelator:
    """AI-powered alert correlation engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.similarity_threshold = config.get('correlation_threshold', 0.7)
        
    def correlate_alerts(self, alerts: List[Alert]) -> Dict[str, List[Alert]]:
        """Group related alerts using AI correlation"""
        if len(alerts) < 2:
            return {alert.id: [alert] for alert in alerts}
        
        try:
            # Extract features for correlation
            features = self._extract_features(alerts)
            
            # Calculate similarity matrix
            similarity_matrix = self._calculate_similarity(features)
            
            # Cluster similar alerts
            clusters = self._cluster_alerts(similarity_matrix, alerts)
            
            return clusters
            
        except Exception as e:
            logging.error(f"Alert correlation failed: {e}")
            # Fallback to simple grouping
            return self._simple_correlation(alerts)
    
    def _extract_features(self, alerts: List[Alert]) -> np.ndarray:
        """Extract features for alert correlation"""
        # Combine textual features
        texts = []
        for alert in alerts:
            text = f"{alert.title} {alert.description} {alert.component} {alert.source}"
            texts.append(text)
        
        # Vectorize text features
        text_features = self.vectorizer.fit_transform(texts).toarray()
        
        # Add numerical features
        numerical_features = []
        for alert in alerts:
            num_features = [
                alert.severity.value == AlertSeverity.CRITICAL.value,
                alert.severity.value == AlertSeverity.HIGH.value,
                alert.severity.value == AlertSeverity.MEDIUM.value,
                alert.impact_score,
                alert.escalation_level
            ]
            numerical_features.append(num_features)
        
        numerical_features = np.array(numerical_features)
        
        # Combine features
        combined_features = np.hstack([text_features, numerical_features])
        
        return combined_features
    
    def _calculate_similarity(self, features: np.ndarray) -> np.ndarray:
        """Calculate similarity matrix between alerts"""
        return cosine_similarity(features)
    
    def _cluster_alerts(self, similarity_matrix: np.ndarray, 
                       alerts: List[Alert]) -> Dict[str, List[Alert]]:
        """Cluster alerts based on similarity"""
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(
            eps=1 - self.similarity_threshold,
            min_samples=1,
            metric='precomputed'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Group alerts by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            cluster_id = f"cluster_{label}" if label != -1 else f"single_{i}"
            
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            
            clusters[cluster_id].append(alerts[i])
            
            # Update correlation IDs
            alerts[i].correlation_id = cluster_id
        
        return clusters
    
    def _simple_correlation(self, alerts: List[Alert]) -> Dict[str, List[Alert]]:
        """Simple fallback correlation by component and time"""
        clusters = {}
        
        for alert in alerts:
            # Group by component and time window (5 minutes)
            time_window = alert.timestamp.replace(second=0, microsecond=0)
            time_window = time_window.replace(minute=(time_window.minute // 5) * 5)
            
            cluster_key = f"{alert.component}_{time_window.isoformat()}"
            
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            
            clusters[cluster_key].append(alert)
            alert.correlation_id = cluster_key
        
        return clusters

class AlertDeduplicator:
    """Smart alert deduplication engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dedup_window = config.get('deduplication_window', 300)  # 5 minutes
        self.fingerprint_cache = {}
        
    def deduplicate_alert(self, new_alert: Alert, 
                         existing_alerts: List[Alert]) -> Tuple[bool, Optional[Alert]]:
        """Deduplicate alert against existing alerts"""
        
        # Check for exact fingerprint match
        for existing in existing_alerts:
            if existing.fingerprint == new_alert.fingerprint:
                # Check if within deduplication window
                time_diff = (new_alert.timestamp - existing.timestamp).total_seconds()
                
                if time_diff <= self.dedup_window:
                    # Update existing alert
                    existing.notification_count += 1
                    existing.timestamp = new_alert.timestamp
                    existing.value = new_alert.value
                    
                    return True, existing
        
        # Check for semantic similarity
        similar_alert = self._find_similar_alert(new_alert, existing_alerts)
        if similar_alert:
            return True, similar_alert
        
        return False, None
    
    def _find_similar_alert(self, new_alert: Alert, 
                           existing_alerts: List[Alert]) -> Optional[Alert]:
        """Find semantically similar alert"""
        try:
            if not existing_alerts:
                return None
            
            # Filter alerts within time window
            recent_alerts = [
                alert for alert in existing_alerts
                if (new_alert.timestamp - alert.timestamp).total_seconds() <= self.dedup_window
            ]
            
            if not recent_alerts:
                return None
            
            # Calculate similarity
            new_text = f"{new_alert.title} {new_alert.description}"
            
            for existing in recent_alerts:
                existing_text = f"{existing.title} {existing.description}"
                
                # Simple word overlap similarity
                new_words = set(new_text.lower().split())
                existing_words = set(existing_text.lower().split())
                
                if len(new_words.union(existing_words)) == 0:
                    continue
                
                similarity = len(new_words.intersection(existing_words)) / len(new_words.union(existing_words))
                
                if similarity > 0.8:  # 80% similarity threshold
                    return existing
            
            return None
            
        except Exception as e:
            logging.error(f"Similar alert detection failed: {e}")
            return None

class ImpactAnalyzer:
    """Business impact analysis for alerts"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.impact_rules = config.get('impact_rules', {})
        self.component_weights = config.get('component_weights', {})
        
    def calculate_impact_score(self, alert: Alert) -> float:
        """Calculate business impact score for alert"""
        try:
            base_score = self._get_base_score(alert.severity)
            component_weight = self.component_weights.get(alert.component, 1.0)
            time_factor = self._get_time_factor(alert.timestamp)
            
            # Apply business rules
            business_factor = self._apply_business_rules(alert)
            
            impact_score = base_score * component_weight * time_factor * business_factor
            
            return min(100.0, max(0.0, impact_score))
            
        except Exception as e:
            logging.error(f"Impact score calculation failed: {e}")
            return 50.0  # Default medium impact
    
    def _get_base_score(self, severity: AlertSeverity) -> float:
        """Get base score for severity level"""
        severity_scores = {
            AlertSeverity.CRITICAL: 100.0,
            AlertSeverity.HIGH: 75.0,
            AlertSeverity.MEDIUM: 50.0,
            AlertSeverity.LOW: 25.0,
            AlertSeverity.INFO: 10.0
        }
        return severity_scores.get(severity, 50.0)
    
    def _get_time_factor(self, timestamp: datetime) -> float:
        """Calculate time-based impact factor"""
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        
        # Higher impact during business hours
        if 9 <= hour <= 17 and day_of_week < 5:  # Business hours
            return 1.5
        elif 18 <= hour <= 22:  # Evening
            return 1.2
        else:  # Night/Weekend
            return 0.8
    
    def _apply_business_rules(self, alert: Alert) -> float:
        """Apply business-specific impact rules"""
        factor = 1.0
        
        # Critical services have higher impact
        if alert.component in ['jellyfin', 'traefik', 'prometheus']:
            factor *= 1.3
        
        # User-facing components have higher impact
        if 'user' in alert.tags.get('type', '').lower():
            factor *= 1.2
        
        # Security alerts have higher impact
        if 'security' in alert.tags.get('category', '').lower():
            factor *= 1.4
        
        return factor

class NotificationManager:
    """Intelligent notification management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.channels = self._load_channels()
        self.rate_limits = {}
        self.notification_history = []
        
    def _load_channels(self) -> Dict[str, NotificationChannel]:
        """Load notification channels from config"""
        channels = {}
        
        for channel_config in self.config.get('channels', []):
            channel = NotificationChannel(**channel_config)
            channels[channel.name] = channel
        
        return channels
    
    async def send_notification(self, alert: Alert, channel_names: List[str] = None):
        """Send notification through specified channels"""
        if channel_names is None:
            channel_names = list(self.channels.keys())
        
        tasks = []
        for channel_name in channel_names:
            if channel_name in self.channels:
                channel = self.channels[channel_name]
                if channel.enabled and self._check_rate_limit(channel):
                    task = self._send_to_channel(alert, channel)
                    tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check if channel is within rate limits"""
        if not channel.rate_limit:
            return True
        
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Count notifications in the last hour
        recent_notifications = [
            n for n in self.notification_history
            if n['channel'] == channel.name and n['timestamp'] > hour_ago
        ]
        
        return len(recent_notifications) < channel.rate_limit
    
    async def _send_to_channel(self, alert: Alert, channel: NotificationChannel):
        """Send alert to specific channel"""
        try:
            if channel.type == 'email':
                await self._send_email(alert, channel)
            elif channel.type == 'slack':
                await self._send_slack(alert, channel)
            elif channel.type == 'discord':
                await self._send_discord(alert, channel)
            elif channel.type == 'webhook':
                await self._send_webhook(alert, channel)
            else:
                logging.warning(f"Unknown channel type: {channel.type}")
            
            # Record notification
            self.notification_history.append({
                'channel': channel.name,
                'alert_id': alert.id,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            logging.error(f"Failed to send notification to {channel.name}: {e}")
    
    async def _send_email(self, alert: Alert, channel: NotificationChannel):
        """Send email notification"""
        config = channel.config
        
        msg = MIMEMultipart()
        msg['From'] = config['from']
        msg['To'] = ', '.join(config['to'])
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        
        body = self._format_alert_email(alert)
        msg.attach(MIMEText(body, 'html'))
        
        # Send email in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                self._send_email_sync,
                msg, config
            )
    
    def _send_email_sync(self, msg: MIMEMultipart, config: Dict[str, Any]):
        """Send email synchronously"""
        server = smtplib.SMTP(config['smtp_host'], config['smtp_port'])
        server.starttls()
        server.login(config['username'], config['password'])
        server.send_message(msg)
        server.quit()
    
    async def _send_slack(self, alert: Alert, channel: NotificationChannel):
        """Send Slack notification"""
        webhook_url = channel.config['webhook_url']
        
        payload = {
            'text': f"*{alert.severity.value.upper()}*: {alert.title}",
            'attachments': [{
                'color': self._get_alert_color(alert.severity),
                'fields': [
                    {'title': 'Component', 'value': alert.component, 'short': True},
                    {'title': 'Source', 'value': alert.source, 'short': True},
                    {'title': 'Value', 'value': str(alert.value), 'short': True},
                    {'title': 'Threshold', 'value': str(alert.threshold), 'short': True},
                    {'title': 'Description', 'value': alert.description, 'short': False}
                ],
                'ts': int(alert.timestamp.timestamp())
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Slack webhook failed: {response.status}")
    
    async def _send_discord(self, alert: Alert, channel: NotificationChannel):
        """Send Discord notification"""
        webhook_url = channel.config['webhook_url']
        
        embed = {
            'title': alert.title,
            'description': alert.description,
            'color': self._get_discord_color(alert.severity),
            'fields': [
                {'name': 'Component', 'value': alert.component, 'inline': True},
                {'name': 'Source', 'value': alert.source, 'inline': True},
                {'name': 'Severity', 'value': alert.severity.value.upper(), 'inline': True},
                {'name': 'Value', 'value': str(alert.value), 'inline': True},
                {'name': 'Threshold', 'value': str(alert.threshold), 'inline': True}
            ],
            'timestamp': alert.timestamp.isoformat(),
            'footer': {'text': f'Alert ID: {alert.id}'}
        }
        
        payload = {'embeds': [embed]}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status not in [200, 204]:
                    raise Exception(f"Discord webhook failed: {response.status}")
    
    async def _send_webhook(self, alert: Alert, channel: NotificationChannel):
        """Send generic webhook notification"""
        webhook_url = channel.config['url']
        
        payload = {
            'alert': asdict(alert),
            'timestamp': alert.timestamp.isoformat()
        }
        
        headers = channel.config.get('headers', {})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload, headers=headers) as response:
                if response.status not in [200, 201, 202, 204]:
                    raise Exception(f"Webhook failed: {response.status}")
    
    def _format_alert_email(self, alert: Alert) -> str:
        """Format alert as HTML email"""
        color = self._get_alert_color(alert.severity)
        
        return f"""
        <html>
        <body>
            <div style="border-left: 4px solid {color}; padding: 15px; background-color: #f9f9f9;">
                <h2 style="color: {color}; margin: 0;">{alert.title}</h2>
                <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                <p><strong>Component:</strong> {alert.component}</p>
                <p><strong>Source:</strong> {alert.source}</p>
                <p><strong>Value:</strong> {alert.value}</p>
                <p><strong>Threshold:</strong> {alert.threshold}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
                <p>{alert.description}</p>
                {f'<p><strong>Runbook:</strong> <a href="{alert.runbook_url}">View Runbook</a></p>' if alert.runbook_url else ''}
                <p><em>Alert ID: {alert.id}</em></p>
            </div>
        </body>
        </html>
        """
    
    def _get_alert_color(self, severity: AlertSeverity) -> str:
        """Get color for alert severity"""
        colors = {
            AlertSeverity.CRITICAL: '#ff0000',
            AlertSeverity.HIGH: '#ff8800',
            AlertSeverity.MEDIUM: '#ffaa00',
            AlertSeverity.LOW: '#00aa00',
            AlertSeverity.INFO: '#0088ff'
        }
        return colors.get(severity, '#666666')
    
    def _get_discord_color(self, severity: AlertSeverity) -> int:
        """Get Discord embed color for alert severity"""
        colors = {
            AlertSeverity.CRITICAL: 0xff0000,
            AlertSeverity.HIGH: 0xff8800,
            AlertSeverity.MEDIUM: 0xffaa00,
            AlertSeverity.LOW: 0x00aa00,
            AlertSeverity.INFO: 0x0088ff
        }
        return colors.get(severity, 0x666666)

class EscalationManager:
    """Intelligent alert escalation management"""
    
    def __init__(self, config: Dict[str, Any], notification_manager: NotificationManager):
        self.config = config
        self.notification_manager = notification_manager
        self.escalation_rules = self._load_escalation_rules()
        
    def _load_escalation_rules(self) -> Dict[AlertSeverity, EscalationRule]:
        """Load escalation rules from config"""
        rules = {}
        
        for rule_config in self.config.get('escalation_rules', []):
            severity = AlertSeverity(rule_config['severity'])
            rule = EscalationRule(**rule_config)
            rules[severity] = rule
        
        return rules
    
    async def check_escalation(self, alert: Alert) -> bool:
        """Check if alert needs escalation"""
        if alert.severity not in self.escalation_rules:
            return False
        
        rule = self.escalation_rules[alert.severity]
        
        # Check if initial delay has passed
        time_since_alert = (datetime.now() - alert.timestamp).total_seconds() / 60
        
        if alert.escalation_level == 0:
            if time_since_alert >= rule.initial_delay:
                await self._escalate_alert(alert, rule)
                return True
        else:
            # Check for further escalation
            if alert.escalation_level < rule.max_escalations:
                if time_since_alert >= (rule.initial_delay + rule.escalation_delay * alert.escalation_level):
                    await self._escalate_alert(alert, rule)
                    return True
        
        return False
    
    async def _escalate_alert(self, alert: Alert, rule: EscalationRule):
        """Escalate alert to next level"""
        alert.escalation_level += 1
        alert.status = AlertStatus.ESCALATED
        
        # Determine escalation channels
        escalation_channels = rule.channels
        if alert.escalation_level <= len(escalation_channels):
            channel_name = escalation_channels[alert.escalation_level - 1]
            await self.notification_manager.send_notification(alert, [channel_name])
        
        logging.info(f"Escalated alert {alert.id} to level {alert.escalation_level}")

class IntelligentAlertingSystem:
    """Main intelligent alerting orchestrator"""
    
    def __init__(self, config_path: str = '/app/config/alerting.yml'):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.correlator = AlertCorrelator(self.config.get('correlation', {}))
        self.deduplicator = AlertDeduplicator(self.config.get('deduplication', {}))
        self.impact_analyzer = ImpactAnalyzer(self.config.get('impact_analysis', {}))
        self.notification_manager = NotificationManager(self.config.get('notifications', {}))
        self.escalation_manager = EscalationManager(
            self.config.get('escalation', {}),
            self.notification_manager
        )
        
        # Database setup
        self.db_path = self.config.get('database_path', '/app/data/alerts.db')
        self._init_database()
        
        # Active alerts storage
        self.active_alerts: Dict[str, Alert] = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load alerting configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default alerting configuration"""
        return {
            'correlation': {
                'correlation_threshold': 0.7,
                'time_window': 300
            },
            'deduplication': {
                'deduplication_window': 300
            },
            'impact_analysis': {
                'component_weights': {
                    'jellyfin': 1.5,
                    'traefik': 1.3,
                    'prometheus': 1.2,
                    'grafana': 1.0
                }
            },
            'notifications': {
                'channels': [
                    {
                        'name': 'email',
                        'type': 'email',
                        'enabled': True,
                        'config': {
                            'smtp_host': 'localhost',
                            'smtp_port': 587,
                            'from': 'alerts@example.com',
                            'to': ['admin@example.com']
                        }
                    }
                ]
            },
            'escalation': {
                'escalation_rules': [
                    {
                        'severity': 'critical',
                        'initial_delay': 5,
                        'escalation_delay': 15,
                        'max_escalations': 3,
                        'channels': ['email', 'slack', 'phone']
                    }
                ]
            }
        }
    
    def _init_database(self):
        """Initialize alerts database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                source TEXT NOT NULL,
                component TEXT NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                threshold REAL NOT NULL,
                tags TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                correlation_id TEXT,
                escalation_level INTEGER DEFAULT 0,
                acknowledged_by TEXT,
                acknowledged_at DATETIME,
                resolved_at DATETIME,
                suppressed_until DATETIME,
                notification_count INTEGER DEFAULT 0,
                last_notification DATETIME,
                runbook_url TEXT,
                impact_score REAL DEFAULT 0.0,
                business_impact TEXT,
                related_alerts TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                action TEXT NOT NULL,
                details TEXT,
                user_id TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def process_alert(self, alert_data: Dict[str, Any]) -> Alert:
        """Process incoming alert through intelligent pipeline"""
        try:
            # Create alert object
            alert = self._create_alert(alert_data)
            
            # Calculate impact score
            alert.impact_score = self.impact_analyzer.calculate_impact_score(alert)
            
            # Check for deduplication
            existing_alerts = list(self.active_alerts.values())
            is_duplicate, existing_alert = self.deduplicator.deduplicate_alert(alert, existing_alerts)
            
            if is_duplicate and existing_alert:
                logging.info(f"Alert {alert.id} deduplicated with {existing_alert.id}")
                await self._update_alert(existing_alert)
                return existing_alert
            
            # Store new alert
            self.active_alerts[alert.id] = alert
            await self._store_alert(alert)
            
            # Correlate with other alerts
            await self._correlate_new_alert(alert)
            
            # Send initial notification
            await self.notification_manager.send_notification(alert)
            alert.notification_count = 1
            alert.last_notification = datetime.now()
            
            # Log alert creation
            await self._log_alert_action(alert.id, 'created', f'New alert: {alert.title}')
            
            logging.info(f"Processed new alert: {alert.id} ({alert.severity.value})")
            
            return alert
            
        except Exception as e:
            logging.error(f"Failed to process alert: {e}")
            raise
    
    def _create_alert(self, alert_data: Dict[str, Any]) -> Alert:
        """Create Alert object from data"""
        return Alert(
            id=alert_data.get('id', f"alert_{int(datetime.now().timestamp())}"),
            timestamp=datetime.fromisoformat(alert_data.get('timestamp', datetime.now().isoformat())),
            severity=AlertSeverity(alert_data.get('severity', 'medium')),
            status=AlertStatus.ACTIVE,
            title=alert_data['title'],
            description=alert_data.get('description', ''),
            source=alert_data.get('source', 'unknown'),
            component=alert_data.get('component', 'unknown'),
            metric=alert_data.get('metric', 'unknown'),
            value=float(alert_data.get('value', 0)),
            threshold=float(alert_data.get('threshold', 0)),
            tags=alert_data.get('tags', {}),
            fingerprint=alert_data.get('fingerprint', ''),
            runbook_url=alert_data.get('runbook_url')
        )
    
    async def _correlate_new_alert(self, new_alert: Alert):
        """Correlate new alert with existing alerts"""
        try:
            recent_alerts = [
                alert for alert in self.active_alerts.values()
                if (new_alert.timestamp - alert.timestamp).total_seconds() <= 600  # 10 minutes
            ]
            
            if len(recent_alerts) > 1:
                clusters = self.correlator.correlate_alerts(recent_alerts)
                
                # Update correlation IDs
                for cluster_id, alerts in clusters.items():
                    if len(alerts) > 1:
                        for alert in alerts:
                            alert.correlation_id = cluster_id
                            
                            # Link related alerts
                            related_ids = [a.id for a in alerts if a.id != alert.id]
                            alert.related_alerts = related_ids
                            
                            await self._update_alert(alert)
                
        except Exception as e:
            logging.error(f"Alert correlation failed: {e}")
    
    async def acknowledge_alert(self, alert_id: str, user_id: str = None):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = user_id
            alert.acknowledged_at = datetime.now()
            
            await self._update_alert(alert)
            await self._log_alert_action(alert_id, 'acknowledged', f'Acknowledged by {user_id}', user_id)
            
            logging.info(f"Alert {alert_id} acknowledged by {user_id}")
    
    async def resolve_alert(self, alert_id: str, user_id: str = None):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            await self._update_alert(alert)
            await self._log_alert_action(alert_id, 'resolved', f'Resolved by {user_id}', user_id)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logging.info(f"Alert {alert_id} resolved by {user_id}")
    
    async def start_alerting_system(self):
        """Start the intelligent alerting system"""
        logging.info("Starting Intelligent Alerting System...")
        
        while True:
            try:
                # Check for escalations
                for alert in list(self.active_alerts.values()):
                    if alert.status == AlertStatus.ACTIVE:
                        await self.escalation_manager.check_escalation(alert)
                
                # Clean up old resolved alerts from database
                await self._cleanup_old_alerts()
                
                # Sleep for 1 minute
                await asyncio.sleep(60)
                
            except Exception as e:
                logging.error(f"Alerting system error: {e}")
                await asyncio.sleep(5)
    
    async def _store_alert(self, alert: Alert):
        """Store alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO alerts (
                    id, timestamp, severity, status, title, description, source,
                    component, metric, value, threshold, tags, fingerprint,
                    correlation_id, escalation_level, acknowledged_by, acknowledged_at,
                    resolved_at, suppressed_until, notification_count, last_notification,
                    runbook_url, impact_score, business_impact, related_alerts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.timestamp, alert.severity.value, alert.status.value,
                alert.title, alert.description, alert.source, alert.component,
                alert.metric, alert.value, alert.threshold, json.dumps(alert.tags),
                alert.fingerprint, alert.correlation_id, alert.escalation_level,
                alert.acknowledged_by, alert.acknowledged_at, alert.resolved_at,
                alert.suppressed_until, alert.notification_count, alert.last_notification,
                alert.runbook_url, alert.impact_score, alert.business_impact,
                json.dumps(alert.related_alerts)
            ))
            
            conn.commit()
        finally:
            conn.close()
    
    async def _update_alert(self, alert: Alert):
        """Update existing alert in database"""
        await self._store_alert(alert)
    
    async def _log_alert_action(self, alert_id: str, action: str, 
                               details: str = None, user_id: str = None):
        """Log alert action to history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO alert_history (alert_id, action, details, user_id)
                VALUES (?, ?, ?, ?)
            ''', (alert_id, action, details, user_id))
            
            conn.commit()
        finally:
            conn.close()
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        cutoff_date = datetime.now() - timedelta(days=30)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                DELETE FROM alerts 
                WHERE status = 'resolved' AND resolved_at < ?
            ''', (cutoff_date,))
            
            cursor.execute('''
                DELETE FROM alert_history 
                WHERE timestamp < ?
            ''', (cutoff_date,))
            
            conn.commit()
        finally:
            conn.close()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    system = IntelligentAlertingSystem()
    asyncio.run(system.start_alerting_system())