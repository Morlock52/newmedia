#!/usr/bin/env python3
"""
User Experience Automation
Handles request automation, notification systems, progress tracking, and usage analytics
"""

import os
import json
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import websockets
import jwt
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/user_experience.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RequestStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"

class NotificationType(Enum):
    EMAIL = "email"
    WEBHOOK = "webhook"
    WEBSOCKET = "websocket"
    PUSH = "push"
    SMS = "sms"

@dataclass
class UserRequest:
    user_id: str
    request_type: str
    title: str
    media_type: str
    status: RequestStatus = RequestStatus.PENDING
    priority: int = 5
    requested_date: datetime = None
    completed_date: Optional[datetime] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.requested_date is None:
            self.requested_date = datetime.now()

@dataclass
class Notification:
    user_id: str
    type: NotificationType
    title: str
    message: str
    priority: int = 5
    data: Optional[Dict] = None
    sent: bool = False
    sent_date: Optional[datetime] = None

class UserExperienceManager:
    def __init__(self, config_path='/config/user_experience.json'):
        self.config = self.load_config(config_path)
        self.db_path = self.config.get('database_path', '/config/user_experience.db')
        self.init_database()
        self.notification_handlers = self.init_notification_handlers()
        self.websocket_clients = set()
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        default_config = {
            'api_port': 8001,
            'websocket_port': 8002,
            'jwt_secret': 'change-me-in-production',
            'notifications': {
                'email': {
                    'enabled': True,
                    'smtp_host': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'smtp_user': '',
                    'smtp_password': '',
                    'from_address': 'media-server@example.com'
                },
                'webhook': {
                    'enabled': True,
                    'endpoints': []
                },
                'websocket': {
                    'enabled': True
                }
            },
            'auto_approval': {
                'enabled': True,
                'user_rating_threshold': 3,
                'max_requests_per_day': 10,
                'quality_limits': {
                    'movie': '1080p',
                    'tv': '1080p'
                }
            },
            'analytics': {
                'enabled': True,
                'retention_days': 90,
                'report_schedule': 'weekly'
            },
            'rate_limiting': {
                'enabled': True,
                'requests_per_hour': 20,
                'requests_per_day': 50
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        return default_config
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                rating INTEGER DEFAULT 3,
                request_count INTEGER DEFAULT 0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP
            )
        ''')
        
        # Requests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                request_type TEXT NOT NULL,
                title TEXT NOT NULL,
                media_type TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER DEFAULT 5,
                requested_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_date TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Notifications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                priority INTEGER DEFAULT 5,
                data TEXT,
                sent BOOLEAN DEFAULT 0,
                sent_date TIMESTAMP,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Activity log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                dimension1 TEXT,
                dimension2 TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def init_notification_handlers(self):
        """Initialize notification handlers"""
        handlers = {}
        
        if self.config['notifications']['email']['enabled']:
            handlers[NotificationType.EMAIL] = self.send_email_notification
        
        if self.config['notifications']['webhook']['enabled']:
            handlers[NotificationType.WEBHOOK] = self.send_webhook_notification
        
        if self.config['notifications']['websocket']['enabled']:
            handlers[NotificationType.WEBSOCKET] = self.send_websocket_notification
        
        return handlers
    
    async def create_request(self, user_id: str, request_data: Dict) -> Dict:
        """Create a new user request"""
        try:
            # Validate user
            user = await self.get_or_create_user(user_id, request_data.get('username'))
            
            # Check rate limiting
            if not await self.check_rate_limit(user_id):
                return {'error': 'Rate limit exceeded', 'status': 429}
            
            # Create request object
            request = UserRequest(
                user_id=user_id,
                request_type=request_data.get('type', 'download'),
                title=request_data['title'],
                media_type=request_data['media_type'],
                priority=request_data.get('priority', 5),
                notes=request_data.get('notes')
            )
            
            # Check auto-approval
            if await self.check_auto_approval(user, request):
                request.status = RequestStatus.APPROVED
                await self.process_approved_request(request)
            
            # Save to database
            request_id = await self.save_request(request)
            
            # Log activity
            await self.log_activity(user_id, 'request_created', {
                'request_id': request_id,
                'title': request.title
            })
            
            # Send notification
            await self.notify_user(
                user_id,
                "Request Received",
                f"Your request for '{request.title}' has been received and is {request.status.value}.",
                NotificationType.WEBSOCKET,
                {'request_id': request_id, 'status': request.status.value}
            )
            
            return {
                'request_id': request_id,
                'status': request.status.value,
                'message': f'Request for {request.title} has been {request.status.value}'
            }
            
        except Exception as e:
            logger.error(f"Error creating request: {e}")
            return {'error': str(e), 'status': 500}
    
    async def get_or_create_user(self, user_id: str, username: Optional[str] = None) -> Dict:
        """Get or create user record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        
        if not user:
            # Create new user
            username = username or f"user_{user_id}"
            cursor.execute('''
                INSERT INTO users (id, username, last_active)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, username))
            conn.commit()
            
            user = {
                'id': user_id,
                'username': username,
                'rating': 3,
                'request_count': 0
            }
        else:
            # Update last active
            cursor.execute('''
                UPDATE users SET last_active = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (user_id,))
            conn.commit()
            
            user = {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'rating': user[3],
                'request_count': user[4]
            }
        
        conn.close()
        return user
    
    async def check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limits"""
        if not self.config['rate_limiting']['enabled']:
            return True
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check hourly limit
        cursor.execute('''
            SELECT COUNT(*) FROM requests
            WHERE user_id = ? AND requested_date > datetime('now', '-1 hour')
        ''', (user_id,))
        
        hourly_count = cursor.fetchone()[0]
        if hourly_count >= self.config['rate_limiting']['requests_per_hour']:
            conn.close()
            return False
        
        # Check daily limit
        cursor.execute('''
            SELECT COUNT(*) FROM requests
            WHERE user_id = ? AND requested_date > datetime('now', '-1 day')
        ''', (user_id,))
        
        daily_count = cursor.fetchone()[0]
        conn.close()
        
        return daily_count < self.config['rate_limiting']['requests_per_day']
    
    async def check_auto_approval(self, user: Dict, request: UserRequest) -> bool:
        """Check if request should be auto-approved"""
        if not self.config['auto_approval']['enabled']:
            return False
        
        # Check user rating
        if user['rating'] < self.config['auto_approval']['user_rating_threshold']:
            return False
        
        # Check daily request count
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM requests
            WHERE user_id = ? AND requested_date > datetime('now', '-1 day')
        ''', (user['id'],))
        
        daily_count = cursor.fetchone()[0]
        conn.close()
        
        if daily_count >= self.config['auto_approval']['max_requests_per_day']:
            return False
        
        # Check quality limits (would need external API to verify)
        # For now, auto-approve
        return True
    
    async def process_approved_request(self, request: UserRequest):
        """Process an approved request"""
        try:
            # Update status
            request.status = RequestStatus.PROCESSING
            await self.update_request_status(request)
            
            # Forward to appropriate service (Sonarr/Radarr/etc)
            if request.media_type == 'movie':
                success = await self.add_to_radarr(request)
            elif request.media_type == 'tv':
                success = await self.add_to_sonarr(request)
            else:
                success = False
            
            # Update final status
            if success:
                request.status = RequestStatus.COMPLETED
                request.completed_date = datetime.now()
            else:
                request.status = RequestStatus.FAILED
            
            await self.update_request_status(request)
            
            # Notify user
            await self.notify_user(
                request.user_id,
                f"Request {request.status.value.title()}",
                f"Your request for '{request.title}' has been {request.status.value}.",
                NotificationType.EMAIL,
                {'request': asdict(request)}
            )
            
        except Exception as e:
            logger.error(f"Error processing approved request: {e}")
            request.status = RequestStatus.FAILED
            await self.update_request_status(request)
    
    async def save_request(self, request: UserRequest) -> int:
        """Save request to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO requests 
            (user_id, request_type, title, media_type, status, priority, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            request.user_id,
            request.request_type,
            request.title,
            request.media_type,
            request.status.value,
            request.priority,
            request.notes
        ))
        
        request_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return request_id
    
    async def update_request_status(self, request: UserRequest):
        """Update request status in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE requests 
            SET status = ?, completed_date = ?
            WHERE user_id = ? AND title = ? AND status != ?
        ''', (
            request.status.value,
            request.completed_date,
            request.user_id,
            request.title,
            RequestStatus.COMPLETED.value
        ))
        
        conn.commit()
        conn.close()
    
    async def notify_user(self, user_id: str, title: str, message: str, 
                         notification_type: NotificationType, data: Optional[Dict] = None):
        """Send notification to user"""
        try:
            notification = Notification(
                user_id=user_id,
                type=notification_type,
                title=title,
                message=message,
                data=data
            )
            
            # Save notification
            await self.save_notification(notification)
            
            # Send notification
            if notification_type in self.notification_handlers:
                await self.notification_handlers[notification_type](notification)
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    async def save_notification(self, notification: Notification):
        """Save notification to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO notifications 
            (user_id, type, title, message, priority, data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            notification.user_id,
            notification.type.value,
            notification.title,
            notification.message,
            notification.priority,
            json.dumps(notification.data) if notification.data else None
        ))
        
        conn.commit()
        conn.close()
    
    async def send_email_notification(self, notification: Notification):
        """Send email notification"""
        try:
            # Get user email
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT email FROM users WHERE id = ?', (notification.user_id,))
            result = cursor.fetchone()
            conn.close()
            
            if not result or not result[0]:
                logger.warning(f"No email address for user {notification.user_id}")
                return
            
            email = result[0]
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config['notifications']['email']['from_address']
            msg['To'] = email
            msg['Subject'] = notification.title
            
            # Add body
            body = notification.message
            if notification.data:
                body += f"\n\nAdditional Details:\n{json.dumps(notification.data, indent=2)}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.config['notifications']['email']['smtp_host'], 
                             self.config['notifications']['email']['smtp_port']) as server:
                server.starttls()
                server.login(
                    self.config['notifications']['email']['smtp_user'],
                    self.config['notifications']['email']['smtp_password']
                )
                server.send_message(msg)
            
            # Mark as sent
            await self.mark_notification_sent(notification)
            logger.info(f"Email sent to {email}")
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
    
    async def send_webhook_notification(self, notification: Notification):
        """Send webhook notification"""
        try:
            payload = {
                'user_id': notification.user_id,
                'title': notification.title,
                'message': notification.message,
                'priority': notification.priority,
                'timestamp': datetime.now().isoformat(),
                'data': notification.data
            }
            
            async with aiohttp.ClientSession() as session:
                for endpoint in self.config['notifications']['webhook']['endpoints']:
                    try:
                        async with session.post(endpoint, json=payload) as response:
                            if response.status == 200:
                                logger.info(f"Webhook sent to {endpoint}")
                    except Exception as e:
                        logger.error(f"Error sending webhook to {endpoint}: {e}")
            
            await self.mark_notification_sent(notification)
            
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
    
    async def send_websocket_notification(self, notification: Notification):
        """Send websocket notification to connected clients"""
        try:
            message = json.dumps({
                'type': 'notification',
                'user_id': notification.user_id,
                'title': notification.title,
                'message': notification.message,
                'priority': notification.priority,
                'data': notification.data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Send to all connected clients for this user
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    if hasattr(client, 'user_id') and client.user_id == notification.user_id:
                        await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
            
            await self.mark_notification_sent(notification)
            
        except Exception as e:
            logger.error(f"Error sending websocket notification: {e}")
    
    async def mark_notification_sent(self, notification: Notification):
        """Mark notification as sent in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE notifications 
            SET sent = 1, sent_date = CURRENT_TIMESTAMP
            WHERE user_id = ? AND title = ? AND sent = 0
        ''', (notification.user_id, notification.title))
        
        conn.commit()
        conn.close()
    
    async def log_activity(self, user_id: str, action: str, details: Optional[Dict] = None,
                          ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """Log user activity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO activity_log (user_id, action, details, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            action,
            json.dumps(details) if details else None,
            ip_address,
            user_agent
        ))
        
        conn.commit()
        conn.close()
    
    async def track_metric(self, metric_name: str, metric_value: float,
                          dimension1: Optional[str] = None, dimension2: Optional[str] = None):
        """Track analytics metric"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analytics (metric_name, metric_value, dimension1, dimension2)
            VALUES (?, ?, ?, ?)
        ''', (metric_name, metric_value, dimension1, dimension2))
        
        conn.commit()
        conn.close()
    
    async def get_user_progress(self, user_id: str) -> Dict:
        """Get user's request progress"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get request statistics
        cursor.execute('''
            SELECT status, COUNT(*) as count
            FROM requests
            WHERE user_id = ?
            GROUP BY status
        ''', (user_id,))
        
        status_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get recent requests
        cursor.execute('''
            SELECT title, media_type, status, requested_date, completed_date
            FROM requests
            WHERE user_id = ?
            ORDER BY requested_date DESC
            LIMIT 10
        ''', (user_id,))
        
        recent_requests = []
        for row in cursor.fetchall():
            recent_requests.append({
                'title': row[0],
                'media_type': row[1],
                'status': row[2],
                'requested_date': row[3],
                'completed_date': row[4]
            })
        
        conn.close()
        
        return {
            'status_summary': status_counts,
            'recent_requests': recent_requests,
            'total_requests': sum(status_counts.values())
        }
    
    async def generate_analytics_report(self, period: str = 'weekly') -> Dict:
        """Generate analytics report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate date range
        if period == 'daily':
            date_filter = "datetime('now', '-1 day')"
        elif period == 'weekly':
            date_filter = "datetime('now', '-7 days')"
        elif period == 'monthly':
            date_filter = "datetime('now', '-30 days')"
        else:
            date_filter = "datetime('now', '-7 days')"
        
        # User activity
        cursor.execute(f'''
            SELECT COUNT(DISTINCT user_id) as active_users,
                   COUNT(*) as total_requests
            FROM requests
            WHERE requested_date > {date_filter}
        ''')
        
        activity = cursor.fetchone()
        
        # Popular content
        cursor.execute(f'''
            SELECT title, media_type, COUNT(*) as request_count
            FROM requests
            WHERE requested_date > {date_filter}
            GROUP BY title, media_type
            ORDER BY request_count DESC
            LIMIT 10
        ''')
        
        popular_content = []
        for row in cursor.fetchall():
            popular_content.append({
                'title': row[0],
                'media_type': row[1],
                'requests': row[2]
            })
        
        # Request completion rate
        cursor.execute(f'''
            SELECT 
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                COUNT(*) as total
            FROM requests
            WHERE requested_date > {date_filter}
        ''')
        
        completion = cursor.fetchone()
        completion_rate = (completion[0] / completion[1] * 100) if completion[1] > 0 else 0
        
        # Average processing time
        cursor.execute(f'''
            SELECT AVG(julianday(completed_date) - julianday(requested_date)) * 24 as avg_hours
            FROM requests
            WHERE status = 'completed' 
            AND requested_date > {date_filter}
            AND completed_date IS NOT NULL
        ''')
        
        avg_processing_time = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'period': period,
            'active_users': activity[0],
            'total_requests': activity[1],
            'popular_content': popular_content,
            'completion_rate': round(completion_rate, 2),
            'average_processing_hours': round(avg_processing_time, 2)
        }
    
    async def create_usage_chart(self, user_id: Optional[str] = None) -> str:
        """Create usage chart and return as base64 encoded image"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get daily request counts for last 30 days
        if user_id:
            cursor.execute('''
                SELECT DATE(requested_date) as date, COUNT(*) as count
                FROM requests
                WHERE user_id = ?
                AND requested_date > datetime('now', '-30 days')
                GROUP BY DATE(requested_date)
                ORDER BY date
            ''', (user_id,))
        else:
            cursor.execute('''
                SELECT DATE(requested_date) as date, COUNT(*) as count
                FROM requests
                WHERE requested_date > datetime('now', '-30 days')
                GROUP BY DATE(requested_date)
                ORDER BY date
            ''')
        
        data = cursor.fetchall()
        conn.close()
        
        if not data:
            return ""
        
        # Create plot
        dates = [row[0] for row in data]
        counts = [row[1] for row in data]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, counts, marker='o')
        plt.title('Media Requests - Last 30 Days')
        plt.xlabel('Date')
        plt.ylabel('Number of Requests')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    async def websocket_handler(self, websocket, path):
        """Handle websocket connections"""
        try:
            # Authenticate user
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            # Verify JWT token
            try:
                payload = jwt.decode(
                    auth_data['token'],
                    self.config['jwt_secret'],
                    algorithms=['HS256']
                )
                user_id = payload['user_id']
            except jwt.InvalidTokenError:
                await websocket.send(json.dumps({'error': 'Invalid token'}))
                return
            
            # Add to connected clients
            websocket.user_id = user_id
            self.websocket_clients.add(websocket)
            
            # Send welcome message
            await websocket.send(json.dumps({
                'type': 'connected',
                'message': 'Connected to media server'
            }))
            
            # Handle messages
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'ping':
                    await websocket.send(json.dumps({'type': 'pong'}))
                elif data['type'] == 'get_progress':
                    progress = await self.get_user_progress(user_id)
                    await websocket.send(json.dumps({
                        'type': 'progress',
                        'data': progress
                    }))
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.websocket_clients.discard(websocket)
    
    async def start_websocket_server(self):
        """Start websocket server"""
        server = await websockets.serve(
            self.websocket_handler,
            'localhost',
            self.config['websocket_port']
        )
        logger.info(f"WebSocket server started on port {self.config['websocket_port']}")
        await server.wait_closed()
    
    async def add_to_radarr(self, request: UserRequest) -> bool:
        """Add movie to Radarr (placeholder)"""
        # This would integrate with Radarr API
        await self.track_metric('radarr_request', 1, request.media_type, request.title)
        return True
    
    async def add_to_sonarr(self, request: UserRequest) -> bool:
        """Add TV show to Sonarr (placeholder)"""
        # This would integrate with Sonarr API
        await self.track_metric('sonarr_request', 1, request.media_type, request.title)
        return True

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='User Experience Automation')
    parser.add_argument('--config', default='/config/user_experience.json', help='Config file path')
    parser.add_argument('--server', action='store_true', help='Start WebSocket server')
    parser.add_argument('--report', choices=['daily', 'weekly', 'monthly'], help='Generate analytics report')
    parser.add_argument('--test-request', help='Create test request with title')
    parser.add_argument('--user', default='test-user', help='User ID for testing')
    
    args = parser.parse_args()
    
    manager = UserExperienceManager(args.config)
    
    if args.server:
        await manager.start_websocket_server()
    elif args.report:
        report = await manager.generate_analytics_report(args.report)
        print(json.dumps(report, indent=2))
    elif args.test_request:
        result = await manager.create_request(args.user, {
            'title': args.test_request,
            'media_type': 'movie',
            'username': args.user
        })
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())