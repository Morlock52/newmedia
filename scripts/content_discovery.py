#!/usr/bin/env python3
"""
Content Discovery Automation
Handles automated searches, quality preferences, release monitoring, and duplicate handling
"""

import os
import json
import logging
import requests
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import hashlib
from typing import List, Dict, Optional
import re
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/content_discovery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    MOVIE = "movie"
    TV_SHOW = "tv"
    MUSIC = "music"
    BOOK = "book"

class Quality(Enum):
    SD = "SD"
    HD_720 = "720p"
    HD_1080 = "1080p"
    UHD_4K = "2160p"
    
    @property
    def min_size_mb(self):
        """Minimum expected file size in MB for quality validation"""
        sizes = {
            self.SD: 300,
            self.HD_720: 700,
            self.HD_1080: 1500,
            self.UHD_4K: 4000
        }
        return sizes.get(self, 0)

@dataclass
class ContentRequest:
    title: str
    content_type: ContentType
    year: Optional[int] = None
    season: Optional[int] = None
    episode: Optional[int] = None
    quality: Quality = Quality.HD_1080
    priority: int = 5
    requested_date: datetime = None
    fulfilled: bool = False
    
    def __post_init__(self):
        if self.requested_date is None:
            self.requested_date = datetime.now()

class ContentDiscovery:
    def __init__(self, config_path='/config/content_discovery.json'):
        self.config = self.load_config(config_path)
        self.db_path = self.config.get('database_path', '/config/content_discovery.db')
        self.init_database()
        self.api_clients = self.init_api_clients()
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        default_config = {
            'sonarr_url': 'http://sonarr:8989',
            'radarr_url': 'http://radarr:7878',
            'lidarr_url': 'http://lidarr:8686',
            'readarr_url': 'http://readarr:8787',
            'prowlarr_url': 'http://prowlarr:9696',
            'overseerr_url': 'http://overseerr:5055',
            'quality_profiles': {
                'movie': 'HD-1080p',
                'tv': 'HD-1080p',
                'music': 'FLAC',
                'book': 'EPUB'
            },
            'monitoring_interval': 30,  # minutes
            'duplicate_threshold': 0.95,  # similarity threshold
            'auto_approve_threshold': 7.0,  # IMDB/TMDB rating
            'preferred_release_groups': ['REMUX', 'FLUX', 'SPARKS'],
            'avoided_terms': ['CAM', 'TS', 'HDCAM'],
            'size_limits': {
                'movie_max_gb': 50,
                'episode_max_gb': 10
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        return default_config
    
    def init_database(self):
        """Initialize SQLite database for tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Content requests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content_type TEXT NOT NULL,
                year INTEGER,
                season INTEGER,
                episode INTEGER,
                quality TEXT,
                priority INTEGER DEFAULT 5,
                requested_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                fulfilled BOOLEAN DEFAULT 0,
                fulfilled_date TIMESTAMP,
                file_path TEXT,
                tmdb_id INTEGER,
                imdb_id TEXT
            )
        ''')
        
        # Search history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_term TEXT NOT NULL,
                content_type TEXT,
                search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                results_count INTEGER,
                selected_result TEXT
            )
        ''')
        
        # Release monitoring table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS release_monitoring (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content_type TEXT NOT NULL,
                expected_date DATE,
                monitored BOOLEAN DEFAULT 1,
                notified BOOLEAN DEFAULT 0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Duplicate tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS duplicate_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                title TEXT,
                quality TEXT,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def init_api_clients(self):
        """Initialize API clients for various services"""
        clients = {}
        
        # Initialize if API keys are available
        if self.config.get('sonarr_api_key'):
            clients['sonarr'] = {
                'url': self.config['sonarr_url'],
                'headers': {'X-Api-Key': self.config['sonarr_api_key']}
            }
        
        if self.config.get('radarr_api_key'):
            clients['radarr'] = {
                'url': self.config['radarr_url'],
                'headers': {'X-Api-Key': self.config['radarr_api_key']}
            }
            
        if self.config.get('overseerr_api_key'):
            clients['overseerr'] = {
                'url': self.config['overseerr_url'],
                'headers': {'X-Api-Key': self.config['overseerr_api_key']}
            }
        
        return clients
    
    def search_content(self, query: str, content_type: ContentType, year: Optional[int] = None):
        """Search for content across configured services"""
        results = []
        
        # Search via Overseerr (if available)
        if 'overseerr' in self.api_clients:
            overseerr_results = self.search_overseerr(query, content_type, year)
            results.extend(overseerr_results)
        
        # Direct API searches
        if content_type == ContentType.MOVIE and 'radarr' in self.api_clients:
            radarr_results = self.search_radarr(query, year)
            results.extend(radarr_results)
        elif content_type == ContentType.TV_SHOW and 'sonarr' in self.api_clients:
            sonarr_results = self.search_sonarr(query)
            results.extend(sonarr_results)
        
        # Log search
        self.log_search(query, content_type, len(results))
        
        # Rank results by quality and preferences
        ranked_results = self.rank_search_results(results)
        
        return ranked_results
    
    def search_overseerr(self, query: str, content_type: ContentType, year: Optional[int] = None):
        """Search content via Overseerr API"""
        try:
            client = self.api_clients['overseerr']
            search_type = 'movie' if content_type == ContentType.MOVIE else 'tv'
            
            params = {
                'query': query,
                'page': 1,
                'language': 'en'
            }
            
            response = requests.get(
                f"{client['url']}/api/v1/search/{search_type}",
                headers=client['headers'],
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('results', []):
                    # Filter by year if specified
                    if year:
                        item_year = int(item.get('release_date', '0000')[:4]) if content_type == ContentType.MOVIE else int(item.get('first_air_date', '0000')[:4])
                        if abs(item_year - year) > 1:
                            continue
                    
                    results.append({
                        'title': item.get('title') or item.get('name'),
                        'year': item.get('release_date', '')[:4] or item.get('first_air_date', '')[:4],
                        'overview': item.get('overview'),
                        'tmdb_id': item.get('id'),
                        'poster_path': item.get('poster_path'),
                        'vote_average': item.get('vote_average', 0),
                        'media_type': search_type,
                        'source': 'overseerr'
                    })
                
                return results
            
        except Exception as e:
            logger.error(f"Error searching Overseerr: {e}")
        
        return []
    
    def search_radarr(self, query: str, year: Optional[int] = None):
        """Search movies via Radarr API"""
        try:
            client = self.api_clients['radarr']
            
            params = {'term': query}
            
            response = requests.get(
                f"{client['url']}/api/v3/movie/lookup",
                headers=client['headers'],
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data:
                    # Filter by year if specified
                    if year and item.get('year') != year:
                        continue
                    
                    results.append({
                        'title': item.get('title'),
                        'year': item.get('year'),
                        'overview': item.get('overview'),
                        'tmdb_id': item.get('tmdbId'),
                        'imdb_id': item.get('imdbId'),
                        'vote_average': item.get('ratings', {}).get('imdb', {}).get('value', 0),
                        'media_type': 'movie',
                        'source': 'radarr',
                        'has_file': item.get('hasFile', False)
                    })
                
                return results
            
        except Exception as e:
            logger.error(f"Error searching Radarr: {e}")
        
        return []
    
    def search_sonarr(self, query: str):
        """Search TV shows via Sonarr API"""
        try:
            client = self.api_clients['sonarr']
            
            params = {'term': query}
            
            response = requests.get(
                f"{client['url']}/api/v3/series/lookup",
                headers=client['headers'],
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data:
                    results.append({
                        'title': item.get('title'),
                        'year': item.get('year'),
                        'overview': item.get('overview'),
                        'tvdb_id': item.get('tvdbId'),
                        'imdb_id': item.get('imdbId'),
                        'vote_average': item.get('ratings', {}).get('value', 0),
                        'media_type': 'tv',
                        'source': 'sonarr',
                        'status': item.get('status'),
                        'seasons': item.get('seasons', [])
                    })
                
                return results
            
        except Exception as e:
            logger.error(f"Error searching Sonarr: {e}")
        
        return []
    
    def rank_search_results(self, results: List[Dict]) -> List[Dict]:
        """Rank search results based on quality preferences and ratings"""
        for result in results:
            score = 0
            
            # Rating score (0-10 points)
            score += result.get('vote_average', 0)
            
            # Availability bonus
            if result.get('has_file'):
                score += 2
            
            # Source preference
            source_scores = {'overseerr': 3, 'radarr': 2, 'sonarr': 2}
            score += source_scores.get(result.get('source'), 0)
            
            # Year recency bonus (for TV shows)
            if result.get('media_type') == 'tv' and result.get('status') == 'continuing':
                score += 2
            
            result['search_score'] = score
        
        # Sort by score
        return sorted(results, key=lambda x: x['search_score'], reverse=True)
    
    def add_content_request(self, request: ContentRequest) -> bool:
        """Add a new content request to the queue"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO content_requests 
                (title, content_type, year, season, episode, quality, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                request.title,
                request.content_type.value,
                request.year,
                request.season,
                request.episode,
                request.quality.value,
                request.priority
            ))
            
            conn.commit()
            conn.close()
            
            # Trigger immediate search
            self.process_content_request(request)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding content request: {e}")
            return False
    
    def process_content_request(self, request: ContentRequest):
        """Process a content request and add to appropriate service"""
        try:
            # Search for content
            results = self.search_content(request.title, request.content_type, request.year)
            
            if not results:
                logger.warning(f"No results found for: {request.title}")
                return False
            
            # Use top result
            best_match = results[0]
            
            # Check if already exists
            if best_match.get('has_file'):
                logger.info(f"Content already available: {request.title}")
                self.mark_request_fulfilled(request, "Already available")
                return True
            
            # Add to appropriate service
            if request.content_type == ContentType.MOVIE:
                return self.add_to_radarr(best_match, request.quality)
            elif request.content_type == ContentType.TV_SHOW:
                return self.add_to_sonarr(best_match, request.quality, request.season)
            
        except Exception as e:
            logger.error(f"Error processing content request: {e}")
            return False
    
    def add_to_radarr(self, movie_info: Dict, quality: Quality) -> bool:
        """Add movie to Radarr"""
        try:
            client = self.api_clients['radarr']
            
            # Get quality profile ID
            quality_profile = self.get_radarr_quality_profile(quality)
            
            # Get root folder
            root_folder = self.get_radarr_root_folder()
            
            data = {
                'title': movie_info['title'],
                'year': movie_info.get('year'),
                'tmdbId': movie_info.get('tmdb_id'),
                'imdbId': movie_info.get('imdb_id'),
                'qualityProfileId': quality_profile,
                'rootFolderPath': root_folder,
                'monitored': True,
                'addOptions': {
                    'searchForMovie': True
                }
            }
            
            response = requests.post(
                f"{client['url']}/api/v3/movie",
                headers=client['headers'],
                json=data
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Successfully added to Radarr: {movie_info['title']}")
                return True
            else:
                logger.error(f"Failed to add to Radarr: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding to Radarr: {e}")
            return False
    
    def add_to_sonarr(self, show_info: Dict, quality: Quality, season: Optional[int] = None) -> bool:
        """Add TV show to Sonarr"""
        try:
            client = self.api_clients['sonarr']
            
            # Get quality profile ID
            quality_profile = self.get_sonarr_quality_profile(quality)
            
            # Get root folder
            root_folder = self.get_sonarr_root_folder()
            
            # Configure season monitoring
            seasons = []
            for s in show_info.get('seasons', []):
                monitored = True if season is None or s['seasonNumber'] == season else False
                seasons.append({
                    'seasonNumber': s['seasonNumber'],
                    'monitored': monitored
                })
            
            data = {
                'title': show_info['title'],
                'tvdbId': show_info.get('tvdb_id'),
                'imdbId': show_info.get('imdb_id'),
                'qualityProfileId': quality_profile,
                'rootFolderPath': root_folder,
                'seasonFolder': True,
                'monitored': True,
                'seasons': seasons,
                'addOptions': {
                    'searchForMissingEpisodes': True
                }
            }
            
            response = requests.post(
                f"{client['url']}/api/v3/series",
                headers=client['headers'],
                json=data
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Successfully added to Sonarr: {show_info['title']}")
                return True
            else:
                logger.error(f"Failed to add to Sonarr: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding to Sonarr: {e}")
            return False
    
    def monitor_releases(self):
        """Monitor for new releases and notify when available"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get monitored releases
            cursor.execute('''
                SELECT * FROM release_monitoring
                WHERE monitored = 1 AND notified = 0
                AND expected_date <= date('now')
            ''')
            
            releases = cursor.fetchall()
            
            for release in releases:
                # Check if now available
                results = self.search_content(
                    release[1],  # title
                    ContentType(release[2]),  # content_type
                    None
                )
                
                if results and results[0].get('has_file'):
                    # Send notification
                    self.send_notification(
                        f"New Release Available",
                        f"{release[1]} is now available!"
                    )
                    
                    # Mark as notified
                    cursor.execute('''
                        UPDATE release_monitoring
                        SET notified = 1
                        WHERE id = ?
                    ''', (release[0],))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error monitoring releases: {e}")
    
    def check_duplicates(self, file_path: str) -> Optional[str]:
        """Check if file is a duplicate based on hash"""
        try:
            # Calculate file hash
            file_hash = self.calculate_file_hash(file_path)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for existing hash
            cursor.execute('''
                SELECT file_path FROM duplicate_tracking
                WHERE file_hash = ?
            ''', (file_hash,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return result[0]  # Return path of duplicate
            
            # Add to tracking
            self.add_to_duplicate_tracking(file_path, file_hash)
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking duplicates: {e}")
            return None
    
    def calculate_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def add_to_duplicate_tracking(self, file_path: str, file_hash: str):
        """Add file to duplicate tracking database"""
        try:
            file_size = os.path.getsize(file_path)
            
            # Extract title from filename
            title = Path(file_path).stem
            
            # Detect quality
            quality = self.detect_quality(file_path)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR IGNORE INTO duplicate_tracking
                (file_hash, file_path, file_size, title, quality)
                VALUES (?, ?, ?, ?, ?)
            ''', (file_hash, file_path, file_size, title, quality))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error adding to duplicate tracking: {e}")
    
    def detect_quality(self, file_path: str) -> str:
        """Detect quality from filename"""
        filename = Path(file_path).name.lower()
        
        if '2160p' in filename or '4k' in filename:
            return Quality.UHD_4K.value
        elif '1080p' in filename:
            return Quality.HD_1080.value
        elif '720p' in filename:
            return Quality.HD_720.value
        else:
            return Quality.SD.value
    
    def validate_quality(self, file_path: str, expected_quality: Quality) -> bool:
        """Validate file meets quality expectations"""
        try:
            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            if file_size_mb < expected_quality.min_size_mb:
                logger.warning(f"File size too small for {expected_quality.value}: {file_path}")
                return False
            
            # Check filename
            detected_quality = self.detect_quality(file_path)
            if detected_quality != expected_quality.value:
                logger.warning(f"Quality mismatch for {file_path}: expected {expected_quality.value}, got {detected_quality}")
                return False
            
            # Check for avoided terms
            filename = Path(file_path).name.lower()
            for term in self.config['avoided_terms']:
                if term.lower() in filename:
                    logger.warning(f"Avoided term '{term}' found in: {file_path}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating quality: {e}")
            return False
    
    def get_radarr_quality_profile(self, quality: Quality) -> int:
        """Get Radarr quality profile ID"""
        try:
            client = self.api_clients['radarr']
            
            response = requests.get(
                f"{client['url']}/api/v3/qualityprofile",
                headers=client['headers']
            )
            
            if response.status_code == 200:
                profiles = response.json()
                
                # Map quality to profile name
                quality_map = {
                    Quality.SD: "SD",
                    Quality.HD_720: "HD-720p",
                    Quality.HD_1080: "HD-1080p",
                    Quality.UHD_4K: "Ultra-HD"
                }
                
                profile_name = quality_map.get(quality, "HD-1080p")
                
                for profile in profiles:
                    if profile['name'] == profile_name:
                        return profile['id']
                
                # Return first profile as fallback
                return profiles[0]['id'] if profiles else 1
            
        except Exception as e:
            logger.error(f"Error getting quality profile: {e}")
        
        return 1  # Default profile
    
    def get_radarr_root_folder(self) -> str:
        """Get Radarr root folder path"""
        try:
            client = self.api_clients['radarr']
            
            response = requests.get(
                f"{client['url']}/api/v3/rootfolder",
                headers=client['headers']
            )
            
            if response.status_code == 200:
                folders = response.json()
                return folders[0]['path'] if folders else "/movies"
            
        except Exception as e:
            logger.error(f"Error getting root folder: {e}")
        
        return "/movies"
    
    def get_sonarr_quality_profile(self, quality: Quality) -> int:
        """Get Sonarr quality profile ID"""
        # Similar to get_radarr_quality_profile
        return 1
    
    def get_sonarr_root_folder(self) -> str:
        """Get Sonarr root folder path"""
        # Similar to get_radarr_root_folder
        return "/tv"
    
    def log_search(self, query: str, content_type: ContentType, results_count: int):
        """Log search history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO search_history
                (search_term, content_type, results_count)
                VALUES (?, ?, ?)
            ''', (query, content_type.value, results_count))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging search: {e}")
    
    def mark_request_fulfilled(self, request: ContentRequest, file_path: str):
        """Mark a content request as fulfilled"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE content_requests
                SET fulfilled = 1,
                    fulfilled_date = CURRENT_TIMESTAMP,
                    file_path = ?
                WHERE title = ? AND content_type = ?
            ''', (file_path, request.title, request.content_type.value))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error marking request fulfilled: {e}")
    
    def send_notification(self, title: str, message: str):
        """Send notification via configured service"""
        try:
            # Implement notification logic here
            # Could use Apprise, Discord, Telegram, etc.
            logger.info(f"Notification: {title} - {message}")
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def run_scheduled_tasks(self):
        """Run scheduled monitoring tasks"""
        schedule.every(self.config['monitoring_interval']).minutes.do(self.monitor_releases)
        schedule.every(60).minutes.do(self.process_pending_requests)
        schedule.every(24).hours.do(self.cleanup_old_data)
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def process_pending_requests(self):
        """Process any pending content requests"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM content_requests
                WHERE fulfilled = 0
                ORDER BY priority DESC, requested_date ASC
                LIMIT 10
            ''')
            
            requests = cursor.fetchall()
            conn.close()
            
            for req_data in requests:
                request = ContentRequest(
                    title=req_data[1],
                    content_type=ContentType(req_data[2]),
                    year=req_data[3],
                    season=req_data[4],
                    episode=req_data[5],
                    quality=Quality(req_data[6]),
                    priority=req_data[7]
                )
                
                self.process_content_request(request)
                
        except Exception as e:
            logger.error(f"Error processing pending requests: {e}")
    
    def cleanup_old_data(self):
        """Clean up old data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clean old search history (>30 days)
            cursor.execute('''
                DELETE FROM search_history
                WHERE search_date < datetime('now', '-30 days')
            ''')
            
            # Clean fulfilled requests (>90 days)
            cursor.execute('''
                DELETE FROM content_requests
                WHERE fulfilled = 1
                AND fulfilled_date < datetime('now', '-90 days')
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Cleaned up old data from database")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Content Discovery Automation')
    parser.add_argument('--config', default='/config/content_discovery.json', help='Config file path')
    parser.add_argument('--search', help='Search for content')
    parser.add_argument('--type', choices=['movie', 'tv', 'music', 'book'], help='Content type')
    parser.add_argument('--year', type=int, help='Year of release')
    parser.add_argument('--request', help='Add content request')
    parser.add_argument('--monitor', action='store_true', help='Run monitoring daemon')
    
    args = parser.parse_args()
    
    discovery = ContentDiscovery(args.config)
    
    if args.search:
        content_type = ContentType(args.type) if args.type else ContentType.MOVIE
        results = discovery.search_content(args.search, content_type, args.year)
        
        for i, result in enumerate(results[:10]):
            print(f"{i+1}. {result['title']} ({result.get('year', 'N/A')}) - Score: {result.get('search_score', 0):.1f}")
            print(f"   {result.get('overview', 'No description')[:100]}...")
            print()
    
    elif args.request:
        content_type = ContentType(args.type) if args.type else ContentType.MOVIE
        request = ContentRequest(
            title=args.request,
            content_type=content_type,
            year=args.year
        )
        
        if discovery.add_content_request(request):
            print(f"Successfully added request for: {args.request}")
        else:
            print(f"Failed to add request for: {args.request}")
    
    elif args.monitor:
        print("Starting content discovery monitoring...")
        discovery.run_scheduled_tasks()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()