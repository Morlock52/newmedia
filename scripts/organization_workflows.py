#!/usr/bin/env python3
"""
Media Organization Workflows
Handles file naming conventions, directory structures, media categorization, and collection management
"""

import os
import re
import shutil
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple
import requests
from dataclasses import dataclass
from enum import Enum
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/organization_workflows.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MediaType(Enum):
    MOVIE = "movie"
    TV_SHOW = "tv_show"
    MUSIC = "music"
    BOOK = "book"
    PHOTO = "photo"
    UNKNOWN = "unknown"

@dataclass
class MediaFile:
    path: Path
    media_type: MediaType
    title: str
    year: Optional[int] = None
    season: Optional[int] = None
    episode: Optional[int] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    track: Optional[int] = None
    quality: Optional[str] = None
    source: Optional[str] = None
    audio_codec: Optional[str] = None
    video_codec: Optional[str] = None
    release_group: Optional[str] = None

class MediaOrganizer:
    def __init__(self, config_path='/config/organization.json'):
        self.config = self.load_config(config_path)
        self.patterns = self.compile_patterns()
        self.db_path = self.config.get('database_path', '/config/media_organization.db')
        self.init_database()
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        default_config = {
            'media_root': '/media',
            'structures': {
                'movie': '{media_root}/Movies/{title} ({year})/{title} ({year}) - {quality}.{ext}',
                'tv_show': '{media_root}/TV Shows/{title}/Season {season:02d}/{title} - S{season:02d}E{episode:02d} - {episode_title}.{ext}',
                'music': '{media_root}/Music/{artist}/{album}/{track:02d} - {title}.{ext}',
                'book': '{media_root}/Books/{author}/{title}.{ext}',
                'photo': '{media_root}/Photos/{year}/{month}/{date} - {time}.{ext}'
            },
            'naming_conventions': {
                'replace_spaces': False,
                'case_style': 'title',  # title, lower, upper, original
                'remove_special_chars': True,
                'max_length': 255
            },
            'quality_tags': ['2160p', '1080p', '720p', '480p', 'REMUX', 'BluRay', 'WEB-DL', 'WEBRip', 'HDTV'],
            'audio_tags': ['DTS-HD', 'TrueHD', 'Atmos', 'DTS', 'AC3', 'AAC'],
            'video_tags': ['x265', 'x264', 'HEVC', 'AVC'],
            'collections': {
                'enable': True,
                'auto_create': True,
                'min_items': 3
            },
            'duplicate_handling': 'rename',  # rename, skip, replace
            'preserve_original': False,
            'scan_interval': 60  # minutes
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        return default_config
    
    def init_database(self):
        """Initialize SQLite database for organization tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Media files table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS media_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_path TEXT NOT NULL,
                organized_path TEXT,
                media_type TEXT NOT NULL,
                title TEXT,
                year INTEGER,
                season INTEGER,
                episode INTEGER,
                quality TEXT,
                file_size INTEGER,
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                date_organized TIMESTAMP,
                collection_id INTEGER
            )
        ''')
        
        # Collections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                media_type TEXT NOT NULL,
                item_count INTEGER DEFAULT 0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tmdb_collection_id INTEGER
            )
        ''')
        
        # Organization rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS organization_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                media_type TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                active BOOLEAN DEFAULT 1,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def compile_patterns(self):
        """Compile regex patterns for media detection"""
        patterns = {
            'movie': [
                # Movie.Title.2020.1080p.BluRay.x264-GROUP
                re.compile(r'^(?P<title>.+?)\.(?P<year>\d{4})\.(?P<quality>\d{3,4}p)\.(?P<source>[\w-]+)\.(?P<codec>[\w-]+)(?:-(?P<group>[\w]+))?'),
                # Movie Title (2020) [1080p]
                re.compile(r'^(?P<title>.+?)\s*\((?P<year>\d{4})\)\s*\[(?P<quality>\d{3,4}p)\]'),
                # Movie.Title.2020.1080p
                re.compile(r'^(?P<title>.+?)\.(?P<year>\d{4})\.(?P<quality>\d{3,4}p)'),
                # Movie Title 2020
                re.compile(r'^(?P<title>.+?)\s+(?P<year>\d{4})$')
            ],
            'tv_show': [
                # Show.Name.S01E01.1080p.WEB-DL
                re.compile(r'^(?P<title>.+?)\.S(?P<season>\d{2})E(?P<episode>\d{2})\.(?P<quality>\d{3,4}p)?\.?(?P<source>[\w-]+)?'),
                # Show Name - 1x01 - Episode Title
                re.compile(r'^(?P<title>.+?)\s*-\s*(?P<season>\d+)x(?P<episode>\d{2})\s*-?\s*(?P<episode_title>.+)?'),
                # Show.Name.2020.S01E01
                re.compile(r'^(?P<title>.+?)\.(?P<year>\d{4})\.S(?P<season>\d{2})E(?P<episode>\d{2})'),
                # Show Name S01E01
                re.compile(r'^(?P<title>.+?)\s+S(?P<season>\d{2})E(?P<episode>\d{2})')
            ],
            'music': [
                # Artist - Album - 01 - Track Title
                re.compile(r'^(?P<artist>.+?)\s*-\s*(?P<album>.+?)\s*-\s*(?P<track>\d{2})\s*-\s*(?P<title>.+)'),
                # 01. Artist - Track Title
                re.compile(r'^(?P<track>\d{2})\.?\s*(?P<artist>.+?)\s*-\s*(?P<title>.+)'),
                # Artist - Track Title
                re.compile(r'^(?P<artist>.+?)\s*-\s*(?P<title>.+)')
            ]
        }
        return patterns
    
    def detect_media_type(self, file_path: Path) -> MediaType:
        """Detect media type based on file extension and content"""
        ext = file_path.suffix.lower()
        
        # Extension-based detection
        video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
        audio_exts = {'.mp3', '.flac', '.aac', '.ogg', '.wav', '.m4a', '.opus', '.wma'}
        book_exts = {'.epub', '.pdf', '.mobi', '.azw3', '.fb2', '.txt', '.cbr', '.cbz'}
        photo_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.raw', '.heic'}
        
        if ext in video_exts:
            # Further distinguish between movie and TV show
            if self.is_tv_show(file_path):
                return MediaType.TV_SHOW
            return MediaType.MOVIE
        elif ext in audio_exts:
            return MediaType.MUSIC
        elif ext in book_exts:
            return MediaType.BOOK
        elif ext in photo_exts:
            return MediaType.PHOTO
        else:
            return MediaType.UNKNOWN
    
    def is_tv_show(self, file_path: Path) -> bool:
        """Check if video file is a TV show"""
        filename = file_path.stem
        
        # Check for TV show patterns
        tv_patterns = [
            r'S\d{2}E\d{2}',  # S01E01
            r'\d+x\d{2}',     # 1x01
            r'Season\s*\d+',   # Season 1
            r'Episode\s*\d+',  # Episode 1
        ]
        
        for pattern in tv_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return True
        
        # Check parent directory
        parent = file_path.parent.name
        if re.search(r'Season\s*\d+', parent, re.IGNORECASE):
            return True
        
        return False
    
    def parse_filename(self, file_path: Path) -> MediaFile:
        """Parse filename to extract metadata"""
        media_type = self.detect_media_type(file_path)
        filename = file_path.stem
        
        # Initialize MediaFile
        media_file = MediaFile(
            path=file_path,
            media_type=media_type,
            title=filename  # Default to filename
        )
        
        # Try to match patterns
        if media_type in self.patterns:
            for pattern in self.patterns[media_type]:
                match = pattern.match(filename)
                if match:
                    data = match.groupdict()
                    
                    # Clean up title
                    if 'title' in data:
                        media_file.title = self.clean_title(data['title'])
                    
                    # Extract other metadata
                    media_file.year = int(data.get('year')) if data.get('year') else None
                    media_file.season = int(data.get('season')) if data.get('season') else None
                    media_file.episode = int(data.get('episode')) if data.get('episode') else None
                    media_file.quality = data.get('quality')
                    media_file.source = data.get('source')
                    media_file.release_group = data.get('group')
                    media_file.artist = data.get('artist')
                    media_file.album = data.get('album')
                    media_file.track = int(data.get('track')) if data.get('track') else None
                    
                    break
        
        # Extract quality and codec info from full filename
        self.extract_additional_info(filename, media_file)
        
        return media_file
    
    def clean_title(self, title: str) -> str:
        """Clean up title string"""
        # Replace dots and underscores with spaces
        title = title.replace('.', ' ').replace('_', ' ')
        
        # Remove extra spaces
        title = ' '.join(title.split())
        
        # Apply case style
        case_style = self.config['naming_conventions']['case_style']
        if case_style == 'title':
            title = title.title()
        elif case_style == 'lower':
            title = title.lower()
        elif case_style == 'upper':
            title = title.upper()
        
        # Remove special characters if configured
        if self.config['naming_conventions']['remove_special_chars']:
            title = re.sub(r'[^\w\s-]', '', title)
        
        return title.strip()
    
    def extract_additional_info(self, filename: str, media_file: MediaFile):
        """Extract quality, codec, and other info from filename"""
        # Quality detection
        for quality in self.config['quality_tags']:
            if quality.lower() in filename.lower():
                media_file.quality = quality
                break
        
        # Audio codec detection
        for audio in self.config['audio_tags']:
            if audio.lower() in filename.lower():
                media_file.audio_codec = audio
                break
        
        # Video codec detection
        for video in self.config['video_tags']:
            if video.lower() in filename.lower():
                media_file.video_codec = video
                break
        
        # Source detection
        sources = ['BluRay', 'WEB-DL', 'WEBRip', 'HDTV', 'DVDRip', 'BDRip']
        for source in sources:
            if source.lower() in filename.lower():
                media_file.source = source
                break
    
    def generate_organized_path(self, media_file: MediaFile) -> Path:
        """Generate organized file path based on naming convention"""
        # Get structure template
        structure = self.config['structures'].get(media_file.media_type.value, '')
        if not structure:
            return media_file.path
        
        # Prepare variables for formatting
        ext = media_file.path.suffix[1:]  # Remove dot
        
        # Build format variables
        format_vars = {
            'media_root': self.config['media_root'],
            'title': media_file.title,
            'year': media_file.year or 'Unknown',
            'season': media_file.season or 1,
            'episode': media_file.episode or 1,
            'episode_title': 'Episode',  # Would need external lookup
            'artist': media_file.artist or 'Unknown Artist',
            'album': media_file.album or 'Unknown Album',
            'track': media_file.track or 1,
            'author': 'Unknown Author',  # Would need external lookup
            'quality': media_file.quality or 'Unknown',
            'ext': ext,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H-%M-%S'),
            'month': datetime.now().strftime('%B')
        }
        
        # Format path
        try:
            organized_path = structure.format(**format_vars)
            
            # Handle duplicate naming
            if self.config['naming_conventions']['replace_spaces']:
                organized_path = organized_path.replace(' ', '.')
            
            # Ensure path length is within limits
            if len(organized_path) > self.config['naming_conventions']['max_length']:
                # Truncate filename while preserving extension
                base, ext = os.path.splitext(organized_path)
                max_base = self.config['naming_conventions']['max_length'] - len(ext) - 1
                organized_path = base[:max_base] + ext
            
            return Path(organized_path)
            
        except Exception as e:
            logger.error(f"Error generating organized path: {e}")
            return media_file.path
    
    def organize_file(self, file_path: Path, dry_run: bool = False) -> Optional[Path]:
        """Organize a single media file"""
        try:
            # Parse file information
            media_file = self.parse_filename(file_path)
            
            # Skip unknown media types
            if media_file.media_type == MediaType.UNKNOWN:
                logger.warning(f"Unknown media type for: {file_path}")
                return None
            
            # Generate organized path
            organized_path = self.generate_organized_path(media_file)
            
            # Check if already organized
            if file_path == organized_path:
                logger.info(f"Already organized: {file_path}")
                return file_path
            
            # Handle existing file at destination
            if organized_path.exists():
                organized_path = self.handle_duplicate(organized_path)
            
            if dry_run:
                logger.info(f"[DRY RUN] Would move: {file_path} -> {organized_path}")
                return organized_path
            
            # Create directory structure
            organized_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move or copy file
            if self.config['preserve_original']:
                shutil.copy2(file_path, organized_path)
                logger.info(f"Copied: {file_path} -> {organized_path}")
            else:
                shutil.move(str(file_path), str(organized_path))
                logger.info(f"Moved: {file_path} -> {organized_path}")
            
            # Update database
            self.update_database(file_path, organized_path, media_file)
            
            # Check for collection
            if self.config['collections']['enable']:
                self.check_collection(media_file)
            
            return organized_path
            
        except Exception as e:
            logger.error(f"Error organizing file {file_path}: {e}")
            return None
    
    def handle_duplicate(self, path: Path) -> Path:
        """Handle duplicate file at destination"""
        handling = self.config['duplicate_handling']
        
        if handling == 'skip':
            logger.info(f"Skipping duplicate: {path}")
            return path
        elif handling == 'replace':
            logger.info(f"Replacing existing file: {path}")
            return path
        elif handling == 'rename':
            # Find unique name
            base = path.stem
            ext = path.suffix
            counter = 1
            
            while path.exists():
                new_name = f"{base} ({counter}){ext}"
                path = path.parent / new_name
                counter += 1
            
            logger.info(f"Renamed to avoid duplicate: {path}")
            return path
        
        return path
    
    def organize_directory(self, directory: Path, recursive: bool = True, dry_run: bool = False):
        """Organize all media files in a directory"""
        media_files = []
        
        # Find all media files
        pattern = '**/*' if recursive else '*'
        for file_path in directory.glob(pattern):
            if file_path.is_file() and not file_path.name.startswith('.'):
                media_type = self.detect_media_type(file_path)
                if media_type != MediaType.UNKNOWN:
                    media_files.append(file_path)
        
        logger.info(f"Found {len(media_files)} media files to organize")
        
        # Organize files
        organized_count = 0
        for file_path in media_files:
            result = self.organize_file(file_path, dry_run)
            if result:
                organized_count += 1
        
        logger.info(f"Successfully organized {organized_count}/{len(media_files)} files")
    
    def check_collection(self, media_file: MediaFile):
        """Check if media should be part of a collection"""
        if media_file.media_type != MediaType.MOVIE:
            return
        
        try:
            # Check for collection info via TMDB API
            collection_info = self.get_collection_info(media_file.title, media_file.year)
            
            if collection_info:
                self.add_to_collection(media_file, collection_info)
            
        except Exception as e:
            logger.error(f"Error checking collection: {e}")
    
    def get_collection_info(self, title: str, year: Optional[int]) -> Optional[Dict]:
        """Get collection information from TMDB"""
        # This would require TMDB API integration
        # Placeholder for demonstration
        return None
    
    def add_to_collection(self, media_file: MediaFile, collection_info: Dict):
        """Add media file to a collection"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if collection exists
            cursor.execute('''
                SELECT id FROM collections WHERE name = ?
            ''', (collection_info['name'],))
            
            result = cursor.fetchone()
            
            if result:
                collection_id = result[0]
                # Update item count
                cursor.execute('''
                    UPDATE collections SET item_count = item_count + 1
                    WHERE id = ?
                ''', (collection_id,))
            else:
                # Create new collection
                cursor.execute('''
                    INSERT INTO collections (name, media_type, item_count, tmdb_collection_id)
                    VALUES (?, ?, 1, ?)
                ''', (collection_info['name'], media_file.media_type.value, collection_info.get('id')))
                collection_id = cursor.lastrowid
            
            # Update media file with collection ID
            cursor.execute('''
                UPDATE media_files SET collection_id = ?
                WHERE organized_path = ?
            ''', (collection_id, str(media_file.path)))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added {media_file.title} to collection: {collection_info['name']}")
            
        except Exception as e:
            logger.error(f"Error adding to collection: {e}")
    
    def update_database(self, original_path: Path, organized_path: Path, media_file: MediaFile):
        """Update database with organization information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO media_files 
                (original_path, organized_path, media_type, title, year, season, episode, quality, file_size, date_organized)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                str(original_path),
                str(organized_path),
                media_file.media_type.value,
                media_file.title,
                media_file.year,
                media_file.season,
                media_file.episode,
                media_file.quality,
                organized_path.stat().st_size if organized_path.exists() else 0
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating database: {e}")
    
    def create_custom_rule(self, pattern: str, media_type: MediaType, priority: int = 0):
        """Create custom organization rule"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO organization_rules (pattern, media_type, priority)
                VALUES (?, ?, ?)
            ''', (pattern, media_type.value, priority))
            
            conn.commit()
            conn.close()
            
            # Recompile patterns
            self.patterns = self.compile_patterns()
            
            logger.info(f"Created custom rule: {pattern}")
            
        except Exception as e:
            logger.error(f"Error creating custom rule: {e}")
    
    def generate_report(self) -> Dict:
        """Generate organization report"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get statistics
            cursor.execute('''
                SELECT media_type, COUNT(*) as count, SUM(file_size) as total_size
                FROM media_files
                GROUP BY media_type
            ''')
            
            stats = {}
            for row in cursor.fetchall():
                stats[row[0]] = {
                    'count': row[1],
                    'total_size': row[2]
                }
            
            # Get collection info
            cursor.execute('''
                SELECT name, item_count FROM collections
                ORDER BY item_count DESC
            ''')
            
            collections = [{'name': row[0], 'count': row[1]} for row in cursor.fetchall()]
            
            # Get recent activity
            cursor.execute('''
                SELECT title, media_type, date_organized
                FROM media_files
                ORDER BY date_organized DESC
                LIMIT 10
            ''')
            
            recent = [{'title': row[0], 'type': row[1], 'date': row[2]} for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                'statistics': stats,
                'collections': collections,
                'recent_activity': recent
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {}
    
    def cleanup_empty_directories(self, root_path: Path):
        """Remove empty directories after organization"""
        try:
            for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
                if not dirnames and not filenames:
                    os.rmdir(dirpath)
                    logger.info(f"Removed empty directory: {dirpath}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up directories: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Media Organization Workflow')
    parser.add_argument('path', help='Path to file or directory to organize')
    parser.add_argument('--config', default='/config/organization.json', help='Config file path')
    parser.add_argument('--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without doing it')
    parser.add_argument('--report', action='store_true', help='Generate organization report')
    parser.add_argument('--cleanup', action='store_true', help='Clean up empty directories')
    
    args = parser.parse_args()
    
    organizer = MediaOrganizer(args.config)
    
    if args.report:
        report = organizer.generate_report()
        print(json.dumps(report, indent=2))
    elif args.cleanup:
        organizer.cleanup_empty_directories(Path(args.path))
    else:
        path = Path(args.path)
        
        if path.is_file():
            result = organizer.organize_file(path, args.dry_run)
            if result:
                print(f"Organized: {path} -> {result}")
        elif path.is_dir():
            organizer.organize_directory(path, args.recursive, args.dry_run)
        else:
            print(f"Invalid path: {args.path}")

if __name__ == "__main__":
    main()