#!/usr/bin/env python3
"""
Media Organizer Script
Automatically organizes media files with proper naming, structure, and metadata
Version: 2025.1
"""

import os
import re
import sys
import json
import shutil
import hashlib
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
import concurrent.futures
from dataclasses import dataclass
from collections import defaultdict

# Try to import optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: 'requests' module not found. API features will be limited.")

try:
    from guessit import guessit
    HAS_GUESSIT = True
except ImportError:
    HAS_GUESSIT = False
    print("Warning: 'guessit' module not found. Install with: pip install guessit")

try:
    import ffmpeg
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False
    print("Warning: 'ffmpeg-python' module not found. Install with: pip install ffmpeg-python")

# Configuration
@dataclass
class Config:
    """Configuration settings for media organizer"""
    media_root: Path = Path("/media")
    tv_dir: Path = Path("/media/tv")
    movie_dir: Path = Path("/media/movies")
    music_dir: Path = Path("/media/music")
    temp_dir: Path = Path("/tmp/media-organizer")
    
    # File extensions
    video_extensions: Set[str] = None
    audio_extensions: Set[str] = None
    subtitle_extensions: Set[str] = None
    
    # Naming patterns
    tv_pattern: str = "{series} - S{season:02d}E{episode:02d} - {title} [{quality}]"
    movie_pattern: str = "{title} ({year}) [{quality}]"
    
    # Quality mappings
    quality_priority: Dict[str, int] = None
    
    # API settings
    tmdb_api_key: str = ""
    tvdb_api_key: str = ""
    
    # Behavior settings
    dry_run: bool = False
    interactive: bool = False
    keep_originals: bool = True
    create_hardlinks: bool = True
    min_file_size: int = 50 * 1024 * 1024  # 50MB
    
    def __post_init__(self):
        if self.video_extensions is None:
            self.video_extensions = {
                '.mkv', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm',
                '.m4v', '.mpg', '.mpeg', '.3gp', '.ogv', '.ts', '.m2ts'
            }
        
        if self.audio_extensions is None:
            self.audio_extensions = {
                '.mp3', '.flac', '.ogg', '.opus', '.m4a', '.aac', '.wav',
                '.wma', '.ape', '.alac', '.dts', '.ac3'
            }
        
        if self.subtitle_extensions is None:
            self.subtitle_extensions = {
                '.srt', '.ass', '.ssa', '.sub', '.vtt', '.idx'
            }
        
        if self.quality_priority is None:
            self.quality_priority = {
                '2160p': 100, '4K': 100, 'UHD': 100,
                '1080p': 90, 'FHD': 90,
                '720p': 80, 'HD': 80,
                '480p': 70, 'SD': 70,
                'BluRay': 95, 'BDRip': 85,
                'WEB-DL': 85, 'WEBRip': 75,
                'HDTV': 70, 'DVDRip': 60
            }

# Setup logging
def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    
    return logging.getLogger(__name__)

# Media file class
@dataclass
class MediaFile:
    """Represents a media file with metadata"""
    path: Path
    type: str  # 'movie', 'tv', 'music'
    title: str = ""
    year: Optional[int] = None
    season: Optional[int] = None
    episode: Optional[int] = None
    quality: str = ""
    codec: str = ""
    audio: str = ""
    size: int = 0
    hash: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class MediaOrganizer:
    """Main media organizer class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.stats = defaultdict(int)
        
        # Create directories
        for dir_path in [config.tv_dir, config.movie_dir, config.music_dir, config.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def calculate_file_hash(self, filepath: Path, chunk_size: int = 8192) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def extract_metadata(self, filepath: Path) -> Dict:
        """Extract metadata from media file using ffprobe"""
        if not HAS_FFMPEG:
            return {}
        
        try:
            probe = ffmpeg.probe(str(filepath))
            
            metadata = {
                'format': probe.get('format', {}).get('format_name', ''),
                'duration': float(probe.get('format', {}).get('duration', 0)),
                'size': int(probe.get('format', {}).get('size', 0)),
                'bitrate': int(probe.get('format', {}).get('bit_rate', 0)),
                'streams': []
            }
            
            for stream in probe.get('streams', []):
                stream_info = {
                    'type': stream.get('codec_type'),
                    'codec': stream.get('codec_name'),
                    'language': stream.get('tags', {}).get('language', 'und')
                }
                
                if stream['codec_type'] == 'video':
                    stream_info.update({
                        'width': stream.get('width'),
                        'height': stream.get('height'),
                        'fps': eval(stream.get('r_frame_rate', '0/1'))
                    })
                elif stream['codec_type'] == 'audio':
                    stream_info.update({
                        'channels': stream.get('channels'),
                        'sample_rate': stream.get('sample_rate')
                    })
                
                metadata['streams'].append(stream_info)
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Failed to extract metadata from {filepath}: {e}")
            return {}
    
    def parse_filename(self, filepath: Path) -> MediaFile:
        """Parse filename to extract media information"""
        filename = filepath.stem
        
        # Initialize media file
        media = MediaFile(path=filepath, type='unknown')
        media.size = filepath.stat().st_size
        
        if HAS_GUESSIT:
            # Use guessit for parsing
            guess = guessit(filename)
            
            media.title = guess.get('title', '')
            media.year = guess.get('year')
            media.season = guess.get('season')
            media.episode = guess.get('episode')
            media.quality = guess.get('screen_size', '')
            media.codec = guess.get('video_codec', '')
            media.audio = guess.get('audio_codec', '')
            
            # Determine type
            if guess.get('type') == 'episode' or media.season is not None:
                media.type = 'tv'
            elif guess.get('type') == 'movie' or media.year:
                media.type = 'movie'
        else:
            # Basic regex parsing
            # TV show pattern: Show Name S01E01
            tv_match = re.search(r'^(.*?)\s*S(\d+)E(\d+)', filename, re.IGNORECASE)
            if tv_match:
                media.type = 'tv'
                media.title = tv_match.group(1).strip()
                media.season = int(tv_match.group(2))
                media.episode = int(tv_match.group(3))
            else:
                # Movie pattern: Movie Name (Year)
                movie_match = re.search(r'^(.*?)\s*\((\d{4})\)', filename)
                if movie_match:
                    media.type = 'movie'
                    media.title = movie_match.group(1).strip()
                    media.year = int(movie_match.group(2))
                else:
                    # Fallback to simple title
                    media.title = re.sub(r'[\._\-]', ' ', filename).strip()
            
            # Extract quality
            quality_match = re.search(r'(2160p|1080p|720p|480p|4K|UHD|HD|SD)', filename, re.IGNORECASE)
            if quality_match:
                media.quality = quality_match.group(1)
        
        # Clean up title
        media.title = re.sub(r'[\._]', ' ', media.title)
        media.title = re.sub(r'\s+', ' ', media.title).strip()
        
        # Extract metadata
        media.metadata = self.extract_metadata(filepath)
        
        return media
    
    def find_duplicates(self, directory: Path) -> List[List[Path]]:
        """Find duplicate files based on hash"""
        self.logger.info(f"Scanning for duplicates in {directory}")
        
        hash_map = defaultdict(list)
        
        for filepath in directory.rglob('*'):
            if filepath.is_file() and filepath.suffix.lower() in self.config.video_extensions:
                file_hash = self.calculate_file_hash(filepath)
                hash_map[file_hash].append(filepath)
        
        # Return only duplicates
        duplicates = [paths for paths in hash_map.values() if len(paths) > 1]
        
        return duplicates
    
    def get_quality_score(self, media: MediaFile) -> int:
        """Calculate quality score for media file"""
        score = 0
        
        # Quality score
        for quality, points in self.config.quality_priority.items():
            if quality.lower() in media.quality.lower():
                score += points
                break
        
        # Codec bonus
        if media.codec in ['hevc', 'h265', 'x265']:
            score += 10
        elif media.codec in ['h264', 'x264']:
            score += 5
        
        # Audio bonus
        if 'dts' in media.audio.lower() or 'atmos' in media.audio.lower():
            score += 5
        
        # File size penalty (avoid unnecessarily large files)
        if media.size > 10 * 1024 * 1024 * 1024:  # > 10GB
            score -= 5
        
        return score
    
    def choose_best_duplicate(self, duplicates: List[Path]) -> Path:
        """Choose the best quality file from duplicates"""
        best_file = None
        best_score = -1
        
        for filepath in duplicates:
            media = self.parse_filename(filepath)
            score = self.get_quality_score(media)
            
            if score > best_score:
                best_score = score
                best_file = filepath
        
        return best_file
    
    def generate_new_path(self, media: MediaFile) -> Path:
        """Generate new organized path for media file"""
        if media.type == 'tv':
            # TV Show organization
            show_name = media.title.title()
            season_folder = f"Season {media.season:02d}"
            
            if media.episode:
                filename = self.config.tv_pattern.format(
                    series=show_name,
                    season=media.season,
                    episode=media.episode,
                    title=media.metadata.get('episode_title', f"Episode {media.episode}"),
                    quality=media.quality
                )
            else:
                filename = f"{show_name} - S{media.season:02d} - Unknown Episode [{media.quality}]"
            
            new_path = self.config.tv_dir / show_name / season_folder / f"{filename}{media.path.suffix}"
            
        elif media.type == 'movie':
            # Movie organization
            movie_title = media.title.title()
            year_str = f"({media.year})" if media.year else ""
            
            filename = self.config.movie_pattern.format(
                title=movie_title,
                year=media.year or "Unknown",
                quality=media.quality
            )
            
            folder_name = f"{movie_title} {year_str}".strip()
            new_path = self.config.movie_dir / folder_name / f"{filename}{media.path.suffix}"
            
        else:
            # Unknown type - organize by extension
            new_path = self.config.media_root / "unsorted" / media.path.name
        
        return new_path
    
    def organize_file(self, filepath: Path) -> Optional[Path]:
        """Organize a single media file"""
        try:
            # Skip small files
            if filepath.stat().st_size < self.config.min_file_size:
                self.logger.debug(f"Skipping small file: {filepath}")
                return None
            
            # Parse file information
            media = self.parse_filename(filepath)
            
            # Generate new path
            new_path = self.generate_new_path(media)
            
            # Check if already organized
            if filepath == new_path:
                self.logger.debug(f"File already organized: {filepath}")
                return filepath
            
            # Create directory structure
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move or link file
            if self.config.dry_run:
                self.logger.info(f"[DRY RUN] Would move: {filepath} -> {new_path}")
            else:
                if new_path.exists():
                    # Compare files
                    if self.calculate_file_hash(filepath) == self.calculate_file_hash(new_path):
                        self.logger.info(f"Identical file already exists: {new_path}")
                        if not self.config.keep_originals:
                            filepath.unlink()
                        return new_path
                    else:
                        # Handle naming conflict
                        new_path = self.get_unique_path(new_path)
                
                if self.config.create_hardlinks:
                    try:
                        os.link(filepath, new_path)
                        self.logger.info(f"Created hardlink: {filepath} -> {new_path}")
                    except OSError:
                        # Fallback to copy if hardlink fails (different filesystem)
                        shutil.copy2(filepath, new_path)
                        self.logger.info(f"Copied file: {filepath} -> {new_path}")
                        if not self.config.keep_originals:
                            filepath.unlink()
                else:
                    shutil.move(str(filepath), str(new_path))
                    self.logger.info(f"Moved file: {filepath} -> {new_path}")
            
            # Handle associated files (subtitles, etc.)
            self.organize_associated_files(filepath, new_path)
            
            self.stats['organized'] += 1
            return new_path
            
        except Exception as e:
            self.logger.error(f"Failed to organize {filepath}: {e}")
            self.stats['errors'] += 1
            return None
    
    def organize_associated_files(self, original_path: Path, new_path: Path):
        """Organize subtitle and other associated files"""
        base_name = original_path.stem
        parent_dir = original_path.parent
        
        for file in parent_dir.glob(f"{base_name}*"):
            if file == original_path:
                continue
            
            if file.suffix.lower() in self.config.subtitle_extensions:
                # Subtitle file
                new_sub_path = new_path.parent / f"{new_path.stem}{file.suffix}"
                
                if self.config.dry_run:
                    self.logger.info(f"[DRY RUN] Would move subtitle: {file} -> {new_sub_path}")
                else:
                    try:
                        shutil.move(str(file), str(new_sub_path))
                        self.logger.info(f"Moved subtitle: {file} -> {new_sub_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to move subtitle {file}: {e}")
    
    def get_unique_path(self, path: Path) -> Path:
        """Generate unique path if file already exists"""
        if not path.exists():
            return path
        
        counter = 1
        while True:
            new_path = path.parent / f"{path.stem} ({counter}){path.suffix}"
            if not new_path.exists():
                return new_path
            counter += 1
    
    def scan_and_organize(self, source_dir: Path):
        """Scan directory and organize all media files"""
        self.logger.info(f"Scanning directory: {source_dir}")
        
        media_files = []
        
        # Collect all media files
        for ext in self.config.video_extensions:
            media_files.extend(source_dir.rglob(f"*{ext}"))
        
        self.logger.info(f"Found {len(media_files)} media files")
        
        # Process files
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.organize_file, file) for file in media_files]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    self.logger.error(f"Error processing file: {e}")
    
    def remove_empty_directories(self, directory: Path):
        """Remove empty directories recursively"""
        for dirpath in sorted(directory.rglob('*'), reverse=True):
            if dirpath.is_dir():
                try:
                    dirpath.rmdir()
                    self.logger.info(f"Removed empty directory: {dirpath}")
                except OSError:
                    pass  # Directory not empty
    
    def generate_report(self):
        """Generate organization report"""
        report = f"""
Media Organization Report
========================
Total files organized: {self.stats['organized']}
Errors encountered: {self.stats['errors']}
Duplicates found: {self.stats['duplicates']}
Space saved: {self.stats['space_saved'] / (1024**3):.2f} GB

Timestamp: {datetime.now().isoformat()}
"""
        return report

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Media Organizer - Automatically organize your media files")
    parser.add_argument('source', nargs='?', default='.', help='Source directory to organize')
    parser.add_argument('-d', '--dry-run', action='store_true', help='Perform dry run without moving files')
    parser.add_argument('-i', '--interactive', action='store_true', help='Ask before each action')
    parser.add_argument('--tv-dir', default='/media/tv', help='TV shows directory')
    parser.add_argument('--movie-dir', default='/media/movies', help='Movies directory')
    parser.add_argument('--no-hardlinks', action='store_true', help='Copy files instead of creating hardlinks')
    parser.add_argument('--remove-duplicates', action='store_true', help='Remove duplicate files')
    parser.add_argument('--min-size', type=int, default=50, help='Minimum file size in MB')
    parser.add_argument('--log', help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log)
    
    # Create config
    config = Config(
        tv_dir=Path(args.tv_dir),
        movie_dir=Path(args.movie_dir),
        dry_run=args.dry_run,
        interactive=args.interactive,
        create_hardlinks=not args.no_hardlinks,
        min_file_size=args.min_size * 1024 * 1024
    )
    
    # Create organizer
    organizer = MediaOrganizer(config)
    
    # Process source directory
    source_path = Path(args.source).resolve()
    if not source_path.exists():
        logger.error(f"Source directory does not exist: {source_path}")
        sys.exit(1)
    
    # Find and handle duplicates
    if args.remove_duplicates:
        logger.info("Searching for duplicates...")
        duplicates = organizer.find_duplicates(source_path)
        
        for dup_group in duplicates:
            logger.info(f"Found {len(dup_group)} duplicates:")
            for dup in dup_group:
                logger.info(f"  - {dup}")
            
            best = organizer.choose_best_duplicate(dup_group)
            logger.info(f"  Best quality: {best}")
            
            if not config.dry_run and not config.interactive:
                # Remove other duplicates
                for dup in dup_group:
                    if dup != best:
                        dup.unlink()
                        logger.info(f"  Removed duplicate: {dup}")
                        organizer.stats['duplicates'] += 1
    
    # Organize files
    organizer.scan_and_organize(source_path)
    
    # Clean up empty directories
    if not config.dry_run:
        organizer.remove_empty_directories(source_path)
    
    # Print report
    print(organizer.generate_report())

if __name__ == "__main__":
    main()