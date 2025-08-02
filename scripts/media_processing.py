#!/usr/bin/env python3
"""
Media Processing Automation Script
Handles format conversion, quality optimization, metadata enrichment, and subtitle management
"""

import os
import json
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime
import requests
import ffmpeg
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/media_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MediaProcessor:
    def __init__(self, config_path='/config/media_processing.json'):
        self.config = self.load_config(config_path)
        self.supported_formats = {
            'video': ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm'],
            'audio': ['.mp3', '.flac', '.aac', '.ogg', '.wav', '.m4a'],
            'subtitle': ['.srt', '.ass', '.ssa', '.sub', '.vtt']
        }
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        default_config = {
            'target_video_codec': 'h265',
            'target_audio_codec': 'aac',
            'video_quality': 23,
            'audio_bitrate': '192k',
            'max_resolution': '1080p',
            'preserve_quality': True,
            'hardware_acceleration': 'auto',
            'subtitle_languages': ['eng', 'spa'],
            'metadata_sources': ['tmdb', 'imdb'],
            'parallel_jobs': 2
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        return default_config
    
    def analyze_media(self, file_path):
        """Analyze media file and return metadata"""
        try:
            probe = ffmpeg.probe(file_path)
            file_info = {
                'path': file_path,
                'format': probe['format']['format_name'],
                'duration': float(probe['format']['duration']),
                'size': int(probe['format']['size']),
                'bit_rate': int(probe['format']['bit_rate']),
                'streams': []
            }
            
            for stream in probe['streams']:
                stream_info = {
                    'type': stream['codec_type'],
                    'codec': stream['codec_name'],
                    'index': stream['index']
                }
                
                if stream['codec_type'] == 'video':
                    stream_info.update({
                        'width': stream['width'],
                        'height': stream['height'],
                        'fps': eval(stream['r_frame_rate']),
                        'bit_rate': stream.get('bit_rate', 0)
                    })
                elif stream['codec_type'] == 'audio':
                    stream_info.update({
                        'channels': stream['channels'],
                        'sample_rate': stream['sample_rate'],
                        'bit_rate': stream.get('bit_rate', 0),
                        'language': stream.get('tags', {}).get('language', 'und')
                    })
                elif stream['codec_type'] == 'subtitle':
                    stream_info.update({
                        'language': stream.get('tags', {}).get('language', 'und')
                    })
                
                file_info['streams'].append(stream_info)
            
            return file_info
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return None
    
    def needs_conversion(self, file_info):
        """Check if file needs conversion based on config"""
        needs_convert = False
        reasons = []
        
        for stream in file_info['streams']:
            if stream['type'] == 'video':
                if stream['codec'] != self.config['target_video_codec']:
                    needs_convert = True
                    reasons.append(f"Video codec: {stream['codec']} -> {self.config['target_video_codec']}")
                
                if self.config['max_resolution'] == '1080p' and stream['height'] > 1080:
                    needs_convert = True
                    reasons.append(f"Resolution: {stream['height']}p -> 1080p")
                    
            elif stream['type'] == 'audio':
                if stream['codec'] != self.config['target_audio_codec']:
                    needs_convert = True
                    reasons.append(f"Audio codec: {stream['codec']} -> {self.config['target_audio_codec']}")
        
        return needs_convert, reasons
    
    def convert_media(self, file_path, output_path=None):
        """Convert media file to target format"""
        file_info = self.analyze_media(file_path)
        if not file_info:
            return False
        
        needs_convert, reasons = self.needs_conversion(file_info)
        if not needs_convert:
            logger.info(f"{file_path} doesn't need conversion")
            return True
        
        logger.info(f"Converting {file_path}: {', '.join(reasons)}")
        
        if not output_path:
            output_path = file_path.replace(Path(file_path).suffix, '_converted.mkv')
        
        try:
            # Build ffmpeg command
            input_stream = ffmpeg.input(file_path)
            
            # Video stream processing
            video = input_stream['v']
            if self.config['hardware_acceleration'] == 'nvidia':
                video = video.filter('scale_cuda', w=-1, h='min(ih,1080)')
                output_args = {'c:v': 'hevc_nvenc', 'preset': 'slow'}
            elif self.config['hardware_acceleration'] == 'intel':
                video = video.filter('scale_qsv', w=-1, h='min(ih,1080)')
                output_args = {'c:v': 'hevc_qsv', 'preset': 'slow'}
            else:
                if self.config['max_resolution'] == '1080p':
                    video = video.filter('scale', w=-1, h='min(ih,1080)')
                output_args = {'c:v': self.config['target_video_codec'], 'crf': self.config['video_quality']}
            
            # Audio stream processing
            audio = input_stream['a']
            output_args.update({
                'c:a': self.config['target_audio_codec'],
                'b:a': self.config['audio_bitrate']
            })
            
            # Subtitle stream processing
            output_args['c:s'] = 'copy'
            
            # Output
            output = ffmpeg.output(
                video, audio, input_stream['s'],
                output_path,
                **output_args
            )
            
            # Run conversion
            ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            # Verify output
            output_info = self.analyze_media(output_path)
            if output_info and output_info['duration'] >= file_info['duration'] * 0.95:
                logger.info(f"Successfully converted {file_path}")
                
                # Replace original if configured
                if self.config.get('replace_original', False):
                    os.remove(file_path)
                    os.rename(output_path, file_path)
                
                return True
            else:
                logger.error(f"Conversion failed verification for {file_path}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                return False
                
        except Exception as e:
            logger.error(f"Error converting {file_path}: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    
    def optimize_quality(self, file_path):
        """Optimize media quality based on content type"""
        file_info = self.analyze_media(file_path)
        if not file_info:
            return False
        
        # Calculate optimal bitrate based on resolution and content
        video_stream = next((s for s in file_info['streams'] if s['type'] == 'video'), None)
        if not video_stream:
            return False
        
        # Determine content type (action, animation, etc.) using scene detection
        scene_complexity = self.analyze_scene_complexity(file_path)
        
        # Calculate optimal settings
        if video_stream['height'] <= 480:
            target_bitrate = '1M' if scene_complexity > 0.5 else '800k'
        elif video_stream['height'] <= 720:
            target_bitrate = '3M' if scene_complexity > 0.5 else '2M'
        elif video_stream['height'] <= 1080:
            target_bitrate = '6M' if scene_complexity > 0.5 else '4M'
        else:  # 4K
            target_bitrate = '15M' if scene_complexity > 0.5 else '10M'
        
        # Apply optimization
        optimized_path = file_path.replace(Path(file_path).suffix, '_optimized.mkv')
        
        try:
            (
                ffmpeg
                .input(file_path)
                .output(
                    optimized_path,
                    vcodec='libx265',
                    preset='slow',
                    b=target_bitrate,
                    maxrate=f"{int(target_bitrate[:-1]) * 1.5}M",
                    bufsize=f"{int(target_bitrate[:-1]) * 2}M",
                    acodec='copy',
                    scodec='copy'
                )
                .run(overwrite_output=True)
            )
            
            # Compare file sizes
            original_size = os.path.getsize(file_path)
            optimized_size = os.path.getsize(optimized_path)
            
            if optimized_size < original_size * 0.9:  # At least 10% reduction
                logger.info(f"Optimized {file_path}: {original_size/(1024**3):.2f}GB -> {optimized_size/(1024**3):.2f}GB")
                os.remove(file_path)
                os.rename(optimized_path, file_path)
                return True
            else:
                os.remove(optimized_path)
                return False
                
        except Exception as e:
            logger.error(f"Error optimizing {file_path}: {e}")
            if os.path.exists(optimized_path):
                os.remove(optimized_path)
            return False
    
    def analyze_scene_complexity(self, file_path, sample_duration=60):
        """Analyze scene complexity for optimal encoding"""
        try:
            # Sample middle portion of video
            probe = ffmpeg.probe(file_path)
            duration = float(probe['format']['duration'])
            start_time = max(0, (duration - sample_duration) / 2)
            
            # Extract scene change metrics
            stats = (
                ffmpeg
                .input(file_path, ss=start_time, t=sample_duration)
                .filter('select', 'gt(scene,0.4)')
                .output('-', format='null')
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # Parse output for scene changes
            scene_changes = stats[1].decode().count('scene:')
            complexity = min(1.0, scene_changes / (sample_duration * 2))  # Normalize
            
            return complexity
            
        except Exception as e:
            logger.error(f"Error analyzing scene complexity: {e}")
            return 0.5  # Default medium complexity
    
    def enrich_metadata(self, file_path, media_type='movie'):
        """Enrich media file with metadata from external sources"""
        try:
            # Extract title from filename
            filename = Path(file_path).stem
            
            # Clean up filename
            import re
            title_match = re.match(r'^(.*?)[\.\s](\d{4})', filename)
            if title_match:
                title = title_match.group(1).replace('.', ' ')
                year = title_match.group(2)
            else:
                title = filename.replace('.', ' ')
                year = None
            
            # Fetch metadata from TMDB
            metadata = self.fetch_tmdb_metadata(title, year, media_type)
            if not metadata:
                return False
            
            # Apply metadata using ffmpeg
            metadata_args = {
                'metadata:g:0': f"title={metadata['title']}",
                'metadata:g:1': f"year={metadata['year']}",
                'metadata:g:2': f"genre={metadata['genre']}",
                'metadata:g:3': f"description={metadata['overview']}"
            }
            
            temp_path = file_path + '.tmp'
            (
                ffmpeg
                .input(file_path)
                .output(temp_path, codec='copy', **metadata_args)
                .run(overwrite_output=True)
            )
            
            os.remove(file_path)
            os.rename(temp_path, file_path)
            
            logger.info(f"Enriched metadata for {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error enriching metadata for {file_path}: {e}")
            return False
    
    def fetch_tmdb_metadata(self, title, year, media_type):
        """Fetch metadata from TMDB API"""
        # Note: You'll need to get a TMDB API key
        api_key = os.environ.get('TMDB_API_KEY', '')
        if not api_key:
            return None
        
        try:
            # Search for media
            search_url = f"https://api.themoviedb.org/3/search/{media_type}"
            params = {
                'api_key': api_key,
                'query': title,
                'year': year
            }
            
            response = requests.get(search_url, params=params)
            results = response.json().get('results', [])
            
            if not results:
                return None
            
            # Get detailed info
            media_id = results[0]['id']
            details_url = f"https://api.themoviedb.org/3/{media_type}/{media_id}"
            
            response = requests.get(details_url, params={'api_key': api_key})
            details = response.json()
            
            return {
                'title': details.get('title') or details.get('name'),
                'year': details.get('release_date', '')[:4] or details.get('first_air_date', '')[:4],
                'genre': ', '.join([g['name'] for g in details.get('genres', [])]),
                'overview': details.get('overview', '')
            }
            
        except Exception as e:
            logger.error(f"Error fetching TMDB metadata: {e}")
            return None
    
    def manage_subtitles(self, file_path):
        """Download and manage subtitles for media file"""
        try:
            # Use subliminal for subtitle management
            import subliminal
            
            # Configure providers
            providers = ['opensubtitles', 'podnapisi', 'subscenter']
            
            # Scan video
            video = subliminal.scan_video(file_path)
            
            # Download subtitles
            subtitles = subliminal.download_best_subtitles(
                {video},
                self.config['subtitle_languages'],
                providers=providers
            )
            
            # Save subtitles
            for subtitle in subtitles[video]:
                subliminal.save_subtitles(video, [subtitle])
                logger.info(f"Downloaded {subtitle.language} subtitle for {file_path}")
            
            # Embed subtitles if .mkv
            if file_path.endswith('.mkv'):
                self.embed_subtitles(file_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error managing subtitles for {file_path}: {e}")
            return False
    
    def embed_subtitles(self, file_path):
        """Embed external subtitles into MKV file"""
        try:
            # Find subtitle files
            base_path = Path(file_path).with_suffix('')
            subtitle_files = list(Path(file_path).parent.glob(f"{base_path.name}*.srt"))
            
            if not subtitle_files:
                return False
            
            # Build ffmpeg command
            inputs = [ffmpeg.input(file_path)]
            for sub_file in subtitle_files:
                inputs.append(ffmpeg.input(str(sub_file)))
            
            # Map all streams
            output_args = {
                'c': 'copy',
                'map': '0',
            }
            
            # Add subtitle mappings
            for i, _ in enumerate(subtitle_files):
                output_args[f'map'] = f'{i+1}'
            
            temp_path = file_path + '.tmp'
            output = ffmpeg.output(*inputs, temp_path, **output_args)
            ffmpeg.run(output, overwrite_output=True)
            
            # Replace original
            os.remove(file_path)
            os.rename(temp_path, file_path)
            
            # Remove external subtitle files
            for sub_file in subtitle_files:
                os.remove(sub_file)
            
            logger.info(f"Embedded {len(subtitle_files)} subtitles into {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error embedding subtitles: {e}")
            return False
    
    def process_directory(self, directory_path, recursive=True):
        """Process all media files in a directory"""
        media_files = []
        
        # Find all media files
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.supported_formats['video']):
                    media_files.append(os.path.join(root, file))
            
            if not recursive:
                break
        
        logger.info(f"Found {len(media_files)} media files to process")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.config['parallel_jobs']) as executor:
            futures = []
            
            for file_path in media_files:
                future = executor.submit(self.process_file, file_path)
                futures.append((future, file_path))
            
            for future, file_path in futures:
                try:
                    result = future.result()
                    if result:
                        logger.info(f"Successfully processed {file_path}")
                    else:
                        logger.error(f"Failed to process {file_path}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
    
    def process_file(self, file_path):
        """Process a single media file through all workflows"""
        try:
            # Skip if file is being written
            if self.is_file_growing(file_path):
                logger.info(f"Skipping {file_path} - file is still being written")
                return False
            
            # Convert if needed
            if self.config.get('enable_conversion', True):
                self.convert_media(file_path)
            
            # Optimize quality
            if self.config.get('enable_optimization', True):
                self.optimize_quality(file_path)
            
            # Enrich metadata
            if self.config.get('enable_metadata', True):
                self.enrich_metadata(file_path)
            
            # Manage subtitles
            if self.config.get('enable_subtitles', True):
                self.manage_subtitles(file_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return False
    
    def is_file_growing(self, file_path, check_interval=5):
        """Check if file is still being written"""
        try:
            size1 = os.path.getsize(file_path)
            time.sleep(check_interval)
            size2 = os.path.getsize(file_path)
            return size1 != size2
        except:
            return False

def main():
    parser = argparse.ArgumentParser(description='Media Processing Automation')
    parser.add_argument('path', help='Path to media file or directory')
    parser.add_argument('--config', default='/config/media_processing.json', help='Config file path')
    parser.add_argument('--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('--watch', action='store_true', help='Watch directory for new files')
    
    args = parser.parse_args()
    
    processor = MediaProcessor(args.config)
    
    if os.path.isfile(args.path):
        processor.process_file(args.path)
    elif os.path.isdir(args.path):
        if args.watch:
            # Watch mode implementation would go here
            logger.info("Watch mode not implemented yet")
        else:
            processor.process_directory(args.path, args.recursive)
    else:
        logger.error(f"Invalid path: {args.path}")

if __name__ == "__main__":
    main()