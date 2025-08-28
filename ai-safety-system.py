#!/usr/bin/env python3
"""
AI Safety System - O3-mini Style Reasoning with Content Moderation
Comprehensive safety layer for media server with ethical AI guardrails
"""

import asyncio
import json
import logging
import hashlib
import time
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import aiohttp
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import cv2
from PIL import Image, ImageFilter
import imagehash
import requests
import os
from urllib.parse import urlparse
import mimetypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_safety.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ContentAssessment:
    """Content safety assessment result"""
    content_id: str
    content_type: str
    safety_score: float  # 0.0 (unsafe) to 1.0 (safe)
    nsfw_probability: float
    copyright_risk: float
    ethical_score: float
    reasoning_chain: List[str]
    violations: List[str]
    recommendations: List[str]
    timestamp: datetime
    
class O3MiniReasoner:
    """O3-mini style reasoning agent for content analysis"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reasoning_steps = []
        
        # Initialize models for different analysis tasks
        self.nsfw_classifier = pipeline(
            "image-classification", 
            model="Falconsai/nsfw_image_detection",
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.text_classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.embedding_model = pipeline(
            "feature-extraction",
            model="sentence-transformers/all-MiniLM-L6-v2",
            device=0 if torch.cuda.is_available() else -1
        )
        
    async def reason_through_content(self, content: Any, content_type: str) -> ContentAssessment:
        """O3-mini style reasoning through content safety"""
        reasoning_chain = []
        violations = []
        recommendations = []
        
        # Step 1: Content type analysis
        reasoning_chain.append(f"Analyzing {content_type} content for safety assessment")
        
        # Step 2: Multi-modal analysis
        nsfw_score = 0.0
        ethical_score = 1.0
        copyright_risk = 0.0
        
        if content_type in ['image', 'video']:
            reasoning_chain.append("Performing visual content analysis")
            nsfw_score, visual_reasoning = await self._analyze_visual_content(content)
            reasoning_chain.extend(visual_reasoning)
            
        elif content_type == 'text':
            reasoning_chain.append("Performing text content analysis")
            nsfw_score, text_reasoning = await self._analyze_text_content(content)
            reasoning_chain.extend(text_reasoning)
            
        elif content_type == 'audio':
            reasoning_chain.append("Performing audio content analysis")
            nsfw_score, audio_reasoning = await self._analyze_audio_content(content)
            reasoning_chain.extend(audio_reasoning)
        
        # Step 3: Copyright analysis
        reasoning_chain.append("Analyzing copyright and intellectual property risks")
        copyright_risk, copyright_reasoning = await self._analyze_copyright(content, content_type)
        reasoning_chain.extend(copyright_reasoning)
        
        # Step 4: Ethical assessment
        reasoning_chain.append("Performing ethical assessment")
        ethical_score, ethical_reasoning = await self._ethical_assessment(content, content_type)
        reasoning_chain.extend(ethical_reasoning)
        
        # Step 5: Safety score calculation
        safety_score = self._calculate_safety_score(nsfw_score, copyright_risk, ethical_score)
        reasoning_chain.append(f"Calculated overall safety score: {safety_score:.3f}")
        
        # Step 6: Violation detection
        if nsfw_score > 0.7:
            violations.append("High NSFW content probability")
            recommendations.append("Block content or add content warning")
            
        if copyright_risk > 0.6:
            violations.append("Potential copyright infringement")
            recommendations.append("Review copyright status before distribution")
            
        if ethical_score < 0.3:
            violations.append("Ethical concerns detected")
            recommendations.append("Review content for harmful material")
        
        if safety_score < 0.5:
            recommendations.append("Content should be restricted or filtered")
        elif safety_score < 0.7:
            recommendations.append("Content should include appropriate warnings")
        else:
            recommendations.append("Content appears safe for general distribution")
        
        content_id = hashlib.md5(str(content).encode()).hexdigest()[:12]
        
        return ContentAssessment(
            content_id=content_id,
            content_type=content_type,
            safety_score=safety_score,
            nsfw_probability=nsfw_score,
            copyright_risk=copyright_risk,
            ethical_score=ethical_score,
            reasoning_chain=reasoning_chain,
            violations=violations,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    async def _analyze_visual_content(self, image_path: str) -> Tuple[float, List[str]]:
        """Analyze visual content for NSFW and harmful material"""
        reasoning = []
        
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            reasoning.append(f"Loaded image with dimensions: {image.size}")
            
            # NSFW classification
            nsfw_result = self.nsfw_classifier(image)
            nsfw_score = max([r['score'] for r in nsfw_result if 'nsfw' in r['label'].lower()])
            reasoning.append(f"NSFW classification score: {nsfw_score:.3f}")
            
            # Additional visual analysis
            # Check for explicit visual markers
            if nsfw_score > 0.8:
                reasoning.append("High confidence NSFW content detected")
            elif nsfw_score > 0.5:
                reasoning.append("Moderate NSFW indicators present")
            else:
                reasoning.append("Low NSFW probability")
            
            # Analyze image hash for known harmful content
            img_hash = str(imagehash.dhash(image))
            reasoning.append(f"Generated perceptual hash: {img_hash[:8]}...")
            
            return nsfw_score, reasoning
            
        except Exception as e:
            reasoning.append(f"Error in visual analysis: {str(e)}")
            return 0.5, reasoning  # Default to moderate risk on error
    
    async def _analyze_text_content(self, text: str) -> Tuple[float, List[str]]:
        """Analyze text content for toxicity and harmful material"""
        reasoning = []
        
        try:
            # Toxicity classification
            toxic_result = self.text_classifier(text)
            toxic_score = max([r['score'] for r in toxic_result if 'toxic' in r['label'].lower()])
            reasoning.append(f"Toxicity score: {toxic_score:.3f}")
            
            # Additional text analysis
            # Check for explicit keywords (basic pattern matching)
            explicit_patterns = [
                r'\b(explicit|nsfw|adult|xxx)\b',
                r'\b(nude|naked|sex|porn)\b',
                r'\b(violence|hate|threat)\b'
            ]
            
            explicit_matches = 0
            for pattern in explicit_patterns:
                matches = len(re.findall(pattern, text.lower()))
                explicit_matches += matches
                if matches > 0:
                    reasoning.append(f"Found {matches} matches for pattern: {pattern}")
            
            # Adjust score based on explicit content
            adjusted_score = min(toxic_score + (explicit_matches * 0.1), 1.0)
            reasoning.append(f"Adjusted score with explicit content: {adjusted_score:.3f}")
            
            return adjusted_score, reasoning
            
        except Exception as e:
            reasoning.append(f"Error in text analysis: {str(e)}")
            return 0.3, reasoning
    
    async def _analyze_audio_content(self, audio_path: str) -> Tuple[float, List[str]]:
        """Analyze audio content for harmful material"""
        reasoning = []
        
        try:
            # Basic audio analysis (placeholder for more sophisticated analysis)
            file_size = os.path.getsize(audio_path)
            reasoning.append(f"Audio file size: {file_size} bytes")
            
            # For now, return low risk for audio content
            # In production, this would include speech-to-text and audio classification
            reasoning.append("Audio content analysis requires speech-to-text processing")
            reasoning.append("Defaulting to low risk for audio content")
            
            return 0.2, reasoning
            
        except Exception as e:
            reasoning.append(f"Error in audio analysis: {str(e)}")
            return 0.3, reasoning
    
    async def _analyze_copyright(self, content: Any, content_type: str) -> Tuple[float, List[str]]:
        """Analyze copyright and IP risks"""
        reasoning = []
        
        try:
            copyright_risk = 0.0
            
            # Basic copyright analysis
            if isinstance(content, str) and content_type == 'text':
                # Check for copyright notices
                if re.search(r'©|\(c\)|copyright', content.lower()):
                    copyright_risk += 0.3
                    reasoning.append("Copyright notice found in text")
                
                # Check for known copyrighted phrases (basic)
                if len(content.split()) > 100:  # Long content more likely to be copyrighted
                    copyright_risk += 0.2
                    reasoning.append("Long text content increases copyright risk")
            
            elif content_type in ['image', 'video']:
                # Image/video copyright analysis would require more sophisticated methods
                reasoning.append("Visual copyright analysis requires advanced fingerprinting")
                copyright_risk = 0.1  # Low default risk
            
            reasoning.append(f"Copyright risk assessment: {copyright_risk:.3f}")
            return copyright_risk, reasoning
            
        except Exception as e:
            reasoning.append(f"Error in copyright analysis: {str(e)}")
            return 0.3, reasoning
    
    async def _ethical_assessment(self, content: Any, content_type: str) -> Tuple[float, List[str]]:
        """Perform ethical assessment of content"""
        reasoning = []
        
        try:
            ethical_score = 1.0
            
            # Ethical considerations
            reasoning.append("Evaluating content for ethical concerns")
            
            if content_type == 'text' and isinstance(content, str):
                # Check for harmful patterns
                harmful_patterns = [
                    r'\b(hate|discrimination|bias)\b',
                    r'\b(violence|harm|threat)\b',
                    r'\b(misinformation|fake|false)\b'
                ]
                
                for pattern in harmful_patterns:
                    matches = len(re.findall(pattern, content.lower()))
                    if matches > 0:
                        ethical_score -= 0.2
                        reasoning.append(f"Potential ethical concern: {pattern} ({matches} matches)")
            
            # Ensure score doesn't go below 0
            ethical_score = max(ethical_score, 0.0)
            reasoning.append(f"Ethical assessment score: {ethical_score:.3f}")
            
            return ethical_score, reasoning
            
        except Exception as e:
            reasoning.append(f"Error in ethical assessment: {str(e)}")
            return 0.5, reasoning
    
    def _calculate_safety_score(self, nsfw_score: float, copyright_risk: float, ethical_score: float) -> float:
        """Calculate overall safety score"""
        # Weight the different factors
        safety_score = (
            (1.0 - nsfw_score) * 0.4 +      # NSFW is major concern
            (1.0 - copyright_risk) * 0.3 +   # Copyright is important
            ethical_score * 0.3               # Ethical concerns are important
        )
        
        return max(0.0, min(1.0, safety_score))

class ContentModerationSystem:
    """Comprehensive content moderation system"""
    
    def __init__(self, db_path: str = "content_moderation.db"):
        self.db_path = db_path
        self.reasoner = O3MiniReasoner()
        self._init_database()
        
        # Content filtering rules
        self.blocked_domains = set([
            'malicious-site.com',
            'known-malware.org',
            # Add known harmful domains
        ])
        
        self.safe_domains = set([
            'youtube.com',
            'vimeo.com',
            'github.com',
            'wikipedia.org'
        ])
    
    def _init_database(self):
        """Initialize the moderation database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id TEXT UNIQUE,
                content_type TEXT,
                safety_score REAL,
                nsfw_probability REAL,
                copyright_risk REAL,
                ethical_score REAL,
                violations TEXT,
                recommendations TEXT,
                reasoning_chain TEXT,
                timestamp DATETIME,
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS moderation_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id TEXT,
                action TEXT,
                reason TEXT,
                moderator TEXT,
                timestamp DATETIME,
                FOREIGN KEY (content_id) REFERENCES content_assessments (content_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def assess_content(self, content: Any, content_type: str, source_url: str = None) -> ContentAssessment:
        """Assess content safety using O3-mini reasoning"""
        
        # Pre-filtering based on source
        if source_url:
            domain = urlparse(source_url).netloc
            if domain in self.blocked_domains:
                logger.warning(f"Blocked domain detected: {domain}")
                return self._create_blocked_assessment(content, content_type, "Blocked domain")
        
        # Perform detailed analysis
        assessment = await self.reasoner.reason_through_content(content, content_type)
        
        # Store assessment in database
        await self._store_assessment(assessment)
        
        # Log the assessment
        logger.info(f"Content assessment completed: {assessment.content_id} - Safety: {assessment.safety_score:.3f}")
        
        return assessment
    
    def _create_blocked_assessment(self, content: Any, content_type: str, reason: str) -> ContentAssessment:
        """Create assessment for blocked content"""
        content_id = hashlib.md5(str(content).encode()).hexdigest()[:12]
        
        return ContentAssessment(
            content_id=content_id,
            content_type=content_type,
            safety_score=0.0,
            nsfw_probability=1.0,
            copyright_risk=1.0,
            ethical_score=0.0,
            reasoning_chain=[f"Content blocked: {reason}"],
            violations=[reason],
            recommendations=["Content blocked by safety system"],
            timestamp=datetime.now()
        )
    
    async def _store_assessment(self, assessment: ContentAssessment):
        """Store assessment in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO content_assessments 
            (content_id, content_type, safety_score, nsfw_probability, 
             copyright_risk, ethical_score, violations, recommendations, 
             reasoning_chain, timestamp, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            assessment.content_id,
            assessment.content_type,
            assessment.safety_score,
            assessment.nsfw_probability,
            assessment.copyright_risk,
            assessment.ethical_score,
            json.dumps(assessment.violations),
            json.dumps(assessment.recommendations),
            json.dumps(assessment.reasoning_chain),
            assessment.timestamp.isoformat(),
            'assessed'
        ))
        
        conn.commit()
        conn.close()
    
    async def get_content_status(self, content_id: str) -> Optional[Dict]:
        """Get content moderation status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM content_assessments WHERE content_id = ?
        ''', (content_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'content_id': result[1],
                'content_type': result[2],
                'safety_score': result[3],
                'nsfw_probability': result[4],
                'copyright_risk': result[5],
                'ethical_score': result[6],
                'violations': json.loads(result[7]),
                'recommendations': json.loads(result[8]),
                'reasoning_chain': json.loads(result[9]),
                'timestamp': result[10],
                'status': result[11]
            }
        
        return None
    
    async def moderate_batch(self, content_list: List[Tuple[Any, str, str]]) -> List[ContentAssessment]:
        """Moderate multiple content items in batch"""
        assessments = []
        
        tasks = [
            self.assess_content(content, content_type, source_url)
            for content, content_type, source_url in content_list
        ]
        
        assessments = await asyncio.gather(*tasks)
        
        logger.info(f"Batch moderation completed: {len(assessments)} items processed")
        return assessments
    
    def get_safety_summary(self) -> Dict:
        """Get summary of moderation activities"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get counts by safety level
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                AVG(safety_score) as avg_safety,
                COUNT(CASE WHEN safety_score < 0.3 THEN 1 END) as high_risk,
                COUNT(CASE WHEN safety_score >= 0.3 AND safety_score < 0.7 THEN 1 END) as moderate_risk,
                COUNT(CASE WHEN safety_score >= 0.7 THEN 1 END) as low_risk
            FROM content_assessments
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'total_assessments': result[0],
            'average_safety_score': result[1] or 0.0,
            'high_risk_content': result[2],
            'moderate_risk_content': result[3],
            'low_risk_content': result[4],
            'last_updated': datetime.now().isoformat()
        }

class AIGuardrailSystem:
    """AI system with comprehensive safety guardrails"""
    
    def __init__(self):
        self.moderation_system = ContentModerationSystem()
        self.safety_rules = self._load_safety_rules()
        
    def _load_safety_rules(self) -> Dict:
        """Load safety rules and policies"""
        return {
            'nsfw_threshold': 0.3,
            'copyright_threshold': 0.6,
            'ethical_threshold': 0.4,
            'overall_safety_threshold': 0.5,
            'blocked_content_types': ['malware', 'phishing', 'spam'],
            'restricted_keywords': [
                'explicit', 'adult', 'nsfw', 'xxx',
                'hate', 'violence', 'threat',
                'piracy', 'illegal', 'fraud'
            ],
            'safe_content_indicators': [
                'educational', 'family-friendly', 'safe',
                'verified', 'official', 'legitimate'
            ]
        }
    
    async def safe_content_recommendation(self, user_preferences: Dict, content_pool: List[Dict]) -> List[Dict]:
        """Recommend content with safety filtering"""
        safe_recommendations = []
        
        for content in content_pool:
            # Assess content safety
            assessment = await self.moderation_system.assess_content(
                content.get('content', ''),
                content.get('type', 'unknown'),
                content.get('source_url')
            )
            
            # Apply safety filters
            if assessment.safety_score >= self.safety_rules['overall_safety_threshold']:
                # Add safety metadata
                content['safety_assessment'] = {
                    'safety_score': assessment.safety_score,
                    'content_warnings': assessment.violations,
                    'assessment_id': assessment.content_id
                }
                safe_recommendations.append(content)
        
        return safe_recommendations
    
    async def safe_search_filtering(self, search_results: List[Dict]) -> List[Dict]:
        """Filter search results for safety"""
        filtered_results = []
        
        for result in search_results:
            # Quick domain check
            if 'url' in result:
                domain = urlparse(result['url']).netloc
                if domain in self.moderation_system.blocked_domains:
                    continue
            
            # Content analysis
            content_text = result.get('title', '') + ' ' + result.get('description', '')
            assessment = await self.moderation_system.assess_content(
                content_text,
                'text',
                result.get('url')
            )
            
            if assessment.safety_score >= self.safety_rules['overall_safety_threshold']:
                result['safety_verified'] = True
                result['safety_score'] = assessment.safety_score
                filtered_results.append(result)
        
        return filtered_results
    
    def validate_user_input(self, user_input: str) -> Tuple[bool, List[str]]:
        """Validate user input for safety"""
        violations = []
        
        # Check for restricted keywords
        for keyword in self.safety_rules['restricted_keywords']:
            if keyword.lower() in user_input.lower():
                violations.append(f"Restricted keyword detected: {keyword}")
        
        # Check input length
        if len(user_input) > 10000:
            violations.append("Input too long")
        
        # Check for malicious patterns
        malicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script injection
            r'javascript:',                # JavaScript URLs
            r'data:.*base64',             # Data URLs
            r'\\x[0-9a-f]{2}'            # Hex encoding
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                violations.append(f"Potentially malicious pattern detected")
                break
        
        is_safe = len(violations) == 0
        return is_safe, violations

# API endpoints for the safety system
class SafetyAPI:
    """REST API for the AI Safety System"""
    
    def __init__(self):
        self.guardrail_system = AIGuardrailSystem()
    
    async def assess_content_endpoint(self, content_data: Dict) -> Dict:
        """API endpoint for content assessment"""
        try:
            content = content_data.get('content')
            content_type = content_data.get('type', 'unknown')
            source_url = content_data.get('source_url')
            
            assessment = await self.guardrail_system.moderation_system.assess_content(
                content, content_type, source_url
            )
            
            return {
                'status': 'success',
                'assessment': asdict(assessment),
                'safe_to_use': assessment.safety_score >= 0.5
            }
            
        except Exception as e:
            logger.error(f"Content assessment error: {str(e)}")
            return {
                'status': 'error',
                'message': 'Content assessment failed',
                'safe_to_use': False
            }
    
    async def validate_input_endpoint(self, input_data: Dict) -> Dict:
        """API endpoint for input validation"""
        try:
            user_input = input_data.get('input', '')
            is_safe, violations = self.guardrail_system.validate_user_input(user_input)
            
            return {
                'status': 'success',
                'is_safe': is_safe,
                'violations': violations
            }
            
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            return {
                'status': 'error',
                'message': 'Input validation failed',
                'is_safe': False
            }
    
    def get_safety_summary_endpoint(self) -> Dict:
        """API endpoint for safety summary"""
        try:
            summary = self.guardrail_system.moderation_system.get_safety_summary()
            return {
                'status': 'success',
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Safety summary error: {str(e)}")
            return {
                'status': 'error',
                'message': 'Failed to get safety summary'
            }

if __name__ == "__main__":
    async def test_safety_system():
        """Test the AI safety system"""
        safety_api = SafetyAPI()
        
        # Test content assessment
        test_content = {
            'content': 'This is a safe educational video about science',
            'type': 'text',
            'source_url': 'https://youtube.com/watch?v=example'
        }
        
        result = await safety_api.assess_content_endpoint(test_content)
        print(f"Content Assessment Result: {json.dumps(result, indent=2)}")
        
        # Test input validation
        test_input = {
            'input': 'Can you help me find educational content about astronomy?'
        }
        
        validation_result = await safety_api.validate_input_endpoint(test_input)
        print(f"Input Validation Result: {json.dumps(validation_result, indent=2)}")
        
        # Test safety summary
        summary_result = safety_api.get_safety_summary_endpoint()
        print(f"Safety Summary: {json.dumps(summary_result, indent=2)}")
    
    # Run test
    asyncio.run(test_safety_system())