#!/usr/bin/env python3
"""
Ethical AI Recommendation Engine with Content Filtering
Advanced recommendation system with O3-mini reasoning and safety guardrails
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from collections import defaultdict, deque
import hashlib
import time
import random
from urllib.parse import urlparse
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommendation_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ContentItem:
    """Represents a content item in the system"""
    content_id: str
    title: str
    description: str
    content_type: str  # video, music, book, etc.
    categories: List[str]
    tags: List[str]
    duration: int  # in seconds
    rating: float
    view_count: int
    upload_date: datetime
    source_url: str
    creator_id: str
    language: str
    content_warnings: List[str]
    safety_score: float
    embedding: Optional[np.ndarray] = None
    
@dataclass
class UserProfile:
    """User profile with preferences and safety settings"""
    user_id: str
    age_group: str  # child, teen, adult
    content_preferences: Dict[str, float]
    safety_level: str  # strict, moderate, relaxed
    blocked_categories: List[str]
    blocked_creators: List[str]
    explicit_content: bool
    language_preferences: List[str]
    interaction_history: List[Dict]
    last_updated: datetime

@dataclass
class RecommendationResult:
    """Result of a recommendation query"""
    user_id: str
    recommended_items: List[Dict]
    reasoning_chain: List[str]
    safety_checks: List[str]
    diversity_score: float
    novelty_score: float
    confidence_score: float
    timestamp: datetime
    recommendation_type: str  # collaborative, content-based, hybrid

class O3MiniRecommendationReasoner:
    """O3-mini style reasoning for ethical recommendations"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reasoning_steps = []
        
        # Initialize embedding model
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model.to(self.device)
        
        # Ethical guidelines
        self.ethical_principles = {
            'diversity': 0.3,      # Promote diverse content
            'fairness': 0.25,      # Fair representation of creators
            'safety': 0.2,         # User safety and wellbeing
            'privacy': 0.15,       # User privacy protection
            'transparency': 0.1    # Explainable recommendations
        }
        
        # Content safety thresholds
        self.safety_thresholds = {
            'strict': 0.9,
            'moderate': 0.7,
            'relaxed': 0.5
        }
    
    async def reason_through_recommendation(self, user_profile: UserProfile, 
                                         candidate_items: List[ContentItem],
                                         context: Dict = None) -> RecommendationResult:
        """Apply O3-mini reasoning to generate ethical recommendations"""
        
        reasoning_chain = []
        safety_checks = []
        
        # Step 1: User Profile Analysis
        reasoning_chain.append("Analyzing user profile and safety requirements")
        safe_categories = await self._analyze_user_safety_requirements(user_profile)
        reasoning_chain.append(f"Safe categories identified: {len(safe_categories)}")
        
        # Step 2: Content Filtering
        reasoning_chain.append("Applying safety-first content filtering")
        filtered_items = await self._apply_safety_filtering(
            candidate_items, user_profile, safe_categories
        )
        reasoning_chain.append(f"Filtered {len(candidate_items)} to {len(filtered_items)} safe items")
        
        # Step 3: Ethical Scoring
        reasoning_chain.append("Computing ethical recommendation scores")
        scored_items = await self._compute_ethical_scores(
            filtered_items, user_profile, context or {}
        )
        
        # Step 4: Diversity Enhancement
        reasoning_chain.append("Enhancing recommendation diversity")
        diverse_items = await self._enhance_diversity(scored_items, user_profile)
        
        # Step 5: Final Selection and Ranking
        reasoning_chain.append("Selecting final recommendations with ethical constraints")
        final_recommendations = await self._select_final_recommendations(
            diverse_items, user_profile, reasoning_chain
        )
        
        # Step 6: Quality Assessment
        diversity_score = self._calculate_diversity_score(final_recommendations)
        novelty_score = self._calculate_novelty_score(final_recommendations, user_profile)
        confidence_score = self._calculate_confidence_score(final_recommendations)
        
        reasoning_chain.append(f"Quality metrics - Diversity: {diversity_score:.3f}, "
                             f"Novelty: {novelty_score:.3f}, Confidence: {confidence_score:.3f}")
        
        # Safety verification
        safety_checks = await self._verify_recommendations_safety(
            final_recommendations, user_profile
        )
        
        return RecommendationResult(
            user_id=user_profile.user_id,
            recommended_items=final_recommendations,
            reasoning_chain=reasoning_chain,
            safety_checks=safety_checks,
            diversity_score=diversity_score,
            novelty_score=novelty_score,
            confidence_score=confidence_score,
            timestamp=datetime.now(),
            recommendation_type='hybrid_ethical'
        )
    
    async def _analyze_user_safety_requirements(self, user_profile: UserProfile) -> Set[str]:
        """Analyze user profile to determine safety requirements"""
        safe_categories = set()
        
        # Age-based restrictions
        if user_profile.age_group == 'child':
            safe_categories.update(['educational', 'family', 'animation', 'music'])
        elif user_profile.age_group == 'teen':
            safe_categories.update(['educational', 'entertainment', 'music', 'gaming', 'sports'])
        else:  # adult
            safe_categories.update(['news', 'documentary', 'entertainment', 'education', 'lifestyle'])
        
        # Remove blocked categories
        safe_categories -= set(user_profile.blocked_categories)
        
        return safe_categories
    
    async def _apply_safety_filtering(self, candidate_items: List[ContentItem], 
                                   user_profile: UserProfile,
                                   safe_categories: Set[str]) -> List[ContentItem]:
        """Apply comprehensive safety filtering"""
        filtered_items = []
        safety_threshold = self.safety_thresholds[user_profile.safety_level]
        
        for item in candidate_items:
            # Safety score check
            if item.safety_score < safety_threshold:
                continue
            
            # Category safety check
            if not any(cat in safe_categories for cat in item.categories):
                continue
            
            # Blocked creator check
            if item.creator_id in user_profile.blocked_creators:
                continue
            
            # Explicit content check
            if not user_profile.explicit_content and 'explicit' in item.content_warnings:
                continue
            
            # Language preference check
            if (user_profile.language_preferences and 
                item.language not in user_profile.language_preferences):
                continue
            
            # Content warning check
            if user_profile.safety_level == 'strict' and item.content_warnings:
                continue
            
            filtered_items.append(item)
        
        return filtered_items
    
    async def _compute_ethical_scores(self, items: List[ContentItem], 
                                    user_profile: UserProfile,
                                    context: Dict) -> List[Tuple[ContentItem, float, Dict]]:
        """Compute ethical recommendation scores"""
        scored_items = []
        
        for item in items:
            score_breakdown = {}
            
            # Content relevance score
            relevance_score = await self._compute_relevance_score(item, user_profile)
            score_breakdown['relevance'] = relevance_score
            
            # Diversity promotion score
            diversity_score = await self._compute_diversity_promotion_score(item, context)
            score_breakdown['diversity'] = diversity_score
            
            # Creator fairness score
            fairness_score = await self._compute_creator_fairness_score(item, context)
            score_breakdown['fairness'] = fairness_score
            
            # Safety reinforcement score
            safety_score = item.safety_score
            score_breakdown['safety'] = safety_score
            
            # Quality score
            quality_score = await self._compute_quality_score(item)
            score_breakdown['quality'] = quality_score
            
            # Compute weighted ethical score
            ethical_score = (
                relevance_score * 0.3 +
                diversity_score * 0.2 +
                fairness_score * 0.15 +
                safety_score * 0.2 +
                quality_score * 0.15
            )
            
            scored_items.append((item, ethical_score, score_breakdown))
        
        # Sort by ethical score
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items
    
    async def _compute_relevance_score(self, item: ContentItem, user_profile: UserProfile) -> float:
        """Compute content relevance to user preferences"""
        relevance = 0.0
        
        # Category preferences
        for category in item.categories:
            if category in user_profile.content_preferences:
                relevance += user_profile.content_preferences[category] * 0.4
        
        # Tag preferences (derived from interaction history)
        tag_preferences = self._extract_tag_preferences(user_profile.interaction_history)
        for tag in item.tags:
            if tag in tag_preferences:
                relevance += tag_preferences[tag] * 0.3
        
        # Content embedding similarity
        if item.embedding is not None:
            user_embedding = self._compute_user_embedding(user_profile)
            if user_embedding is not None:
                similarity = cosine_similarity([item.embedding], [user_embedding])[0][0]
                relevance += similarity * 0.3
        
        return min(relevance, 1.0)
    
    async def _compute_diversity_promotion_score(self, item: ContentItem, context: Dict) -> float:
        """Promote diverse content to avoid filter bubbles"""
        diversity_score = 0.5  # Base score
        
        # Check if content promotes underrepresented creators
        if self._is_underrepresented_creator(item.creator_id, context):
            diversity_score += 0.3
        
        # Check category diversity
        recommended_categories = context.get('recommended_categories', set())
        if not any(cat in recommended_categories for cat in item.categories):
            diversity_score += 0.2
        
        return min(diversity_score, 1.0)
    
    async def _compute_creator_fairness_score(self, item: ContentItem, context: Dict) -> float:
        """Ensure fair representation of content creators"""
        fairness_score = 0.5  # Base score
        
        # Check creator recommendation frequency
        creator_frequency = context.get('creator_frequency', {})
        creator_recs = creator_frequency.get(item.creator_id, 0)
        
        # Boost underrepresented creators
        if creator_recs < 3:  # Threshold for fair representation
            fairness_score += 0.3
        elif creator_recs > 10:  # Prevent over-representation
            fairness_score -= 0.2
        
        return max(0.0, min(fairness_score, 1.0))
    
    async def _compute_quality_score(self, item: ContentItem) -> float:
        """Compute content quality score"""
        # Normalize rating (assuming 0-5 scale)
        rating_score = item.rating / 5.0
        
        # View count normalization (log scale)
        view_score = min(np.log10(max(item.view_count, 1)) / 6.0, 1.0)
        
        # Recency bonus (content uploaded recently gets slight boost)
        days_old = (datetime.now() - item.upload_date).days
        recency_score = max(0, 1.0 - days_old / 365.0) * 0.1
        
        quality_score = rating_score * 0.6 + view_score * 0.3 + recency_score
        return min(quality_score, 1.0)
    
    async def _enhance_diversity(self, scored_items: List[Tuple[ContentItem, float, Dict]], 
                               user_profile: UserProfile) -> List[Tuple[ContentItem, float, Dict]]:
        """Enhance diversity in recommendations using MMR-like approach"""
        if len(scored_items) <= 1:
            return scored_items
        
        diverse_items = []
        remaining_items = scored_items.copy()
        
        # Select first item (highest scored)
        diverse_items.append(remaining_items.pop(0))
        
        # MMR-style selection for remaining items
        lambda_param = 0.7  # Balance between relevance and diversity
        
        while remaining_items and len(diverse_items) < min(20, len(scored_items)):
            best_item = None
            best_score = -1
            best_index = -1
            
            for i, (item, score, breakdown) in enumerate(remaining_items):
                # Calculate diversity penalty
                diversity_penalty = 0
                for selected_item, _, _ in diverse_items:
                    similarity = self._compute_content_similarity(item, selected_item)
                    diversity_penalty = max(diversity_penalty, similarity)
                
                # MMR score
                mmr_score = lambda_param * score - (1 - lambda_param) * diversity_penalty
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = (item, score, breakdown)
                    best_index = i
            
            if best_item:
                diverse_items.append(best_item)
                remaining_items.pop(best_index)
        
        return diverse_items
    
    async def _select_final_recommendations(self, diverse_items: List[Tuple[ContentItem, float, Dict]],
                                          user_profile: UserProfile, 
                                          reasoning_chain: List[str]) -> List[Dict]:
        """Select and format final recommendations"""
        max_recommendations = 10
        final_recommendations = []
        
        for item, score, breakdown in diverse_items[:max_recommendations]:
            recommendation = {
                'content_id': item.content_id,
                'title': item.title,
                'description': item.description,
                'content_type': item.content_type,
                'categories': item.categories,
                'duration': item.duration,
                'rating': item.rating,
                'source_url': item.source_url,
                'creator_id': item.creator_id,
                'safety_score': item.safety_score,
                'recommendation_score': score,
                'score_breakdown': breakdown,
                'content_warnings': item.content_warnings,
                'reasoning': f"Recommended based on ethical AI analysis with score {score:.3f}"
            }
            
            final_recommendations.append(recommendation)
        
        reasoning_chain.append(f"Selected {len(final_recommendations)} final recommendations")
        return final_recommendations
    
    def _extract_tag_preferences(self, interaction_history: List[Dict]) -> Dict[str, float]:
        """Extract tag preferences from user interaction history"""
        tag_counts = defaultdict(int)
        total_interactions = len(interaction_history)
        
        for interaction in interaction_history:
            if 'tags' in interaction:
                for tag in interaction['tags']:
                    weight = 1.0
                    if interaction.get('interaction_type') == 'liked':
                        weight = 2.0
                    elif interaction.get('interaction_type') == 'shared':
                        weight = 1.5
                    
                    tag_counts[tag] += weight
        
        # Normalize to preferences
        tag_preferences = {}
        for tag, count in tag_counts.items():
            tag_preferences[tag] = min(count / max(total_interactions * 0.1, 1), 1.0)
        
        return tag_preferences
    
    def _compute_user_embedding(self, user_profile: UserProfile) -> Optional[np.ndarray]:
        """Compute user embedding from interaction history"""
        if not user_profile.interaction_history:
            return None
        
        # Extract text from user interactions
        texts = []
        for interaction in user_profile.interaction_history[-50]:  # Last 50 interactions
            if 'title' in interaction:
                texts.append(interaction['title'])
            if 'description' in interaction:
                texts.append(interaction['description'])
        
        if not texts:
            return None
        
        # Compute embedding
        combined_text = ' '.join(texts)
        inputs = self.tokenizer(combined_text, return_tensors='pt', 
                              truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        
        return embedding
    
    def _is_underrepresented_creator(self, creator_id: str, context: Dict) -> bool:
        """Check if creator is underrepresented"""
        creator_stats = context.get('creator_stats', {})
        creator_recs = creator_stats.get(creator_id, {}).get('total_recommendations', 0)
        avg_recommendations = context.get('avg_creator_recommendations', 10)
        
        return creator_recs < avg_recommendations * 0.5
    
    def _compute_content_similarity(self, item1: ContentItem, item2: ContentItem) -> float:
        """Compute similarity between two content items"""
        similarity = 0.0
        
        # Category similarity
        common_categories = set(item1.categories) & set(item2.categories)
        category_sim = len(common_categories) / max(len(set(item1.categories) | set(item2.categories)), 1)
        similarity += category_sim * 0.4
        
        # Tag similarity
        common_tags = set(item1.tags) & set(item2.tags)
        tag_sim = len(common_tags) / max(len(set(item1.tags) | set(item2.tags)), 1)
        similarity += tag_sim * 0.3
        
        # Creator similarity
        if item1.creator_id == item2.creator_id:
            similarity += 0.3
        
        return similarity
    
    def _calculate_diversity_score(self, recommendations: List[Dict]) -> float:
        """Calculate diversity score of recommendations"""
        if len(recommendations) <= 1:
            return 1.0
        
        total_similarity = 0
        count = 0
        
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                # Category diversity
                cats1 = set(recommendations[i]['categories'])
                cats2 = set(recommendations[j]['categories'])
                cat_similarity = len(cats1 & cats2) / len(cats1 | cats2) if (cats1 | cats2) else 0
                
                # Creator diversity
                creator_similarity = 1.0 if recommendations[i]['creator_id'] == recommendations[j]['creator_id'] else 0.0
                
                similarity = cat_similarity * 0.7 + creator_similarity * 0.3
                total_similarity += similarity
                count += 1
        
        avg_similarity = total_similarity / count if count > 0 else 0
        return 1.0 - avg_similarity  # Higher diversity = lower similarity
    
    def _calculate_novelty_score(self, recommendations: List[Dict], user_profile: UserProfile) -> float:
        """Calculate novelty score based on user's interaction history"""
        if not user_profile.interaction_history:
            return 1.0
        
        # Get categories from user history
        historical_categories = set()
        for interaction in user_profile.interaction_history:
            if 'categories' in interaction:
                historical_categories.update(interaction['categories'])
        
        novel_items = 0
        for rec in recommendations:
            rec_categories = set(rec['categories'])
            # If recommendation has categories not in user history, it's novel
            if not rec_categories.issubset(historical_categories):
                novel_items += 1
        
        return novel_items / len(recommendations) if recommendations else 0
    
    def _calculate_confidence_score(self, recommendations: List[Dict]) -> float:
        """Calculate confidence score for recommendations"""
        if not recommendations:
            return 0.0
        
        # Base confidence on recommendation scores
        scores = [rec['recommendation_score'] for rec in recommendations]
        avg_score = sum(scores) / len(scores)
        score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        
        # High confidence when scores are high and consistent
        confidence = avg_score * (1 - min(score_variance, 0.5))
        return confidence
    
    async def _verify_recommendations_safety(self, recommendations: List[Dict], 
                                           user_profile: UserProfile) -> List[str]:
        """Final safety verification of recommendations"""
        safety_checks = []
        
        # Check safety scores
        safety_threshold = self.safety_thresholds[user_profile.safety_level]
        unsafe_items = [r for r in recommendations if r['safety_score'] < safety_threshold]
        if not unsafe_items:
            safety_checks.append("All recommendations meet safety threshold")
        else:
            safety_checks.append(f"Warning: {len(unsafe_items)} items below safety threshold")
        
        # Check content warnings
        items_with_warnings = [r for r in recommendations if r['content_warnings']]
        if items_with_warnings:
            safety_checks.append(f"{len(items_with_warnings)} items have content warnings")
        
        # Check age appropriateness
        if user_profile.age_group == 'child':
            inappropriate_categories = ['adult', 'mature', 'violence', 'horror']
            for rec in recommendations:
                if any(cat in inappropriate_categories for cat in rec['categories']):
                    safety_checks.append("Warning: Age-inappropriate content detected")
                    break
        
        return safety_checks

class EthicalRecommendationEngine:
    """Main recommendation engine with ethical AI guardrails"""
    
    def __init__(self, db_path: str = "recommendations.db"):
        self.db_path = db_path
        self.reasoner = O3MiniRecommendationReasoner()
        self.content_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.content_embeddings = None
        self.content_items = []
        self.user_profiles = {}
        
        self._init_database()
        self._load_content_data()
    
    def _init_database(self):
        """Initialize recommendation database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Content table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_items (
                content_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                content_type TEXT,
                categories TEXT,
                tags TEXT,
                duration INTEGER,
                rating REAL,
                view_count INTEGER,
                upload_date DATETIME,
                source_url TEXT,
                creator_id TEXT,
                language TEXT,
                content_warnings TEXT,
                safety_score REAL,
                embedding_json TEXT
            )
        ''')
        
        # User profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                age_group TEXT,
                content_preferences TEXT,
                safety_level TEXT,
                blocked_categories TEXT,
                blocked_creators TEXT,
                explicit_content BOOLEAN,
                language_preferences TEXT,
                interaction_history TEXT,
                last_updated DATETIME
            )
        ''')
        
        # Recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                recommended_items TEXT,
                reasoning_chain TEXT,
                safety_checks TEXT,
                diversity_score REAL,
                novelty_score REAL,
                confidence_score REAL,
                timestamp DATETIME,
                recommendation_type TEXT
            )
        ''')
        
        # User interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                content_id TEXT,
                interaction_type TEXT,
                timestamp DATETIME,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_content_data(self):
        """Load content data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM content_items')
        rows = cursor.fetchall()
        
        self.content_items = []
        for row in rows:
            content_item = ContentItem(
                content_id=row[0],
                title=row[1],
                description=row[2],
                content_type=row[3],
                categories=json.loads(row[4]) if row[4] else [],
                tags=json.loads(row[5]) if row[5] else [],
                duration=row[6],
                rating=row[7],
                view_count=row[8],
                upload_date=datetime.fromisoformat(row[9]) if row[9] else datetime.now(),
                source_url=row[10],
                creator_id=row[11],
                language=row[12],
                content_warnings=json.loads(row[13]) if row[13] else [],
                safety_score=row[14],
                embedding=np.array(json.loads(row[15])) if row[15] else None
            )
            self.content_items.append(content_item)
        
        conn.close()
        logger.info(f"Loaded {len(self.content_items)} content items")
    
    async def get_recommendations(self, user_id: str, num_recommendations: int = 10,
                                recommendation_type: str = 'hybrid') -> RecommendationResult:
        """Get ethical recommendations for a user"""
        
        # Load or create user profile
        user_profile = await self._get_user_profile(user_id)
        
        # Get candidate items based on recommendation type
        if recommendation_type == 'collaborative':
            candidate_items = await self._get_collaborative_candidates(user_profile)
        elif recommendation_type == 'content_based':
            candidate_items = await self._get_content_based_candidates(user_profile)
        else:  # hybrid
            candidate_items = await self._get_hybrid_candidates(user_profile)
        
        # Build recommendation context
        context = await self._build_recommendation_context(user_profile)
        
        # Apply O3-mini reasoning
        recommendation_result = await self.reasoner.reason_through_recommendation(
            user_profile, candidate_items, context
        )
        
        # Store recommendation
        await self._store_recommendation(recommendation_result)
        
        # Update user interaction context
        await self._update_user_context(user_id, recommendation_result)
        
        return recommendation_result
    
    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile"""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        
        if row:
            user_profile = UserProfile(
                user_id=row[0],
                age_group=row[1],
                content_preferences=json.loads(row[2]) if row[2] else {},
                safety_level=row[3] or 'moderate',
                blocked_categories=json.loads(row[4]) if row[4] else [],
                blocked_creators=json.loads(row[5]) if row[5] else [],
                explicit_content=bool(row[6]),
                language_preferences=json.loads(row[7]) if row[7] else ['en'],
                interaction_history=json.loads(row[8]) if row[8] else [],
                last_updated=datetime.fromisoformat(row[9]) if row[9] else datetime.now()
            )
        else:
            # Create default profile
            user_profile = UserProfile(
                user_id=user_id,
                age_group='adult',
                content_preferences={'entertainment': 0.5, 'education': 0.3},
                safety_level='moderate',
                blocked_categories=[],
                blocked_creators=[],
                explicit_content=False,
                language_preferences=['en'],
                interaction_history=[],
                last_updated=datetime.now()
            )
            
            # Store new profile
            cursor.execute('''
                INSERT INTO user_profiles 
                (user_id, age_group, content_preferences, safety_level,
                 blocked_categories, blocked_creators, explicit_content,
                 language_preferences, interaction_history, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_profile.user_id,
                user_profile.age_group,
                json.dumps(user_profile.content_preferences),
                user_profile.safety_level,
                json.dumps(user_profile.blocked_categories),
                json.dumps(user_profile.blocked_creators),
                user_profile.explicit_content,
                json.dumps(user_profile.language_preferences),
                json.dumps(user_profile.interaction_history),
                user_profile.last_updated.isoformat()
            ))
        
        conn.commit()
        conn.close()
        
        self.user_profiles[user_id] = user_profile
        return user_profile
    
    async def _get_collaborative_candidates(self, user_profile: UserProfile) -> List[ContentItem]:
        """Get candidates using collaborative filtering"""
        # Find similar users based on interaction patterns
        similar_users = await self._find_similar_users(user_profile)
        
        # Get content liked by similar users
        candidate_items = []
        for similar_user_id in similar_users[:10]:  # Top 10 similar users
            similar_profile = await self._get_user_profile(similar_user_id)
            for interaction in similar_profile.interaction_history:
                if interaction.get('interaction_type') in ['liked', 'shared', 'completed']:
                    content_id = interaction.get('content_id')
                    content_item = self._get_content_by_id(content_id)
                    if content_item and content_item not in candidate_items:
                        candidate_items.append(content_item)
        
        return candidate_items[:100]  # Limit candidates
    
    async def _get_content_based_candidates(self, user_profile: UserProfile) -> List[ContentItem]:
        """Get candidates using content-based filtering"""
        candidate_items = []
        
        # Get items similar to user's liked content
        liked_items = []
        for interaction in user_profile.interaction_history:
            if interaction.get('interaction_type') in ['liked', 'shared']:
                content_id = interaction.get('content_id')
                content_item = self._get_content_by_id(content_id)
                if content_item:
                    liked_items.append(content_item)
        
        # Find similar content
        for liked_item in liked_items[-20]:  # Last 20 liked items
            similar_items = await self._find_similar_content(liked_item)
            candidate_items.extend(similar_items[:10])
        
        # Remove duplicates
        seen_ids = set()
        unique_candidates = []
        for item in candidate_items:
            if item.content_id not in seen_ids:
                unique_candidates.append(item)
                seen_ids.add(item.content_id)
        
        return unique_candidates[:100]
    
    async def _get_hybrid_candidates(self, user_profile: UserProfile) -> List[ContentItem]:
        """Get candidates using hybrid approach"""
        # Combine collaborative and content-based
        collab_candidates = await self._get_collaborative_candidates(user_profile)
        content_candidates = await self._get_content_based_candidates(user_profile)
        
        # Merge and deduplicate
        all_candidates = collab_candidates + content_candidates
        seen_ids = set()
        unique_candidates = []
        
        for item in all_candidates:
            if item.content_id not in seen_ids:
                unique_candidates.append(item)
                seen_ids.add(item.content_id)
        
        # Add some popular content for diversity
        popular_items = sorted(self.content_items, key=lambda x: x.view_count, reverse=True)[:20]
        for item in popular_items:
            if item.content_id not in seen_ids:
                unique_candidates.append(item)
                seen_ids.add(item.content_id)
        
        return unique_candidates[:150]
    
    async def _build_recommendation_context(self, user_profile: UserProfile) -> Dict:
        """Build context for recommendation reasoning"""
        context = {
            'recommended_categories': set(),
            'creator_frequency': defaultdict(int),
            'creator_stats': {},
            'avg_creator_recommendations': 0
        }
        
        # Analyze recent recommendations for this user
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT recommended_items FROM recommendations 
            WHERE user_id = ? AND timestamp > datetime('now', '-7 days')
            ORDER BY timestamp DESC
        ''', (user_profile.user_id,))
        
        recent_recs = cursor.fetchall()
        for row in recent_recs:
            items = json.loads(row[0])
            for item in items:
                context['recommended_categories'].update(item['categories'])
                context['creator_frequency'][item['creator_id']] += 1
        
        # Get global creator statistics
        cursor.execute('''
            SELECT creator_id, COUNT(*) as rec_count
            FROM (
                SELECT json_extract(value, '$.creator_id') as creator_id
                FROM recommendations, json_each(recommended_items)
                WHERE timestamp > datetime('now', '-30 days')
            )
            GROUP BY creator_id
        ''')
        
        creator_stats = cursor.fetchall()
        total_recs = sum(count for _, count in creator_stats)
        context['avg_creator_recommendations'] = total_recs / max(len(creator_stats), 1)
        
        for creator_id, count in creator_stats:
            context['creator_stats'][creator_id] = {
                'total_recommendations': count,
                'recommendation_rate': count / total_recs if total_recs > 0 else 0
            }
        
        conn.close()
        return context
    
    async def _find_similar_users(self, user_profile: UserProfile) -> List[str]:
        """Find users with similar preferences and behavior"""
        # Simplified similarity based on content preferences and categories
        similar_users = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id, content_preferences, interaction_history FROM user_profiles')
        rows = cursor.fetchall()
        
        user_categories = self._extract_user_categories(user_profile.interaction_history)
        
        similarities = []
        for row in rows:
            other_user_id = row[0]
            if other_user_id == user_profile.user_id:
                continue
            
            other_prefs = json.loads(row[1]) if row[1] else {}
            other_history = json.loads(row[2]) if row[2] else []
            other_categories = self._extract_user_categories(other_history)
            
            # Calculate similarity
            pref_similarity = self._calculate_preference_similarity(
                user_profile.content_preferences, other_prefs
            )
            category_similarity = self._calculate_category_similarity(
                user_categories, other_categories
            )
            
            overall_similarity = pref_similarity * 0.6 + category_similarity * 0.4
            similarities.append((other_user_id, overall_similarity))
        
        # Sort by similarity and return top users
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_users = [user_id for user_id, sim in similarities[:20]]
        
        conn.close()
        return similar_users
    
    def _extract_user_categories(self, interaction_history: List[Dict]) -> Dict[str, int]:
        """Extract category preferences from interaction history"""
        categories = defaultdict(int)
        
        for interaction in interaction_history:
            if 'categories' in interaction:
                weight = 1
                if interaction.get('interaction_type') == 'liked':
                    weight = 3
                elif interaction.get('interaction_type') == 'shared':
                    weight = 2
                
                for category in interaction['categories']:
                    categories[category] += weight
        
        return dict(categories)
    
    def _calculate_preference_similarity(self, prefs1: Dict, prefs2: Dict) -> float:
        """Calculate similarity between two preference dictionaries"""
        all_categories = set(prefs1.keys()) | set(prefs2.keys())
        if not all_categories:
            return 0.0
        
        similarity = 0.0
        for category in all_categories:
            p1 = prefs1.get(category, 0)
            p2 = prefs2.get(category, 0)
            similarity += 1 - abs(p1 - p2)  # Similarity based on preference difference
        
        return similarity / len(all_categories)
    
    def _calculate_category_similarity(self, cats1: Dict, cats2: Dict) -> float:
        """Calculate similarity between category interaction patterns"""
        all_categories = set(cats1.keys()) | set(cats2.keys())
        if not all_categories:
            return 0.0
        
        # Normalize counts
        total1 = sum(cats1.values()) or 1
        total2 = sum(cats2.values()) or 1
        
        similarity = 0.0
        for category in all_categories:
            ratio1 = cats1.get(category, 0) / total1
            ratio2 = cats2.get(category, 0) / total2
            similarity += 1 - abs(ratio1 - ratio2)
        
        return similarity / len(all_categories)
    
    async def _find_similar_content(self, target_item: ContentItem) -> List[ContentItem]:
        """Find content similar to target item"""
        similar_items = []
        
        for item in self.content_items:
            if item.content_id == target_item.content_id:
                continue
            
            similarity = self.reasoner._compute_content_similarity(target_item, item)
            if similarity > 0.3:  # Similarity threshold
                similar_items.append((item, similarity))
        
        # Sort by similarity and return top items
        similar_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, sim in similar_items[:20]]
    
    def _get_content_by_id(self, content_id: str) -> Optional[ContentItem]:
        """Get content item by ID"""
        for item in self.content_items:
            if item.content_id == content_id:
                return item
        return None
    
    async def _store_recommendation(self, result: RecommendationResult):
        """Store recommendation result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO recommendations 
            (user_id, recommended_items, reasoning_chain, safety_checks,
             diversity_score, novelty_score, confidence_score, timestamp, recommendation_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.user_id,
            json.dumps(result.recommended_items),
            json.dumps(result.reasoning_chain),
            json.dumps(result.safety_checks),
            result.diversity_score,
            result.novelty_score,
            result.confidence_score,
            result.timestamp.isoformat(),
            result.recommendation_type
        ))
        
        conn.commit()
        conn.close()
    
    async def _update_user_context(self, user_id: str, result: RecommendationResult):
        """Update user context based on recommendation"""
        # This would be expanded to track user responses to recommendations
        pass
    
    async def record_user_interaction(self, user_id: str, content_id: str, 
                                   interaction_type: str, metadata: Dict = None):
        """Record user interaction with content"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_interactions 
            (user_id, content_id, interaction_type, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id, content_id, interaction_type, 
            datetime.now().isoformat(),
            json.dumps(metadata or {})
        ))
        
        # Update user profile interaction history
        user_profile = await self._get_user_profile(user_id)
        content_item = self._get_content_by_id(content_id)
        
        if content_item:
            interaction_record = {
                'content_id': content_id,
                'interaction_type': interaction_type,
                'categories': content_item.categories,
                'tags': content_item.tags,
                'timestamp': datetime.now().isoformat()
            }
            
            user_profile.interaction_history.append(interaction_record)
            
            # Keep only last 1000 interactions
            if len(user_profile.interaction_history) > 1000:
                user_profile.interaction_history = user_profile.interaction_history[-1000:]
            
            # Update database
            cursor.execute('''
                UPDATE user_profiles 
                SET interaction_history = ?, last_updated = ?
                WHERE user_id = ?
            ''', (
                json.dumps(user_profile.interaction_history),
                datetime.now().isoformat(),
                user_id
            ))
        
        conn.commit()
        conn.close()
    
    async def add_content_item(self, content_item: ContentItem):
        """Add new content item to the system"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_json = json.dumps(content_item.embedding.tolist()) if content_item.embedding is not None else None
        
        cursor.execute('''
            INSERT OR REPLACE INTO content_items
            (content_id, title, description, content_type, categories, tags,
             duration, rating, view_count, upload_date, source_url, creator_id,
             language, content_warnings, safety_score, embedding_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            content_item.content_id,
            content_item.title,
            content_item.description,
            content_item.content_type,
            json.dumps(content_item.categories),
            json.dumps(content_item.tags),
            content_item.duration,
            content_item.rating,
            content_item.view_count,
            content_item.upload_date.isoformat(),
            content_item.source_url,
            content_item.creator_id,
            content_item.language,
            json.dumps(content_item.content_warnings),
            content_item.safety_score,
            embedding_json
        ))
        
        conn.commit()
        conn.close()
        
        # Add to memory
        self.content_items.append(content_item)
        
    def get_recommendation_stats(self) -> Dict:
        """Get recommendation system statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get basic stats
        cursor.execute('SELECT COUNT(*) FROM recommendations')
        total_recommendations = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT user_id) FROM recommendations')
        unique_users = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(diversity_score), AVG(novelty_score), AVG(confidence_score) FROM recommendations')
        avg_scores = cursor.fetchone()
        
        # Get recent activity
        cursor.execute('''
            SELECT COUNT(*) FROM recommendations 
            WHERE timestamp > datetime('now', '-24 hours')
        ''')
        recent_recommendations = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_recommendations': total_recommendations,
            'unique_users': unique_users,
            'average_diversity_score': avg_scores[0] or 0,
            'average_novelty_score': avg_scores[1] or 0,
            'average_confidence_score': avg_scores[2] or 0,
            'recommendations_24h': recent_recommendations,
            'content_items': len(self.content_items),
            'last_updated': datetime.now().isoformat()
        }

# API endpoints for the recommendation engine
class RecommendationAPI:
    """REST API for the Ethical Recommendation Engine"""
    
    def __init__(self):
        self.engine = EthicalRecommendationEngine()
    
    async def get_recommendations_endpoint(self, user_id: str, num_recs: int = 10) -> Dict:
        """API endpoint for getting recommendations"""
        try:
            result = await self.engine.get_recommendations(user_id, num_recs)
            
            return {
                'status': 'success',
                'recommendations': result.recommended_items,
                'reasoning': result.reasoning_chain,
                'safety_checks': result.safety_checks,
                'quality_metrics': {
                    'diversity_score': result.diversity_score,
                    'novelty_score': result.novelty_score,
                    'confidence_score': result.confidence_score
                },
                'recommendation_type': result.recommendation_type
            }
            
        except Exception as e:
            logger.error(f"Recommendation error: {str(e)}")
            return {
                'status': 'error',
                'message': 'Failed to generate recommendations'
            }
    
    async def record_interaction_endpoint(self, user_id: str, content_id: str, 
                                        interaction_type: str) -> Dict:
        """API endpoint for recording user interactions"""
        try:
            await self.engine.record_user_interaction(user_id, content_id, interaction_type)
            
            return {
                'status': 'success',
                'message': 'Interaction recorded successfully'
            }
            
        except Exception as e:
            logger.error(f"Interaction recording error: {str(e)}")
            return {
                'status': 'error',
                'message': 'Failed to record interaction'
            }
    
    def get_stats_endpoint(self) -> Dict:
        """API endpoint for recommendation statistics"""
        try:
            stats = self.engine.get_recommendation_stats()
            return {
                'status': 'success',
                'statistics': stats
            }
            
        except Exception as e:
            logger.error(f"Stats error: {str(e)}")
            return {
                'status': 'error',
                'message': 'Failed to get statistics'
            }

if __name__ == "__main__":
    async def test_recommendation_engine():
        """Test the recommendation engine"""
        api = RecommendationAPI()
        
        # Test recommendations
        result = await api.get_recommendations_endpoint('test_user_123', 5)
        print(f"Recommendation Result: {json.dumps(result, indent=2)}")
        
        # Test interaction recording
        interaction_result = await api.record_interaction_endpoint(
            'test_user_123', 'content_001', 'liked'
        )
        print(f"Interaction Result: {json.dumps(interaction_result, indent=2)}")
        
        # Test stats
        stats_result = api.get_stats_endpoint()
        print(f"Stats Result: {json.dumps(stats_result, indent=2)}")
    
    # Run test
    asyncio.run(test_recommendation_engine())