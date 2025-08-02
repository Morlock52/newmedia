# 📡 NEXUS Platform API Reference

Complete API documentation for all services and endpoints in the NEXUS Media Server Platform.

## Table of Contents

- [Core Media Services](#core-media-services)
- [AI/ML Nexus APIs](#aiml-nexus-apis)
- [AR/VR WebXR APIs](#arvr-webxr-apis)
- [Web3 Blockchain APIs](#web3-blockchain-apis)
- [Voice AI System APIs](#voice-ai-system-apis)
- [Monitoring & Analytics APIs](#monitoring--analytics-apis)
- [Authentication & Security](#authentication--security)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

---

## Core Media Services

### Jellyfin Media Server API

**Base URL**: `http://localhost:8096/api`

#### Authentication
```bash
# Get authentication token
POST /Users/authenticatebyname
Content-Type: application/json

{
  "Username": "admin",
  "Pw": "password"
}

# Response
{
  "AccessToken": "jwt_token_here",
  "ServerId": "server_id",
  "User": { ... }
}
```

#### Media Library
```bash
# Get all libraries
GET /Items
Authorization: MediaBrowser Token="jwt_token_here"

# Get items from library
GET /Items?ParentId={library_id}&Recursive=true
Authorization: MediaBrowser Token="jwt_token_here"

# Get item details
GET /Items/{item_id}
Authorization: MediaBrowser Token="jwt_token_here"
```

#### Playback Control
```bash
# Start playback session
POST /Sessions/{session_id}/Playing
Authorization: MediaBrowser Token="jwt_token_here"
Content-Type: application/json

{
  "ItemIds": ["item_id"],
  "StartPositionTicks": 0
}

# Pause/Resume
POST /Sessions/{session_id}/Playing/Pause
POST /Sessions/{session_id}/Playing/Unpause
Authorization: MediaBrowser Token="jwt_token_here"
```

### *arr Suite APIs

#### Sonarr API (TV Shows)
**Base URL**: `http://localhost:8989/api/v3`

```bash
# Get all series
GET /series
X-Api-Key: your_api_key

# Add new series
POST /series
X-Api-Key: your_api_key
Content-Type: application/json

{
  "title": "Series Name",
  "tvdbId": 123456,
  "qualityProfileId": 1,
  "rootFolderPath": "/media/tv"
}

# Search for episodes
POST /command
X-Api-Key: your_api_key
Content-Type: application/json

{
  "name": "SeriesSearch",
  "seriesId": 1
}
```

#### Radarr API (Movies)
**Base URL**: `http://localhost:7878/api/v3`

```bash
# Get all movies
GET /movie
X-Api-Key: your_api_key

# Add new movie
POST /movie
X-Api-Key: your_api_key
Content-Type: application/json

{
  "title": "Movie Title",
  "tmdbId": 123456,
  "qualityProfileId": 1,
  "rootFolderPath": "/media/movies"
}

# Manual search
POST /command
X-Api-Key: your_api_key
Content-Type: application/json

{
  "name": "MoviesSearch",
  "movieIds": [1, 2, 3]
}
```

#### Prowlarr API (Indexers)
**Base URL**: `http://localhost:9696/api/v1`

```bash
# Get all indexers
GET /indexer
X-Api-Key: your_api_key

# Test indexer
POST /indexer/test
X-Api-Key: your_api_key
Content-Type: application/json

{
  "id": 1
}

# Search across indexers
GET /search?query=search_term&categories=2000,5000
X-Api-Key: your_api_key
```

### Overseerr Request Management API
**Base URL**: `http://localhost:5055/api/v1`

```bash
# Get all requests
GET /request
Authorization: Bearer jwt_token

# Create new request
POST /request
Authorization: Bearer jwt_token
Content-Type: application/json

{
  "mediaType": "movie",
  "mediaId": 123456,
  "seasons": [1, 2]
}

# Approve request
POST /request/{request_id}/approve
Authorization: Bearer jwt_token
```

---

## AI/ML Nexus APIs

### Main AI/ML Orchestrator
**Base URL**: `http://localhost:8080/api`

#### System Health
```bash
# Health check
GET /health

# Response
{
  "status": "healthy",
  "services": {
    "recommendation": "running",
    "analysis": "running",
    "compression": "running",
    "emotion": "running"
  },
  "gpu": {
    "available": true,
    "memory": "8GB",
    "utilization": "45%"
  }
}
```

#### Media Processing Pipeline
```bash
# Process media with full AI pipeline
POST /process
Content-Type: application/json

{
  "media_url": "http://example.com/video.mp4",
  "user_id": "user123",
  "analysis_types": ["content", "emotion", "recommendation"],
  "output_formats": ["compressed", "tagged"]
}

# Response
{
  "job_id": "job_abc123",
  "status": "processing",
  "estimated_completion": "2024-01-15T10:30:00Z",
  "results_url": "/api/results/job_abc123"
}
```

#### Real-time AI Interactions
```bash
# WebSocket connection for real-time updates
WS /ws?user_id=user123

# Send interaction
{
  "type": "voice_command",
  "command": "recommend sci-fi movies",
  "context": {
    "current_viewing": "movie_id_123",
    "mood": "relaxed"
  }
}

# Receive response
{
  "type": "recommendation_response",
  "recommendations": [...],
  "confidence": 0.87,
  "processing_time": 45
}
```

### Recommendation Engine API
**Base URL**: `http://localhost:8081/api`

#### Get Recommendations
```bash
# Get personalized recommendations
GET /recommendations/{user_id}
?limit=20&types=movie,tv&min_confidence=0.8

# Response
{
  "user_id": "user123",
  "recommendations": [
    {
      "item_id": "movie_456",
      "title": "Blade Runner 2049",
      "type": "movie",
      "confidence": 0.92,
      "reasons": ["sci-fi genre match", "director preference", "recent viewing patterns"],
      "metadata": {
        "year": 2017,
        "rating": 8.0,
        "genres": ["sci-fi", "thriller"]
      }
    }
  ],
  "generated_at": "2024-01-15T10:15:00Z",
  "model_version": "collaborative-filtering-v2.1"
}
```

#### Train Model
```bash
# Submit training data
POST /train
Content-Type: application/json

{
  "user_interactions": [
    {
      "user_id": "user123",
      "item_id": "movie_456",
      "interaction_type": "watch",
      "rating": 4.5,
      "watch_duration": 7200,
      "timestamp": "2024-01-15T09:00:00Z"
    }
  ],
  "model_config": {
    "algorithm": "deep_collaborative_filtering",
    "epochs": 100,
    "learning_rate": 0.001
  }
}

# Response
{
  "training_job_id": "train_xyz789",
  "status": "queued",
  "estimated_duration": 1800
}
```

#### Submit Feedback
```bash
# Submit user feedback
POST /feedback
Content-Type: application/json

{
  "user_id": "user123",
  "recommendation_id": "rec_abc123",
  "feedback_type": "thumbs_up",
  "explicit_rating": 4.0,
  "implicit_signals": {
    "clicked": true,
    "watched": true,
    "watch_duration": 6800,
    "completed": true
  }
}
```

### Content Analysis API
**Base URL**: `http://localhost:8082/api`

#### Video Analysis
```bash
# Analyze video content
POST /analyze/video
Content-Type: multipart/form-data

file: video_file.mp4
analysis_types: ["objects", "faces", "scenes", "emotions", "nsfw"]
frame_interval: 5  # seconds

# Response
{
  "analysis_id": "analysis_def456",
  "status": "processing",
  "results_url": "/api/results/analysis_def456"
}

# Get analysis results
GET /results/analysis_def456

# Response
{
  "analysis_id": "analysis_def456",
  "video_duration": 7200,
  "results": {
    "objects": [
      {
        "timestamp": 120.5,
        "detections": [
          {
            "class": "person",
            "confidence": 0.95,
            "bbox": [100, 200, 300, 500]
          }
        ]
      }
    ],
    "faces": [
      {
        "timestamp": 120.5,
        "faces": [
          {
            "identity": "actor_john_doe",
            "confidence": 0.88,
            "emotions": {
              "happy": 0.7,
              "surprised": 0.2,
              "neutral": 0.1
            }
          }
        ]
      }
    ],
    "scenes": [
      {
        "start_time": 0,
        "end_time": 300,
        "scene_type": "indoor",
        "lighting": "natural",
        "setting": "living_room"
      }
    ]
  }
}
```

#### Image Analysis
```bash
# Analyze image content
POST /analyze/image
Content-Type: multipart/form-data

file: image.jpg
analysis_types: ["objects", "faces", "text", "nsfw"]

# Response
{
  "image_id": "img_ghi789",
  "dimensions": [1920, 1080],
  "file_size": 2048576,
  "results": {
    "objects": [
      {
        "class": "car",
        "confidence": 0.92,
        "bbox": [300, 400, 800, 700]
      }
    ],
    "faces": [...],
    "text": {
      "detected_text": "STOP SIGN",
      "language": "en",
      "confidence": 0.95
    },
    "nsfw": {
      "is_nsfw": false,
      "confidence": 0.98
    }
  }
}
```

### Neural Compression API
**Base URL**: `http://localhost:8084/api`

#### Video Compression
```bash
# Compress video using neural networks
POST /compress/video
Content-Type: multipart/form-data

file: input_video.mp4
target_quality: high  # low, medium, high, lossless
target_size_mb: 500
preserve_audio: true
hardware_acceleration: true

# Response
{
  "compression_job_id": "comp_jkl012",
  "status": "queued",
  "original_size_mb": 1200,
  "estimated_compressed_size_mb": 480,
  "estimated_processing_time": 600
}

# Check compression status
GET /compress/status/comp_jkl012

# Response
{
  "job_id": "comp_jkl012",
  "status": "processing",
  "progress": 35.2,
  "current_stage": "neural_encoding",
  "estimated_remaining": 390,
  "output_url": null
}

# Get completed compression
GET /compress/download/comp_jkl012
# Returns compressed video file
```

#### Quality Analysis
```bash
# Analyze compression quality
POST /compress/analyze
Content-Type: application/json

{
  "original_url": "http://example.com/original.mp4",
  "compressed_url": "http://example.com/compressed.mp4"
}

# Response
{
  "analysis_id": "qual_mno345",
  "metrics": {
    "psnr": 42.5,
    "ssim": 0.95,
    "vmaf": 88.2,
    "size_reduction": 0.72,
    "bitrate_reduction": 0.68
  },
  "quality_score": 8.7,
  "recommended_settings": {
    "crf": 23,
    "preset": "medium",
    "profile": "high"
  }
}
```

### Emotion Detection API
**Base URL**: `http://localhost:8085/api`

#### Real-time Emotion Analysis
```bash
# Analyze user emotions from video stream
POST /emotion/analyze
Content-Type: multipart/form-data

video_frame: base64_encoded_frame
user_id: user123
context: watching_movie

# Response
{
  "user_id": "user123",
  "timestamp": "2024-01-15T10:20:00Z",
  "emotions": {
    "primary": "joy",
    "confidence": 0.87,
    "distribution": {
      "joy": 0.87,
      "surprise": 0.08,
      "neutral": 0.05
    }
  },
  "engagement_level": 0.92,
  "attention_score": 0.89
}

# WebSocket for real-time emotion streaming
WS /emotion/stream?user_id=user123

# Send frame data
{
  "type": "frame",
  "data": "base64_encoded_frame",
  "timestamp": 1705315200000
}

# Receive emotion data
{
  "type": "emotion_update",
  "emotions": {...},
  "trends": {
    "last_5min": "increasing_engagement",
    "mood_stability": 0.75
  }
}
```

#### Emotion Profile
```bash
# Get user's emotion profile
GET /emotion/profile/{user_id}
?timeframe=week&include_trends=true

# Response
{
  "user_id": "user123",
  "timeframe": "week",
  "profile": {
    "dominant_emotions": ["joy", "excitement", "calm"],
    "viewing_patterns": {
      "preferred_genres_by_mood": {
        "happy": ["comedy", "adventure"],
        "sad": ["drama", "documentary"],
        "stressed": ["nature", "meditation"]
      }
    },
    "engagement_metrics": {
      "average_attention": 0.82,
      "peak_engagement_times": ["19:00-21:00"],
      "content_completion_rate": 0.78
    }
  },
  "trends": {
    "mood_stability": "improving",
    "content_satisfaction": "high",
    "recommendations_effectiveness": 0.89
  }
}
```

---

## AR/VR WebXR APIs

### WebXR Session Management
**Base URL**: `http://localhost:8080/api/xr`

#### Session Control
```javascript
// JavaScript WebXR API integration
// Start XR session
const xrSession = await navigator.xr.requestSession('immersive-vr', {
  requiredFeatures: ['local-floor', 'hand-tracking'],
  optionalFeatures: ['passthrough', 'plane-detection']
});

// Session events
xrSession.addEventListener('selectstart', onSelectStart);
xrSession.addEventListener('selectend', onSelectEnd);
xrSession.addEventListener('visibilitychange', onVisibilityChange);

// Hand tracking data
const frame = session.requestAnimationFrame();
const pose = frame.getViewerPose(referenceSpace);
const hands = frame.getHandPoses();

// Gesture recognition
const gestureRecognizer = new XRGestureRecognizer();
gestureRecognizer.addEventListener('gesture', (event) => {
  console.log('Detected gesture:', event.detail.type);
});
```

#### Spatial Video API
```bash
# Load spatial video content
POST /xr/spatial-video/load
Content-Type: application/json

{
  "video_url": "http://example.com/spatial_video.mp4",
  "format": "side-by-side",  # side-by-side, over-under, mv-hevc
  "fov": 180,  # 180 or 360 degrees
  "stereo": true,
  "environment": "cinema"  # cinema, space, beach, etc.
}

# Response
{
  "session_id": "spatial_pqr678",
  "video_info": {
    "duration": 7200,
    "resolution": "4K",
    "supported_devices": ["quest3", "visionpro", "pico4"]
  },
  "playback_url": "/xr/stream/spatial_pqr678"
}

# Control spatial video playback
POST /xr/spatial-video/control
Content-Type: application/json

{
  "session_id": "spatial_pqr678",
  "action": "play",  # play, pause, seek, volume
  "value": null,     # seek time in seconds, volume 0-1
  "user_position": {
    "x": 0, "y": 1.7, "z": 0,
    "rotation": { "x": 0, "y": 0, "z": 0, "w": 1 }
  }
}
```

#### Mixed Reality Features
```bash
# Enable passthrough mode
POST /xr/mixed-reality/passthrough
Content-Type: application/json

{
  "enabled": true,
  "opacity": 0.8,
  "color_correction": "auto"
}

# Plane detection
GET /xr/mixed-reality/planes

# Response
{
  "detected_planes": [
    {
      "id": "plane_001",
      "type": "horizontal",
      "position": { "x": 0, "y": 0, "z": -2 },
      "orientation": { "x": 0, "y": 0, "z": 0, "w": 1 },
      "extent": { "width": 2.5, "height": 1.8 },
      "confidence": 0.92
    }
  ]
}

# Create anchor
POST /xr/mixed-reality/anchor
Content-Type: application/json

{
  "position": { "x": 0, "y": 1, "z": -1 },
  "orientation": { "x": 0, "y": 0, "z": 0, "w": 1 },
  "persistent": true
}

# Response
{
  "anchor_id": "anchor_stu901",
  "position": { "x": 0, "y": 1, "z": -1 },
  "created_at": "2024-01-15T10:25:00Z"
}
```

---

## Web3 Blockchain APIs

### NFT Content Management
**Base URL**: `http://localhost:3001/api/web3`

#### Mint Content NFT
```bash
# Mint content as NFT
POST /nft/mint
Authorization: Bearer wallet_signature
Content-Type: application/json

{
  "creator_address": "0x742d35Cc6636C0532925a3b8D1666bEF38c4e8B8",
  "content_hash": "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
  "metadata": {
    "title": "Epic Movie Scene",
    "description": "A groundbreaking cinematic moment",
    "duration": 120,
    "resolution": "4K",
    "genre": "sci-fi"
  },
  "royalty_percentage": 10,
  "price_eth": 0.5,
  "blockchain": "ethereum"
}

# Response
{
  "transaction_hash": "0xabc123...",
  "token_id": 12345,
  "contract_address": "0xdef456...",
  "ipfs_url": "ipfs://QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
  "opensea_url": "https://opensea.io/assets/...",
  "estimated_gas": "0.02 ETH",
  "status": "pending"
}
```

#### IPFS Content Storage
```bash
# Upload content to IPFS
POST /ipfs/upload
Content-Type: multipart/form-data

file: content_file.mp4
pin: true
encrypt: false

# Response
{
  "ipfs_hash": "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
  "size": 1048576,
  "pinned": true,
  "gateway_urls": [
    "https://ipfs.io/ipfs/QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
    "https://gateway.pinata.cloud/ipfs/QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
  ]
}

# Retrieve content from IPFS
GET /ipfs/content/{ipfs_hash}
# Returns content file or metadata
```

#### DAO Governance
```bash
# Submit governance proposal
POST /dao/proposal
Authorization: Bearer wallet_signature
Content-Type: application/json

{
  "title": "Add new content category",
  "description": "Proposal to add VR content category to the platform",
  "proposal_type": "content_policy",
  "voting_duration": 604800,  # 1 week in seconds
  "required_votes": 1000,
  "options": ["approve", "reject", "modify"]
}

# Response
{
  "proposal_id": "prop_vwx234",
  "created_at": "2024-01-15T10:30:00Z",
  "voting_ends": "2024-01-22T10:30:00Z",
  "current_votes": 0,
  "required_tokens": "1000 NEXUS"
}

# Vote on proposal
POST /dao/vote
Authorization: Bearer wallet_signature
Content-Type: application/json

{
  "proposal_id": "prop_vwx234",
  "vote": "approve",
  "token_amount": 250
}
```

#### Cryptocurrency Payments
```bash
# Process crypto payment
POST /payments/crypto
Authorization: Bearer wallet_signature
Content-Type: application/json

{
  "content_id": "content_789",
  "payment_type": "purchase",  # purchase, rent, subscription
  "currency": "ETH",
  "amount": "0.1",
  "duration": 2592000,  # 30 days for rental
  "buyer_address": "0x123...",
  "seller_address": "0x456..."
}

# Response
{
  "payment_id": "pay_yzab567",
  "transaction_hash": "0x789def...",
  "status": "confirming",
  "confirmations": 1,
  "required_confirmations": 12,
  "access_token": "temp_access_token",
  "expires_at": "2024-02-14T10:30:00Z"
}

# Check payment status
GET /payments/status/{payment_id}

# Response
{
  "payment_id": "pay_yzab567",
  "status": "confirmed",
  "confirmations": 15,
  "access_granted": true,
  "license": {
    "type": "rental",
    "expires": "2024-02-14T10:30:00Z",
    "permissions": ["view", "download"]
  }
}
```

---

## Voice AI System APIs

### Speech Recognition & NLU
**Base URL**: `http://localhost:8083/api`

#### Voice Command Processing
```bash
# Process voice command
POST /voice/process
Content-Type: multipart/form-data

audio_file: command.wav
user_id: user123
context: jellyfin_browsing
language: en-US

# Response
{
  "command_id": "cmd_cdef890",
  "transcription": "play the latest episode of breaking bad",
  "confidence": 0.94,
  "intent": {
    "action": "play_content",
    "content_type": "tv_episode",
    "series": "Breaking Bad",
    "episode": "latest",
    "confidence": 0.91
  },
  "entities": [
    {
      "type": "series_name",
      "value": "Breaking Bad",
      "confidence": 0.96
    },
    {
      "type": "episode_selector",
      "value": "latest",
      "confidence": 0.89
    }
  ],
  "action_plan": [
    "search_series('Breaking Bad')",
    "get_latest_episode()",
    "start_playback()"
  ]
}

# Execute voice command
POST /voice/execute
Content-Type: application/json

{
  "command_id": "cmd_cdef890",
  "user_confirmation": true,
  "context_override": null
}

# Response
{
  "execution_id": "exec_ghij123",
  "status": "executing",
  "steps": [
    {
      "step": "search_series",
      "status": "completed",
      "result": "series found"
    },
    {
      "step": "get_latest_episode",
      "status": "in_progress",
      "result": null
    }
  ]
}
```

#### WebSocket Voice Streaming
```bash
# Real-time voice processing
WS /voice/stream?user_id=user123&format=pcm

# Send audio chunk
{
  "type": "audio_chunk",
  "data": "base64_encoded_audio",
  "sequence": 1,
  "timestamp": 1705315200000
}

# Receive partial transcription
{
  "type": "partial_transcription",
  "text": "play the latest",
  "confidence": 0.78,
  "is_final": false
}

# Receive final result
{
  "type": "final_result",
  "transcription": "play the latest episode of breaking bad",
  "intent": {...},
  "ready_to_execute": true
}
```

#### Voice Synthesis
```bash
# Generate speech response
POST /voice/synthesize
Content-Type: application/json

{
  "text": "Playing the latest episode of Breaking Bad now.",
  "voice": "female",  # male, female, child, robotic
  "language": "en-US",
  "emotion": "friendly",
  "speed": 1.0,
  "format": "mp3"
}

# Response (audio file)
Content-Type: audio/mpeg
Content-Length: 24576
# Audio data
```

#### Command History
```bash
# Get user's voice command history
GET /voice/history/{user_id}
?limit=50&start_date=2024-01-01&include_failed=false

# Response
{
  "user_id": "user123",
  "total_commands": 234,
  "successful_commands": 198,
  "success_rate": 0.85,
  "history": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "transcription": "pause the movie",
      "intent": "pause_playback",
      "status": "successful",
      "execution_time": 0.8
    }
  ],
  "popular_commands": [
    "play [content]",
    "pause",
    "what's new",
    "recommend something"
  ]
}
```

---

## Monitoring & Analytics APIs

### Grafana API
**Base URL**: `http://localhost:3000/api`

#### Dashboard Management
```bash
# Get all dashboards
GET /search?type=dash-db
Authorization: Bearer grafana_api_key

# Get dashboard by UID
GET /dashboards/uid/{dashboard_uid}
Authorization: Bearer grafana_api_key

# Create dashboard
POST /dashboards/db
Authorization: Bearer grafana_api_key
Content-Type: application/json

{
  "dashboard": {
    "title": "NEXUS Platform Overview",
    "tags": ["nexus", "media"],
    "panels": [...]
  },
  "overwrite": false
}
```

### Prometheus API
**Base URL**: `http://localhost:9090/api/v1`

#### Metrics Query
```bash
# Query current metrics
GET /query?query=up

# Query metrics over time range
GET /query_range?query=cpu_usage&start=2024-01-15T10:00:00Z&end=2024-01-15T11:00:00Z&step=60s

# Get all metric names
GET /label/__name__/values

# Response
{
  "status": "success",
  "data": [
    "jellyfin_active_sessions",
    "ai_ml_processing_queue",
    "gpu_utilization",
    "storage_usage_bytes"
  ]
}
```

### Tautulli API
**Base URL**: `http://localhost:8181/api/v2`

#### Jellyfin Statistics
```bash
# Get current activity
GET /?apikey={api_key}&cmd=get_activity

# Get library statistics
GET /?apikey={api_key}&cmd=get_libraries_table

# Get user statistics
GET /?apikey={api_key}&cmd=get_users_table

# Response
{
  "response": {
    "result": "success",
    "data": {
      "recordsTotal": 5,
      "data": [
        {
          "user_id": 1,
          "username": "admin",
          "plays": 245,
          "duration": 892340
        }
      ]
    }
  }
}
```

---

## Authentication & Security

### JWT Authentication
```bash
# Generate JWT token
POST /auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "secure_password",
  "device_id": "desktop_browser"
}

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "def456...",
  "expires_in": 3600,
  "token_type": "Bearer",
  "user": {
    "id": "user123",
    "username": "admin",
    "roles": ["admin", "user"]
  }
}

# Refresh token
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "def456..."
}
```

### API Key Management
```bash
# Generate API key
POST /auth/api-keys
Authorization: Bearer jwt_token
Content-Type: application/json

{
  "name": "Mobile App Access",
  "permissions": ["read", "write"],
  "expires_in": 2592000  # 30 days
}

# Response
{
  "api_key": "nexus_key_abc123...",
  "key_id": "key_789",
  "expires_at": "2024-02-14T10:30:00Z",
  "permissions": ["read", "write"]
}

# Use API key
GET /api/user/profile
X-API-Key: nexus_key_abc123...
```

### Quantum-Resistant Security
```bash
# Get security status
GET /security/quantum/status
Authorization: Bearer jwt_token

# Response
{
  "quantum_resistance": true,
  "algorithms": {
    "key_exchange": "ML-KEM-768",
    "digital_signature": "ML-DSA-65",
    "hash_signature": "SLH-DSA-SHAKE-128s"
  },
  "hybrid_mode": true,
  "performance_impact": {
    "latency_ms": 0.23,
    "bandwidth_overhead": "2.1KB",
    "cpu_overhead": "3%"
  }
}

# Enable/disable quantum security
POST /security/quantum/configure
Authorization: Bearer admin_jwt_token
Content-Type: application/json

{
  "enabled": true,
  "hybrid_mode": true,
  "algorithms": {
    "key_exchange": "ML-KEM-1024",
    "signature": "ML-DSA-87"
  }
}
```

---

## Error Handling

### Standard Error Response Format
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request parameters are invalid",
    "details": "Missing required field: user_id",
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_abc123",
    "documentation_url": "https://docs.nexus-platform.com/errors/INVALID_REQUEST"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Request parameters are invalid |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Too many requests |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |
| `AI_MODEL_ERROR` | 500 | AI/ML processing error |
| `QUANTUM_SECURITY_ERROR` | 500 | Quantum security operation failed |
| `BLOCKCHAIN_ERROR` | 502 | Web3/blockchain operation failed |
| `XR_DEVICE_ERROR` | 503 | AR/VR device communication error |

---

## Rate Limiting

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1705315200
X-RateLimit-Window: 3600
```

### Rate Limits by Service

| Service | Endpoint | Limit | Window |
|---------|----------|-------|--------|
| **Core Media** | `/api/*` | 1000/hour | 1 hour |
| **AI/ML** | `/api/process` | 100/hour | 1 hour |
| **AI/ML** | `/api/recommendations` | 500/hour | 1 hour |
| **AR/VR** | `/api/xr/*` | 200/hour | 1 hour |
| **Web3** | `/api/web3/*` | 50/hour | 1 hour |
| **Voice** | `/api/voice/process` | 300/hour | 1 hour |
| **Authentication** | `/auth/login` | 10/minute | 1 minute |

### Rate Limit Bypass
```bash
# Premium API key (higher limits)
GET /api/user/profile
X-API-Key: nexus_premium_key_xyz789...
X-Rate-Limit-Tier: premium
```

---

## SDK & Client Libraries

### JavaScript/TypeScript SDK
```javascript
import { NexusClient } from '@nexus-platform/sdk';

const client = new NexusClient({
  baseUrl: 'http://localhost:8080',
  apiKey: 'your_api_key',
  quantumSecurity: true
});

// AI/ML operations
const recommendations = await client.ai.getRecommendations('user123');
const analysis = await client.ai.analyzeContent('video.mp4');

// AR/VR operations
const xrSession = await client.xr.startSession('immersive-vr');
const spatialVideo = await client.xr.loadSpatialVideo('video.mp4');

// Web3 operations
const nft = await client.web3.mintNFT({
  content: 'ipfs://...',
  creator: '0x...'
});

// Voice operations
const command = await client.voice.processCommand(audioBuffer);
```

### Python SDK
```python
from nexus_platform import NexusClient

client = NexusClient(
    base_url='http://localhost:8080',
    api_key='your_api_key',
    quantum_security=True
)

# AI/ML operations
recommendations = client.ai.get_recommendations('user123')
analysis = client.ai.analyze_content('video.mp4')

# Batch processing
results = client.ai.batch_process([
    {'type': 'analysis', 'content': 'video1.mp4'},
    {'type': 'compression', 'content': 'video2.mp4'}
])
```

### cURL Examples
```bash
# Set common variables
export NEXUS_BASE_URL="http://localhost:8080"
export NEXUS_API_KEY="your_api_key"

# Get AI recommendations
curl -H "X-API-Key: $NEXUS_API_KEY" \
     "$NEXUS_BASE_URL/api/ai/recommendations/user123"

# Process voice command
curl -X POST \
     -H "X-API-Key: $NEXUS_API_KEY" \
     -F "audio_file=@command.wav" \
     -F "user_id=user123" \
     "$NEXUS_BASE_URL/api/voice/process"

# Start XR session
curl -X POST \
     -H "X-API-Key: $NEXUS_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"session_type": "immersive-vr", "features": ["hand-tracking"]}' \
     "$NEXUS_BASE_URL/api/xr/session/start"
```

---

## Webhooks & Events

### Webhook Configuration
```bash
# Register webhook endpoint
POST /webhooks/register
Authorization: Bearer jwt_token
Content-Type: application/json

{
  "url": "https://your-app.com/nexus-webhook",
  "events": [
    "media.added",
    "ai.analysis.completed",
    "user.recommendation.generated",
    "xr.session.started",
    "web3.nft.minted"
  ],
  "secret": "webhook_secret_key"
}
```

### Event Types
```json
{
  "event": "ai.analysis.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "analysis_id": "analysis_abc123",
    "content_id": "movie_456",
    "user_id": "user123",
    "results": {
      "objects_detected": 15,
      "faces_recognized": 3,
      "confidence": 0.92
    }
  },
  "signature": "sha256=..."
}
```

---

This comprehensive API reference covers all major endpoints and functionality across the NEXUS Media Server Platform. For additional details, examples, or support, please refer to the individual service documentation or contact the development team.

**Last Updated**: January 2025  
**API Version**: v2.1  
**Platform Version**: NEXUS 2025.1