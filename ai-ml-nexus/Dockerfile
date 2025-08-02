# Multi-stage build for optimized AI/ML image
FROM node:24-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install TensorFlow C library for Node.js bindings
RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.15.0.tar.gz \
    && tar -xzf libtensorflow-gpu-linux-x86_64-2.15.0.tar.gz -C /usr/local \
    && ldconfig \
    && rm libtensorflow-gpu-linux-x86_64-2.15.0.tar.gz

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci --only=production

# Copy application code
COPY . .

# Download model files
RUN mkdir -p models/face models/whisper models/encoder models/decoder

# Download face-api models
RUN cd models/face && \
    wget https://github.com/justadudewhohacks/face-api.js-models/raw/master/ssd_mobilenetv1_model-weights_manifest.json && \
    wget https://github.com/justadudewhohacks/face-api.js-models/raw/master/ssd_mobilenetv1_model.weights && \
    wget https://github.com/justadudewhohacks/face-api.js-models/raw/master/face_landmark_68_model-weights_manifest.json && \
    wget https://github.com/justadudewhohacks/face-api.js-models/raw/master/face_landmark_68_model.weights && \
    wget https://github.com/justadudewhohacks/face-api.js-models/raw/master/face_recognition_model-weights_manifest.json && \
    wget https://github.com/justadudewhohacks/face-api.js-models/raw/master/face_recognition_model.weights && \
    wget https://github.com/justadudewhohacks/face-api.js-models/raw/master/face_expression_model-weights_manifest.json && \
    wget https://github.com/justadudewhohacks/face-api.js-models/raw/master/face_expression_model.weights && \
    wget https://github.com/justadudewhohacks/face-api.js-models/raw/master/age_gender_model-weights_manifest.json && \
    wget https://github.com/justadudewhohacks/face-api.js-models/raw/master/age_gender_model.weights

# Create necessary directories
RUN mkdir -p /tmp/frames /tmp/audio cache

# Set environment variables
ENV NODE_ENV=production
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Expose ports
EXPOSE 8080 8081 8082 8083 8084 8085

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the orchestrator by default
CMD ["node", "services/index.js"]