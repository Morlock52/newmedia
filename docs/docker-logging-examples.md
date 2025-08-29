Docker Compose Logging Examples
===============================

Below are example snippets showing how to mount log volumes and enable JSON + daily rotation for services.

1) API service example

services:
  api:
    image: newmedia-api:latest
    ports:
      - "3002:3002"
    volumes:
      - ./logs/api:/var/log/service
      - ./config/api:/config
    environment:
      <<: *common-variables
      LOG_FORMAT: ${LOG_FORMAT:-json}
      LOG_FILE: /var/log/service/app.log
      LOG_DAILY_ROTATE: ${LOG_DAILY_ROTATE:-true}
    <<: *default-logging

2) Voice AI service example

  voice:
    image: newmedia-voice:latest
    ports:
      - "8080:8080"
    volumes:
      - ./logs/voice:/var/log/service
      - ./voice-media:/media
    environment:
      LOG_FORMAT: json
      LOG_FILE: /var/log/service/voice.log
      LOG_DAILY_ROTATE: true
    <<: *default-logging

Notes:
- Ensure the host directories (./logs/api, ./logs/voice) exist and are writable by the container user.
- The logger will write JSON lines to STDOUT and also to the file transport if LOG_FILE is set.
- LOG_DAILY_ROTATE=true enables daily rotation (requires winston-daily-rotate-file installed in image).
