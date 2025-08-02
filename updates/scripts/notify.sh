#!/bin/bash

# Custom notification script for Diun
# Called when Docker image updates are detected

IMAGE="$1"
STATUS="$2"
TAG="$3"

# Log the notification
echo "[$(date)] Docker update notification: $IMAGE $STATUS $TAG" >> /var/log/diun-notifications.log

# Send to monitoring system (example with curl)
if [ -n "$MONITORING_WEBHOOK" ]; then
    curl -X POST "$MONITORING_WEBHOOK" \
        -H "Content-Type: application/json" \
        -d "{
            \"text\": \"Docker Update: $IMAGE ($STATUS) - $TAG\",
            \"image\": \"$IMAGE\",
            \"status\": \"$STATUS\",
            \"tag\": \"$TAG\",
            \"timestamp\": \"$(date -Iseconds)\"
        }"
fi

# Custom actions based on image
case "$IMAGE" in
    *traefik*)
        echo "Critical service update detected: $IMAGE"
        # Could trigger additional checks or alerts
        ;;
    *jellyfin*|*sonarr*|*radarr*)
        echo "Media service update detected: $IMAGE"
        # Could trigger media-specific notifications
        ;;
    *)
        echo "General service update detected: $IMAGE"
        ;;
esac