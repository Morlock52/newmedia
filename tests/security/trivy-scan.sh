#!/bin/bash

# Container vulnerability scanning with Trivy
# This script scans all Docker images for security vulnerabilities

set -euo pipefail

# Configuration
REPORT_DIR="${1:-./reports/security}"
SEVERITY="${2:-HIGH,CRITICAL}"
FORMAT="${3:-json}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure report directory exists
mkdir -p "${REPORT_DIR}"

# Function to log with timestamp
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to scan a single image
scan_image() {
    local image="$1"
    local image_name=$(echo "$image" | tr '/' '_' | tr ':' '_')
    local report_file="${REPORT_DIR}/trivy-${image_name}.${FORMAT}"
    
    log "${YELLOW}Scanning image: ${image}${NC}"
    
    if docker image inspect "$image" >/dev/null 2>&1; then
        trivy image \
            --severity "$SEVERITY" \
            --format "$FORMAT" \
            --output "$report_file" \
            --quiet \
            "$image"
        
        if [ $? -eq 0 ]; then
            log "${GREEN}✓ Scan completed for ${image}${NC}"
            
            # Count vulnerabilities if JSON format
            if [ "$FORMAT" = "json" ]; then
                vuln_count=$(jq '[.Results[]?.Vulnerabilities[]?] | length' "$report_file" 2>/dev/null || echo "0")
                if [ "$vuln_count" -gt 0 ]; then
                    log "${RED}⚠ Found ${vuln_count} vulnerabilities in ${image}${NC}"
                else
                    log "${GREEN}✓ No vulnerabilities found in ${image}${NC}"
                fi
            fi
        else
            log "${RED}✗ Scan failed for ${image}${NC}"
            return 1
        fi
    else
        log "${YELLOW}⚠ Image ${image} not found locally, pulling...${NC}"
        if docker pull "$image" >/dev/null 2>&1; then
            scan_image "$image"
        else
            log "${RED}✗ Failed to pull ${image}${NC}"
            return 1
        fi
    fi
}

# List of images to scan (based on docker-compose.yml)
IMAGES=(
    "jellyfin/jellyfin:latest"
    "lscr.io/linuxserver/prowlarr:latest"
    "lscr.io/linuxserver/sonarr:latest"
    "lscr.io/linuxserver/radarr:latest"
    "lscr.io/linuxserver/lidarr:latest"
    "lscr.io/linuxserver/bazarr:latest"
    "lscr.io/linuxserver/qbittorrent:latest"
    "lscr.io/linuxserver/sabnzbd:latest"
    "lscr.io/linuxserver/overseerr:latest"
    "lscr.io/linuxserver/tautulli:latest"
    "qmcgaw/gluetun:latest"
    "prom/prometheus:latest"
    "grafana/grafana:latest"
    "ghcr.io/gethomepage/homepage:latest"
    "portainer/portainer-ce:latest"
    "traefik:v3.0"
    "postgres:15-alpine"
    "redis:7-alpine"
)

# Check if Trivy is available
if ! command -v trivy &> /dev/null; then
    log "${RED}✗ Trivy is not installed. Please install it first.${NC}"
    log "Installation: https://aquasecurity.github.io/trivy/latest/getting-started/installation/"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    log "${RED}✗ Docker is not available${NC}"
    exit 1
fi

# Update Trivy database
log "${YELLOW}Updating Trivy vulnerability database...${NC}"
trivy image --download-db-only

# Start scanning
log "${GREEN}Starting security scan with Trivy${NC}"
log "Severity levels: ${SEVERITY}"
log "Output format: ${FORMAT}"
log "Report directory: ${REPORT_DIR}"

total_images=${#IMAGES[@]}
scanned_count=0
failed_count=0

for image in "${IMAGES[@]}"; do
    if scan_image "$image"; then
        ((scanned_count++))
    else
        ((failed_count++))
    fi
done

# Generate summary report
summary_file="${REPORT_DIR}/trivy-summary.json"
log "${YELLOW}Generating summary report...${NC}"

cat > "$summary_file" << EOF
{
  "scan_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "total_images": $total_images,
  "scanned_successfully": $scanned_count,
  "failed_scans": $failed_count,
  "severity_filter": "$SEVERITY",
  "total_vulnerabilities": 0,
  "high_severity": 0,
  "critical_severity": 0,
  "images": []
}
EOF

# Aggregate results if JSON format
if [ "$FORMAT" = "json" ]; then
    total_vulns=0
    high_vulns=0
    critical_vulns=0
    
    for report in "${REPORT_DIR}"/trivy-*.json; do
        if [ -f "$report" ] && [ "$report" != "$summary_file" ]; then
            image_vulns=$(jq '[.Results[]?.Vulnerabilities[]?] | length' "$report" 2>/dev/null || echo "0")
            high_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity=="HIGH")] | length' "$report" 2>/dev/null || echo "0")
            critical_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity=="CRITICAL")] | length' "$report" 2>/dev/null || echo "0")
            
            total_vulns=$((total_vulns + image_vulns))
            high_vulns=$((high_vulns + high_count))
            critical_vulns=$((critical_vulns + critical_count))
        fi
    done
    
    # Update summary with totals
    tmp_file=$(mktemp)
    jq ".total_vulnerabilities = $total_vulns | .high_severity = $high_vulns | .critical_severity = $critical_vulns" "$summary_file" > "$tmp_file"
    mv "$tmp_file" "$summary_file"
fi

# Generate HTML report if requested
if [ "${GENERATE_HTML:-false}" = "true" ]; then
    html_report="${REPORT_DIR}/trivy-report.html"
    log "${YELLOW}Generating HTML report...${NC}"
    
    cat > "$html_report" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Trivy Security Scan Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .critical { color: #d32f2f; font-weight: bold; }
        .high { color: #ff9800; font-weight: bold; }
        .medium { color: #fbc02d; }
        .low { color: #4caf50; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Trivy Security Scan Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Scan Date:</strong> $(date)</p>
        <p><strong>Total Images Scanned:</strong> $scanned_count</p>
        <p><strong>Failed Scans:</strong> $failed_count</p>
        <p><strong>Total Vulnerabilities:</strong> $total_vulns</p>
        <p><strong>Critical:</strong> <span class="critical">$critical_vulns</span></p>
        <p><strong>High:</strong> <span class="high">$high_vulns</span></p>
    </div>
    
    <h2>Detailed Results</h2>
    <p>See individual JSON reports for detailed vulnerability information.</p>
</body>
</html>
EOF
fi

# Final summary
log "${GREEN}=== SCAN SUMMARY ===${NC}"
log "Total images: $total_images"
log "Successfully scanned: $scanned_count"
log "Failed scans: $failed_count"

if [ "$FORMAT" = "json" ]; then
    log "Total vulnerabilities found: $total_vulns"
    log "Critical: $critical_vulns"
    log "High: $high_vulns"
fi

log "Reports saved to: $REPORT_DIR"

# Exit with error if critical vulnerabilities found
if [ "$critical_vulns" -gt 0 ]; then
    log "${RED}⚠ CRITICAL vulnerabilities found! Review immediately.${NC}"
    exit 1
elif [ "$failed_count" -gt 0 ]; then
    log "${YELLOW}⚠ Some scans failed. Check the logs.${NC}"
    exit 1
else
    log "${GREEN}✓ Security scan completed successfully${NC}"
    exit 0
fi