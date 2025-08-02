#!/bin/bash
# Security scanning automation script for media server containers

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECURITY_DIR="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$SECURITY_DIR/reports"
SBOM_DIR="$SECURITY_DIR/sbom"
COMPOSE_FILE="${COMPOSE_FILE:-$SECURITY_DIR/docker-compose-secure.yml}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create necessary directories
mkdir -p "$REPORTS_DIR" "$SBOM_DIR"

# Function to run Trivy scan
run_trivy_scan() {
    local image=$1
    local report_file="$REPORTS_DIR/trivy-$(echo "$image" | tr '/:' '_')-$(date +%Y%m%d-%H%M%S).json"
    
    log_info "Scanning $image with Trivy..."
    
    if docker run --rm \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v "$REPORTS_DIR:/reports" \
        aquasec/trivy:latest image \
        --format json \
        --output "/reports/$(basename "$report_file")" \
        --severity CRITICAL,HIGH,MEDIUM \
        --no-progress \
        "$image"; then
        log_success "Trivy scan completed for $image"
        
        # Check for critical vulnerabilities
        local critical_count=$(jq '[.Results[].Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "$report_file" 2>/dev/null || echo "0")
        if [[ "$critical_count" -gt 0 ]]; then
            log_error "Found $critical_count CRITICAL vulnerabilities in $image"
            return 1
        fi
    else
        log_error "Trivy scan failed for $image"
        return 1
    fi
}

# Function to generate SBOM
generate_sbom() {
    local image=$1
    local sbom_base="$SBOM_DIR/$(echo "$image" | tr '/:' '_')-$(date +%Y%m%d-%H%M%S)"
    
    log_info "Generating SBOM for $image..."
    
    if docker run --rm \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v "$SBOM_DIR:/sbom" \
        anchore/syft:latest \
        packages "$image" \
        -o json="/sbom/$(basename "$sbom_base")-sbom.json" \
        -o spdx="/sbom/$(basename "$sbom_base")-sbom.spdx" \
        -o cyclonedx="/sbom/$(basename "$sbom_base")-sbom.xml" \
        --quiet; then
        log_success "SBOM generated for $image"
    else
        log_error "SBOM generation failed for $image"
        return 1
    fi
}

# Function to verify image signatures
verify_image_signature() {
    local image=$1
    
    log_info "Verifying signature for $image..."
    
    # Check if Docker Content Trust is enabled
    if [[ "${DOCKER_CONTENT_TRUST:-0}" == "1" ]]; then
        if docker trust inspect "$image" >/dev/null 2>&1; then
            log_success "Image signature verified for $image"
        else
            log_warning "No valid signature found for $image"
            return 1
        fi
    else
        log_warning "Docker Content Trust is not enabled"
    fi
    
    # Check for Cosign signature if public key exists
    local image_key="$SECURITY_DIR/cosign/$(echo "$image" | tr '/:' '_').pub"
    if [[ -f "$image_key" ]]; then
        if docker run --rm \
            -v "$SECURITY_DIR/cosign:/keys:ro" \
            gcr.io/projectsigstore/cosign:latest \
            verify --key "/keys/$(basename "$image_key")" \
            "$image" >/dev/null 2>&1; then
            log_success "Cosign signature verified for $image"
        else
            log_error "Cosign signature verification failed for $image"
            return 1
        fi
    fi
}

# Function to check container runtime security
check_runtime_security() {
    local container=$1
    
    log_info "Checking runtime security for $container..."
    
    # Check if container is running as non-root
    local user=$(docker inspect "$container" --format '{{.Config.User}}' 2>/dev/null || echo "")
    if [[ -z "$user" || "$user" == "root" || "$user" == "0" ]]; then
        log_warning "Container $container is running as root"
    else
        log_success "Container $container is running as non-root user: $user"
    fi
    
    # Check read-only root filesystem
    local readonly=$(docker inspect "$container" --format '{{.HostConfig.ReadonlyRootfs}}' 2>/dev/null || echo "false")
    if [[ "$readonly" == "true" ]]; then
        log_success "Container $container has read-only root filesystem"
    else
        log_warning "Container $container has writable root filesystem"
    fi
    
    # Check capabilities
    local cap_add=$(docker inspect "$container" --format '{{join .HostConfig.CapAdd " "}}' 2>/dev/null || echo "")
    local cap_drop=$(docker inspect "$container" --format '{{join .HostConfig.CapDrop " "}}' 2>/dev/null || echo "")
    
    if [[ "$cap_drop" == *"ALL"* ]]; then
        log_success "Container $container drops all capabilities"
    else
        log_warning "Container $container does not drop all capabilities"
    fi
    
    if [[ -n "$cap_add" ]]; then
        log_info "Container $container adds capabilities: $cap_add"
    fi
    
    # Check security options
    local security_opts=$(docker inspect "$container" --format '{{join .HostConfig.SecurityOpt " "}}' 2>/dev/null || echo "")
    if [[ "$security_opts" == *"no-new-privileges"* ]]; then
        log_success "Container $container has no-new-privileges set"
    else
        log_warning "Container $container does not have no-new-privileges set"
    fi
}

# Function to run CIS Docker Benchmark
run_cis_benchmark() {
    log_info "Running CIS Docker Benchmark..."
    
    local report_file="$REPORTS_DIR/docker-bench-security-$(date +%Y%m%d-%H%M%S).log"
    
    if docker run --rm --net host --pid host --userns host --cap-add audit_control \
        -e DOCKER_CONTENT_TRUST="$DOCKER_CONTENT_TRUST" \
        -v /var/lib:/var/lib:ro \
        -v /var/run/docker.sock:/var/run/docker.sock:ro \
        -v /etc:/etc:ro \
        -v "$REPORTS_DIR:/reports" \
        docker/docker-bench-security \
        -l "/reports/$(basename "$report_file")"; then
        log_success "CIS Docker Benchmark completed"
    else
        log_error "CIS Docker Benchmark failed"
        return 1
    fi
}

# Function to generate security report
generate_security_report() {
    local report_file="$REPORTS_DIR/security-report-$(date +%Y%m%d-%H%M%S).html"
    
    log_info "Generating security report..."
    
    cat > "$report_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Media Server Security Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2, h3 { color: #333; }
        .success { color: green; }
        .warning { color: orange; }
        .error { color: red; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .summary { background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Media Server Security Report</h1>
    <p>Generated: <script>document.write(new Date().toLocaleString());</script></p>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p>Security scan results for media server infrastructure</p>
    </div>
    
    <h2>Vulnerability Scan Results</h2>
    <table id="vulnTable">
        <thead>
            <tr>
                <th>Image</th>
                <th>Critical</th>
                <th>High</th>
                <th>Medium</th>
                <th>Low</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
        </tbody>
    </table>
    
    <h2>Runtime Security Check</h2>
    <table id="runtimeTable">
        <thead>
            <tr>
                <th>Container</th>
                <th>Non-Root User</th>
                <th>Read-Only FS</th>
                <th>Capabilities Dropped</th>
                <th>No New Privileges</th>
            </tr>
        </thead>
        <tbody>
        </tbody>
    </table>
    
    <h2>SBOM Status</h2>
    <ul id="sbomList">
    </ul>
    
    <script>
        // Add vulnerability data
        // Add runtime security data
        // Add SBOM data
    </script>
</body>
</html>
EOF
    
    log_success "Security report generated: $report_file"
}

# Main execution
main() {
    log_info "Starting security scan for media server containers..."
    
    # Get list of images from docker-compose
    local images=$(docker-compose -f "$COMPOSE_FILE" config | grep "image:" | awk '{print $2}' | sort -u)
    
    local scan_errors=0
    
    # Scan each image
    for image in $images; do
        echo ""
        log_info "Processing $image..."
        
        # Pull latest image
        if docker pull "$image" >/dev/null 2>&1; then
            log_success "Image pulled: $image"
        else
            log_warning "Failed to pull $image, using local version"
        fi
        
        # Run security checks
        if ! run_trivy_scan "$image"; then
            ((scan_errors++))
        fi
        
        if ! generate_sbom "$image"; then
            ((scan_errors++))
        fi
        
        if ! verify_image_signature "$image"; then
            ((scan_errors++))
        fi
    done
    
    echo ""
    log_info "Checking running containers..."
    
    # Check runtime security for running containers
    local containers=$(docker ps --format "{{.Names}}" | grep -E "(jellyfin|sonarr|radarr|prowlarr|bazarr|qbittorrent|overseerr|traefik|authelia)" || true)
    
    for container in $containers; do
        check_runtime_security "$container"
    done
    
    echo ""
    # Run CIS benchmark
    run_cis_benchmark
    
    # Generate consolidated report
    generate_security_report
    
    echo ""
    if [[ "$scan_errors" -eq 0 ]]; then
        log_success "Security scan completed successfully!"
    else
        log_error "Security scan completed with $scan_errors errors"
        exit 1
    fi
}

# Run main function
main "$@"