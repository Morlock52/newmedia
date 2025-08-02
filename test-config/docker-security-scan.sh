#!/bin/bash

# Ultimate Media Server 2025 - Docker Security Scanning
# Comprehensive security analysis using multiple tools

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../test-results/security"
LOG_FILE="${RESULTS_DIR}/security-scan-$(date +%Y%m%d_%H%M%S).log"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Logging function
log() {
    local level="$1"
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

# Container images to scan
IMAGES=(
    "jellyfin/jellyfin:latest"
    "plexinc/pms-docker:latest"
    "emby/embyserver:latest"
    "lscr.io/linuxserver/sonarr:latest"
    "lscr.io/linuxserver/radarr:latest"
    "lscr.io/linuxserver/lidarr:latest"
    "lscr.io/linuxserver/readarr:develop"
    "lscr.io/linuxserver/bazarr:latest"
    "lscr.io/linuxserver/prowlarr:latest"
    "fallenbagel/jellyseerr:latest"
    "sctx/overseerr:latest"
    "lscr.io/linuxserver/ombi:latest"
    "lscr.io/linuxserver/qbittorrent:latest"
    "lscr.io/linuxserver/transmission:latest"
    "lscr.io/linuxserver/sabnzbd:latest"
    "lscr.io/linuxserver/nzbget:latest"
    "qmcgaw/gluetun:latest"
    "prom/prometheus:latest"
    "grafana/grafana:latest"
    "grafana/loki:latest"
    "grafana/promtail:latest"
    "louislam/uptime-kuma:latest"
    "ghcr.io/analogj/scrutiny:master-omnibus"
    "nicolargo/glances:latest-full"
    "netdata/netdata:latest"
    "portainer/portainer-ce:latest"
    "postgres:16-alpine"
    "mariadb:11"
    "redis:7-alpine"
    "ghcr.io/gethomepage/homepage:latest"
    "ghcr.io/ajnart/homarr:latest"
)

# Trivy vulnerability scanning
run_trivy_scan() {
    log "INFO" "Starting Trivy vulnerability scans"
    
    local trivy_summary="${RESULTS_DIR}/trivy-summary.json"
    echo '{"scans": []}' > "$trivy_summary"
    
    for image in "${IMAGES[@]}"; do
        local safe_name=$(echo "$image" | sed 's/[^a-zA-Z0-9]/_/g')
        local output_file="${RESULTS_DIR}/trivy-${safe_name}.json"
        
        log "INFO" "Scanning image: $image"
        
        # Run Trivy scan
        if docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
           -v "$RESULTS_DIR":/reports \
           aquasec/trivy:latest image \
           --format json \
           --output "/reports/trivy-${safe_name}.json" \
           --severity HIGH,CRITICAL \
           --ignore-unfixed \
           "$image" 2>>"$LOG_FILE"; then
            
            log "INFO" "✓ Completed scan for $image"
            
            # Extract summary
            if [[ -f "$output_file" ]]; then
                local vulns=$(jq '.Results[0].Vulnerabilities // [] | length' "$output_file" 2>/dev/null || echo "0")
                local critical=$(jq '.Results[0].Vulnerabilities // [] | map(select(.Severity == "CRITICAL")) | length' "$output_file" 2>/dev/null || echo "0")
                local high=$(jq '.Results[0].Vulnerabilities // [] | map(select(.Severity == "HIGH")) | length' "$output_file" 2>/dev/null || echo "0")
                
                # Add to summary
                local scan_result=$(jq -n \
                    --arg image "$image" \
                    --argjson total "$vulns" \
                    --argjson critical "$critical" \
                    --argjson high "$high" \
                    --arg timestamp "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
                    '{
                        image: $image,
                        vulnerabilities: {
                            total: $total,
                            critical: $critical,
                            high: $high
                        },
                        timestamp: $timestamp,
                        status: "completed"
                    }')
                
                jq --argjson scan "$scan_result" '.scans += [$scan]' "$trivy_summary" > "${trivy_summary}.tmp" && mv "${trivy_summary}.tmp" "$trivy_summary"
                
                if [[ $critical -gt 0 ]]; then
                    log "WARN" "⚠️  $image has $critical CRITICAL vulnerabilities"
                elif [[ $high -gt 0 ]]; then
                    log "WARN" "⚠️  $image has $high HIGH vulnerabilities"
                else
                    log "INFO" "✅ $image has no HIGH/CRITICAL vulnerabilities"
                fi
            fi
        else
            log "ERROR" "✗ Failed to scan $image"
            
            # Add failed scan to summary
            local scan_result=$(jq -n \
                --arg image "$image" \
                --arg timestamp "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
                '{
                    image: $image,
                    timestamp: $timestamp,
                    status: "failed",
                    error: "Scan failed"
                }')
            
            jq --argjson scan "$scan_result" '.scans += [$scan]' "$trivy_summary" > "${trivy_summary}.tmp" && mv "${trivy_summary}.tmp" "$trivy_summary"
        fi
    done
    
    log "INFO" "Trivy vulnerability scans completed"
}

# Container security configuration audit
run_container_security_audit() {
    log "INFO" "Running container security configuration audit"
    
    local audit_report="${RESULTS_DIR}/container-security-audit.json"
    echo '{"containers": []}' > "$audit_report"
    
    # Get running containers
    local containers=$(docker ps --format "{{.Names}}" | grep -E "(jellyfin|sonarr|radarr|lidarr|prowlarr|grafana|prometheus)" || true)
    
    if [[ -z "$containers" ]]; then
        log "WARN" "No target containers running"
        return
    fi
    
    while IFS= read -r container; do
        if [[ -z "$container" ]]; then continue; fi
        
        log "INFO" "Auditing container: $container"
        
        # Get container configuration
        local config=$(docker inspect "$container" 2>/dev/null || echo '[]')
        
        if [[ "$config" == "[]" ]]; then
            log "WARN" "Could not inspect container: $container"
            continue
        fi
        
        # Extract security-relevant configuration
        local privileged=$(echo "$config" | jq -r '.[0].HostConfig.Privileged // false')
        local user=$(echo "$config" | jq -r '.[0].Config.User // "root"')
        local readonly_rootfs=$(echo "$config" | jq -r '.[0].HostConfig.ReadonlyRootfs // false')
        local cap_add=$(echo "$config" | jq -r '.[0].HostConfig.CapAdd // []')
        local cap_drop=$(echo "$config" | jq -r '.[0].HostConfig.CapDrop // []')
        local security_opt=$(echo "$config" | jq -r '.[0].HostConfig.SecurityOpt // []')
        local network_mode=$(echo "$config" | jq -r '.[0].HostConfig.NetworkMode // "default"')
        
        # Security assessment
        local issues=()
        local warnings=()
        local good_practices=()
        
        if [[ "$privileged" == "true" ]]; then
            issues+=("Running in privileged mode")
        else
            good_practices+=("Not running in privileged mode")
        fi
        
        if [[ "$user" == "root" || "$user" == "" ]]; then
            warnings+=("Running as root user")
        else
            good_practices+=("Running as non-root user: $user")
        fi
        
        if [[ "$readonly_rootfs" == "true" ]]; then
            good_practices+=("Read-only root filesystem")
        else
            warnings+=("Writable root filesystem")
        fi
        
        if [[ "$cap_add" != "null" && "$cap_add" != "[]" ]]; then
            warnings+=("Additional capabilities added: $cap_add")
        fi
        
        if [[ "$cap_drop" != "null" && "$cap_drop" != "[]" ]]; then
            good_practices+=("Capabilities dropped: $cap_drop")
        fi
        
        # Create container audit result
        local container_audit=$(jq -n \
            --arg name "$container" \
            --arg privileged "$privileged" \
            --arg user "$user" \
            --arg readonly_rootfs "$readonly_rootfs" \
            --argjson cap_add "$cap_add" \
            --argjson cap_drop "$cap_drop" \
            --argjson security_opt "$security_opt" \
            --arg network_mode "$network_mode" \
            --argjson issues "$(printf '%s\n' "${issues[@]}" | jq -R . | jq -s .)" \
            --argjson warnings "$(printf '%s\n' "${warnings[@]}" | jq -R . | jq -s .)" \
            --argjson good_practices "$(printf '%s\n' "${good_practices[@]}" | jq -R . | jq -s .)" \
            --arg timestamp "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
            '{
                name: $name,
                configuration: {
                    privileged: ($privileged | test("true")),
                    user: $user,
                    readonly_rootfs: ($readonly_rootfs | test("true")),
                    cap_add: $cap_add,
                    cap_drop: $cap_drop,
                    security_opt: $security_opt,
                    network_mode: $network_mode
                },
                assessment: {
                    issues: $issues,
                    warnings: $warnings,
                    good_practices: $good_practices
                },
                timestamp: $timestamp
            }')
        
        jq --argjson container "$container_audit" '.containers += [$container]' "$audit_report" > "${audit_report}.tmp" && mv "${audit_report}.tmp" "$audit_report"
        
        # Log findings
        if [[ ${#issues[@]} -gt 0 ]]; then
            log "ERROR" "🚨 $container has security issues: ${issues[*]}"
        elif [[ ${#warnings[@]} -gt 0 ]]; then
            log "WARN" "⚠️  $container has security warnings: ${warnings[*]}"
        else
            log "INFO" "✅ $container security configuration looks good"
        fi
        
    done <<< "$containers"
    
    log "INFO" "Container security audit completed"
}

# Network security analysis
run_network_security_analysis() {
    log "INFO" "Running network security analysis"
    
    local network_report="${RESULTS_DIR}/network-security-analysis.json"
    
    # Get Docker networks
    local networks=$(docker network ls --format "{{.Name}}" | grep -v "bridge\|host\|none" || true)
    
    local network_analysis='{"networks": [], "exposed_ports": [], "analysis_timestamp": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}'
    
    # Analyze each custom network
    while IFS= read -r network; do
        if [[ -z "$network" ]]; then continue; fi
        
        log "INFO" "Analyzing network: $network"
        
        # Get network details
        local network_info=$(docker network inspect "$network" 2>/dev/null || echo '[]')
        
        if [[ "$network_info" != "[]" ]]; then
            local subnet=$(echo "$network_info" | jq -r '.[0].IPAM.Config[0].Subnet // "unknown"')
            local driver=$(echo "$network_info" | jq -r '.[0].Driver // "unknown"')
            local internal=$(echo "$network_info" | jq -r '.[0].Internal // false')
            local containers=$(echo "$network_info" | jq -r '.[0].Containers // {} | keys | length')
            
            local network_data=$(jq -n \
                --arg name "$network" \
                --arg subnet "$subnet" \
                --arg driver "$driver" \
                --arg internal "$internal" \
                --argjson containers "$containers" \
                '{
                    name: $name,
                    subnet: $subnet,
                    driver: $driver,
                    internal: ($internal | test("true")),
                    container_count: $containers
                }')
            
            network_analysis=$(echo "$network_analysis" | jq --argjson net "$network_data" '.networks += [$net]')
        fi
    done <<< "$networks"
    
    # Analyze exposed ports
    local exposed_ports=$(docker ps --format "table {{.Names}}\t{{.Ports}}" | tail -n +2 | grep -v "^$" || true)
    
    while IFS=$'\t' read -r container_name ports; do
        if [[ -z "$container_name" || "$ports" == "" ]]; then continue; fi
        
        # Parse port mappings
        while IFS=',' read -ra PORT_ARRAY; do
            for port_mapping in "${PORT_ARRAY[@]}"; do
                if [[ "$port_mapping" =~ ([0-9]+):([0-9]+) ]]; then
                    local host_port="${BASH_REMATCH[1]}"
                    local container_port="${BASH_REMATCH[2]}"
                    
                    local port_data=$(jq -n \
                        --arg container "$container_name" \
                        --arg host_port "$host_port" \
                        --arg container_port "$container_port" \
                        --arg mapping "$port_mapping" \
                        '{
                            container: $container,
                            host_port: $host_port,
                            container_port: $container_port,
                            mapping: $mapping
                        }')
                    
                    network_analysis=$(echo "$network_analysis" | jq --argjson port "$port_data" '.exposed_ports += [$port]')
                fi
            done
        done <<< "$ports"
    done <<< "$exposed_ports"
    
    echo "$network_analysis" > "$network_report"
    
    log "INFO" "Network security analysis completed"
}

# Generate security report summary
generate_security_summary() {
    log "INFO" "Generating security summary report"
    
    local summary_report="${RESULTS_DIR}/security-summary.json"
    
    # Initialize summary
    local summary='{
        "timestamp": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ')"',
        "scan_type": "comprehensive_security_audit",
        "vulnerability_summary": {},
        "container_security_summary": {},
        "network_security_summary": {},
        "recommendations": []
    }'
    
    # Trivy vulnerability summary
    if [[ -f "${RESULTS_DIR}/trivy-summary.json" ]]; then
        local vuln_summary=$(jq '.scans | {
            total_images: length,
            completed_scans: map(select(.status == "completed")) | length,
            failed_scans: map(select(.status == "failed")) | length,
            total_vulnerabilities: map(.vulnerabilities.total // 0) | add // 0,
            critical_vulnerabilities: map(.vulnerabilities.critical // 0) | add // 0,
            high_vulnerabilities: map(.vulnerabilities.high // 0) | add // 0,
            images_with_critical: map(select(.vulnerabilities.critical > 0)) | length,
            images_with_high: map(select(.vulnerabilities.high > 0)) | length
        }' "${RESULTS_DIR}/trivy-summary.json")
        
        summary=$(echo "$summary" | jq --argjson vuln "$vuln_summary" '.vulnerability_summary = $vuln')
    fi
    
    # Container security summary
    if [[ -f "${RESULTS_DIR}/container-security-audit.json" ]]; then
        local container_summary=$(jq '.containers | {
            total_containers: length,
            containers_with_issues: map(select(.assessment.issues | length > 0)) | length,
            containers_with_warnings: map(select(.assessment.warnings | length > 0)) | length,
            privileged_containers: map(select(.configuration.privileged == true)) | length,
            root_containers: map(select(.configuration.user == "root" or .configuration.user == "")) | length,
            readonly_containers: map(select(.configuration.readonly_rootfs == true)) | length
        }' "${RESULTS_DIR}/container-security-audit.json")
        
        summary=$(echo "$summary" | jq --argjson container "$container_summary" '.container_security_summary = $container')
    fi
    
    # Network security summary
    if [[ -f "${RESULTS_DIR}/network-security-analysis.json" ]]; then
        local network_summary=$(jq '{
            total_networks: .networks | length,
            internal_networks: .networks | map(select(.internal == true)) | length,
            exposed_ports: .exposed_ports | length,
            unique_host_ports: .exposed_ports | map(.host_port) | unique | length
        }' "${RESULTS_DIR}/network-security-analysis.json")
        
        summary=$(echo "$summary" | jq --argjson network "$network_summary" '.network_security_summary = $network')
    fi
    
    # Generate recommendations
    local recommendations=()
    
    # Vulnerability recommendations
    if [[ -f "${RESULTS_DIR}/trivy-summary.json" ]]; then
        local critical_count=$(jq '.scans | map(.vulnerabilities.critical // 0) | add // 0' "${RESULTS_DIR}/trivy-summary.json")
        local high_count=$(jq '.scans | map(.vulnerabilities.high // 0) | add // 0' "${RESULTS_DIR}/trivy-summary.json")
        
        if [[ $critical_count -gt 0 ]]; then
            recommendations+=("CRITICAL: $critical_count critical vulnerabilities found. Update affected images immediately.")
        fi
        
        if [[ $high_count -gt 0 ]]; then
            recommendations+=("HIGH: $high_count high-severity vulnerabilities found. Plan updates for affected images.")
        fi
    fi
    
    # Container security recommendations
    if [[ -f "${RESULTS_DIR}/container-security-audit.json" ]]; then
        local privileged_count=$(jq '.containers | map(select(.configuration.privileged == true)) | length' "${RESULTS_DIR}/container-security-audit.json")
        local root_count=$(jq '.containers | map(select(.configuration.user == "root" or .configuration.user == "")) | length' "${RESULTS_DIR}/container-security-audit.json")
        
        if [[ $privileged_count -gt 0 ]]; then
            recommendations+=("SECURITY: $privileged_count containers running in privileged mode. Review necessity.")
        fi
        
        if [[ $root_count -gt 0 ]]; then
            recommendations+=("SECURITY: $root_count containers running as root. Configure non-root users where possible.")
        fi
    fi
    
    # Add recommendations to summary
    for rec in "${recommendations[@]}"; do
        summary=$(echo "$summary" | jq --arg rec "$rec" '.recommendations += [$rec]')
    done
    
    echo "$summary" > "$summary_report"
    
    log "INFO" "Security summary report generated: $summary_report"
}

# Main execution
main() {
    echo -e "${BLUE}🔒 Ultimate Media Server 2025 - Security Scan${NC}"
    echo "=============================================="
    
    log "INFO" "Starting comprehensive security scan"
    
    # Run security scans
    run_trivy_scan
    run_container_security_audit
    run_network_security_analysis
    generate_security_summary
    
    # Display summary
    echo -e "\n${BLUE}🔍 Security Scan Summary${NC}"
    echo "=========================="
    
    if [[ -f "${RESULTS_DIR}/security-summary.json" ]]; then
        local summary_file="${RESULTS_DIR}/security-summary.json"
        
        echo -e "📊 Vulnerability Summary:"
        local vuln_summary=$(jq -r '.vulnerability_summary | 
            "  • Total Images: \(.total_images // 0)
  • Critical Vulnerabilities: \(.critical_vulnerabilities // 0)
  • High Vulnerabilities: \(.high_vulnerabilities // 0)
  • Images with Critical: \(.images_with_critical // 0)"' "$summary_file" 2>/dev/null || echo "  • No vulnerability data available")
        echo "$vuln_summary"
        
        echo -e "\n🔐 Container Security Summary:"
        local container_summary=$(jq -r '.container_security_summary | 
            "  • Total Containers: \(.total_containers // 0)
  • Privileged Containers: \(.privileged_containers // 0)
  • Root Containers: \(.root_containers // 0)
  • Read-only Containers: \(.readonly_containers // 0)"' "$summary_file" 2>/dev/null || echo "  • No container security data available")
        echo "$container_summary"
        
        echo -e "\n🌐 Network Security Summary:"
        local network_summary=$(jq -r '.network_security_summary | 
            "  • Total Networks: \(.total_networks // 0)
  • Internal Networks: \(.internal_networks // 0)
  • Exposed Ports: \(.exposed_ports // 0)
  • Unique Host Ports: \(.unique_host_ports // 0)"' "$summary_file" 2>/dev/null || echo "  • No network security data available")
        echo "$network_summary"
        
        echo -e "\n💡 Recommendations:"
        local recommendations=$(jq -r '.recommendations[]? // "No specific recommendations"' "$summary_file" 2>/dev/null)
        if [[ -n "$recommendations" ]]; then
            while IFS= read -r rec; do
                echo "  • $rec"
            done <<< "$recommendations"
        else
            echo "  • No specific recommendations"
        fi
    fi
    
    echo -e "\n📁 Detailed reports available in: ${RESULTS_DIR}/"
    echo -e "📄 Summary report: ${RESULTS_DIR}/security-summary.json"
    echo -e "📋 Full log: $LOG_FILE"
    
    log "INFO" "Security scan completed"
}

# Check dependencies
if ! command -v docker >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker is required but not installed.${NC}"
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    echo -e "${RED}Error: jq is required but not installed.${NC}"
    exit 1
fi

# Execute main function
main "$@"