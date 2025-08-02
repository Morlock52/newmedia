#!/bin/bash

# Container permission and security audit
# Checks for security misconfigurations and permission issues

set -euo pipefail

# Configuration
REPORT_DIR="${1:-./reports/security}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure report directory exists
mkdir -p "${REPORT_DIR}"

# Function to log with timestamp
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check container security configuration
check_container_security() {
    local container="$1"
    local report_file="${REPORT_DIR}/container-${container}-security.json"
    
    log "${BLUE}Auditing container: ${container}${NC}"
    
    if ! docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
        log "${YELLOW}⚠ Container ${container} not running, skipping${NC}"
        return 0
    fi
    
    local security_issues=()
    
    # Get container inspection data
    local inspect_data=$(docker inspect "$container")
    
    # Check if running as root
    local user=$(echo "$inspect_data" | jq -r '.[].Config.User // "root"')
    if [ "$user" = "root" ] || [ "$user" = "0" ] || [ -z "$user" ]; then
        security_issues+=("running_as_root")
        log "${RED}✗ ${container} is running as root${NC}"
    else
        log "${GREEN}✓ ${container} is running as user: ${user}${NC}"
    fi
    
    # Check privileged mode
    local privileged=$(echo "$inspect_data" | jq -r '.[].HostConfig.Privileged')
    if [ "$privileged" = "true" ]; then
        security_issues+=("privileged_mode")
        log "${RED}✗ ${container} is running in privileged mode${NC}"
    else
        log "${GREEN}✓ ${container} is not running in privileged mode${NC}"
    fi
    
    # Check capabilities
    local cap_add=$(echo "$inspect_data" | jq -r '.[].HostConfig.CapAdd // [] | join(",")')
    local cap_drop=$(echo "$inspect_data" | jq -r '.[].HostConfig.CapDrop // [] | join(",")')
    
    if [ -n "$cap_add" ] && [ "$cap_add" != "null" ]; then
        log "${YELLOW}⚠ ${container} has added capabilities: ${cap_add}${NC}"
        # Check for dangerous capabilities
        if echo "$cap_add" | grep -qE "(SYS_ADMIN|DAC_OVERRIDE|SYS_PTRACE)"; then
            security_issues+=("dangerous_capabilities")
            log "${RED}✗ ${container} has dangerous capabilities${NC}"
        fi
    fi
    
    if [ -n "$cap_drop" ] && [ "$cap_drop" != "null" ]; then
        log "${GREEN}✓ ${container} has dropped capabilities: ${cap_drop}${NC}"
    fi
    
    # Check read-only root filesystem
    local readonly_rootfs=$(echo "$inspect_data" | jq -r '.[].HostConfig.ReadonlyRootfs')
    if [ "$readonly_rootfs" = "true" ]; then
        log "${GREEN}✓ ${container} has read-only root filesystem${NC}"
    else
        log "${YELLOW}⚠ ${container} does not have read-only root filesystem${NC}"
    fi
    
    # Check no-new-privileges
    local no_new_privs=$(echo "$inspect_data" | jq -r '.[].HostConfig.SecurityOpt // [] | map(select(startswith("no-new-privileges"))) | length > 0')
    if [ "$no_new_privs" = "true" ]; then
        log "${GREEN}✓ ${container} has no-new-privileges enabled${NC}"
    else
        log "${YELLOW}⚠ ${container} does not have no-new-privileges enabled${NC}"
    fi
    
    # Check AppArmor/SELinux
    local security_opt=$(echo "$inspect_data" | jq -r '.[].HostConfig.SecurityOpt // [] | join(",")')
    if echo "$security_opt" | grep -q "apparmor\|selinux"; then
        log "${GREEN}✓ ${container} has security profiles enabled${NC}"
    else
        log "${YELLOW}⚠ ${container} may not have security profiles enabled${NC}"
    fi
    
    # Check device access
    local devices=$(echo "$inspect_data" | jq -r '.[].HostConfig.Devices // [] | length')
    if [ "$devices" -gt 0 ]; then
        local device_list=$(echo "$inspect_data" | jq -r '.[].HostConfig.Devices[] | .PathOnHost')
        log "${YELLOW}⚠ ${container} has device access: ${device_list}${NC}"
        # Check for risky devices
        if echo "$device_list" | grep -qE "(/dev/mem|/dev/kmem|/dev/kcore)"; then
            security_issues+=("dangerous_device_access")
            log "${RED}✗ ${container} has access to dangerous devices${NC}"
        fi
    fi
    
    # Check volume mounts
    local mounts=$(echo "$inspect_data" | jq -r '.[].Mounts[]')
    if [ -n "$mounts" ]; then
        echo "$inspect_data" | jq -r '.[].Mounts[]' | while IFS= read -r mount; do
            local source=$(echo "$mount" | jq -r '.Source')
            local dest=$(echo "$mount" | jq -r '.Destination')
            local mode=$(echo "$mount" | jq -r '.Mode // "rw"')
            
            # Check for sensitive host paths
            if echo "$source" | grep -qE "(^/$|^/etc|^/proc|^/sys|^/dev|^/boot)"; then
                log "${RED}✗ ${container} mounts sensitive host path: ${source} -> ${dest} (${mode})${NC}"
                security_issues+=("sensitive_mount")
            elif [ "$mode" != "ro" ] && echo "$source" | grep -q "/var/run/docker.sock"; then
                log "${RED}✗ ${container} has read-write Docker socket access${NC}"
                security_issues+=("docker_socket_rw")
            else
                log "${GREEN}✓ ${container} mount: ${source} -> ${dest} (${mode})${NC}"
            fi
        done
    fi
    
    # Generate container security report
    cat > "$report_file" << EOF
{
  "container": "$container",
  "audit_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "user": "$user",
  "privileged": $privileged,
  "readonly_rootfs": $readonly_rootfs,
  "capabilities_added": "$cap_add",
  "capabilities_dropped": "$cap_drop",
  "security_opt": "$security_opt",
  "devices": $devices,
  "security_issues": $(printf '%s\n' "${security_issues[@]}" | jq -R . | jq -s .),
  "security_score": $(( 100 - ${#security_issues[@]} * 10 ))
}
EOF
    
    return ${#security_issues[@]}
}

# Function to check file permissions in volumes
check_volume_permissions() {
    local container="$1"
    
    if ! docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
        return 0
    fi
    
    log "${BLUE}Checking volume permissions for: ${container}${NC}"
    
    # Check common directories
    local dirs=("/config" "/downloads" "/media" "/data")
    
    for dir in "${dirs[@]}"; do
        if docker exec "$container" test -d "$dir" 2>/dev/null; then
            local perms=$(docker exec "$container" stat -c "%a %U:%G" "$dir" 2>/dev/null || echo "unknown")
            local owner_writable=$(echo "$perms" | cut -d' ' -f1 | cut -c1)
            
            if [ "$owner_writable" -ge 6 ]; then
                log "${GREEN}✓ ${container}:${dir} permissions: ${perms}${NC}"
            else
                log "${RED}✗ ${container}:${dir} may not be writable: ${perms}${NC}"
            fi
        fi
    done
}

# Function to check environment variables for secrets
check_environment_secrets() {
    local container="$1"
    
    if ! docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
        return 0
    fi
    
    log "${BLUE}Checking environment variables for: ${container}${NC}"
    
    local env_vars=$(docker inspect "$container" | jq -r '.[].Config.Env[]')
    local secret_patterns=("PASSWORD" "SECRET" "KEY" "TOKEN" "PASS")
    local secrets_found=0
    
    for pattern in "${secret_patterns[@]}"; do
        if echo "$env_vars" | grep -i "$pattern" >/dev/null; then
            log "${RED}✗ ${container} has potential secret in environment: $(echo "$env_vars" | grep -i "$pattern" | cut -d'=' -f1)${NC}"
            ((secrets_found++))
        fi
    done
    
    if [ $secrets_found -eq 0 ]; then
        log "${GREEN}✓ ${container} has no obvious secrets in environment${NC}"
    fi
    
    return $secrets_found
}

# Function to check network exposure
check_network_exposure() {
    local container="$1"
    
    if ! docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
        return 0
    fi
    
    log "${BLUE}Checking network exposure for: ${container}${NC}"
    
    local ports=$(docker port "$container" 2>/dev/null || echo "")
    local exposed_count=0
    
    if [ -n "$ports" ]; then
        while IFS= read -r port_mapping; do
            if echo "$port_mapping" | grep -q "0.0.0.0"; then
                log "${YELLOW}⚠ ${container} exposed on all interfaces: ${port_mapping}${NC}"
                ((exposed_count++))
            else
                log "${GREEN}✓ ${container} port: ${port_mapping}${NC}"
            fi
        done <<< "$ports"
    else
        log "${GREEN}✓ ${container} has no exposed ports${NC}"
    fi
    
    return $exposed_count
}

# Function to generate summary report
generate_summary_report() {
    local total_containers="$1"
    local total_issues="$2"
    
    log "${BLUE}Generating security audit summary${NC}"
    
    cat > "${REPORT_DIR}/security-audit-summary.json" << EOF
{
  "audit_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "total_containers": $total_containers,
  "total_security_issues": $total_issues,
  "risk_level": "$([ $total_issues -eq 0 ] && echo "LOW" || [ $total_issues -lt 5 ] && echo "MEDIUM" || echo "HIGH")",
  "recommendations": [
    "Run containers as non-root users",
    "Use read-only root filesystems where possible",
    "Drop unnecessary capabilities",
    "Enable no-new-privileges",
    "Avoid privileged mode",
    "Use secrets management instead of environment variables",
    "Limit device access to necessary devices only"
  ]
}
EOF
}

# Main execution
log "${GREEN}=== Starting Container Security Audit ===${NC}"

# Get list of running containers
containers=$(docker ps --format "{{.Names}}")
total_containers=0
total_issues=0

if [ -z "$containers" ]; then
    log "${YELLOW}No running containers found${NC}"
    exit 0
fi

# Audit each container
for container in $containers; do
    ((total_containers++))
    
    # Check container security configuration
    check_container_security "$container"
    issues=$?
    total_issues=$((total_issues + issues))
    
    # Check volume permissions
    check_volume_permissions "$container"
    
    # Check environment variables
    check_environment_secrets "$container"
    secrets=$?
    total_issues=$((total_issues + secrets))
    
    # Check network exposure
    check_network_exposure "$container"
    exposure=$?
    total_issues=$((total_issues + exposure))
    
    echo # Blank line between containers
done

# Generate summary report
generate_summary_report "$total_containers" "$total_issues"

# Generate human-readable summary
cat > "${REPORT_DIR}/security-audit-summary.txt" << EOF
Container Security Audit Summary
================================

Audit Date: $(date)
Total Containers Audited: $total_containers
Total Security Issues Found: $total_issues

Risk Level: $([ $total_issues -eq 0 ] && echo "LOW" || [ $total_issues -lt 5 ] && echo "MEDIUM" || echo "HIGH")

Recommendations:
- Review individual container reports in ${REPORT_DIR}
- Address any containers running as root
- Consider using read-only root filesystems
- Implement proper secrets management
- Review and minimize exposed ports
- Regular security audits

Detailed reports available in: ${REPORT_DIR}
EOF

# Final summary
log "${GREEN}=== Security Audit Summary ===${NC}"
log "Total containers audited: $total_containers"
log "Total security issues: $total_issues"

if [ $total_issues -eq 0 ]; then
    log "${GREEN}✓ No security issues found${NC}"
    exit 0
elif [ $total_issues -lt 5 ]; then
    log "${YELLOW}⚠ Minor security issues found${NC}"
    exit 0
else
    log "${RED}✗ Multiple security issues found - review immediately${NC}"
    exit 1
fi