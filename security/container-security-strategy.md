# Container Security Strategy for Media Server 2025

## Executive Summary

This comprehensive security strategy provides defense-in-depth protection for the media server infrastructure through multiple layers of container security controls, following industry best practices and compliance standards.

## 1. Non-Root User Configuration

### Base Security User
All containers must run with non-root users to minimize privilege escalation risks.

```yaml
# Standard non-root configuration
environment:
  - PUID=1001  # Non-root user ID
  - PGID=1001  # Non-root group ID
  - UMASK=077  # Restrictive file permissions
```

### User Namespace Remapping
Enable user namespace remapping in Docker daemon:

```json
{
  "userns-remap": "media-server",
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 65536,
      "Soft": 65536
    }
  }
}
```

## 2. Read-Only Root Filesystems

### Container Configuration
```yaml
# Example: Jellyfin with read-only root
jellyfin:
  image: jellyfin/jellyfin:latest
  read_only: true
  tmpfs:
    - /tmp:size=100M,mode=1770,uid=1001,gid=1001
    - /var/log/jellyfin:size=50M,mode=1770,uid=1001,gid=1001
    - /cache:size=500M,mode=1770,uid=1001,gid=1001
  volumes:
    - jellyfin_config:/config:rw
    - ${MEDIA_PATH}:/media:ro
```

### Writable Paths Strategy
- Use tmpfs for temporary files
- Mount specific volumes for persistent data
- Keep application binaries read-only

## 3. Security Scanning Integration

### Trivy Scanner Configuration
```yaml
# security-scanner service
trivy-scanner:
  image: aquasec/trivy:latest
  profiles: [security]
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock:ro
    - ./security/trivy-cache:/root/.cache/trivy
    - ./security/reports:/reports
  command: >
    image
    --format json
    --output /reports/scan-results.json
    --severity CRITICAL,HIGH,MEDIUM
    --exit-code 1
    ${IMAGE_TO_SCAN}
```

### Automated Scanning Pipeline
```bash
#!/bin/bash
# scan-images.sh
for image in $(docker images --format "{{.Repository}}:{{.Tag}}" | grep -v "<none>"); do
  echo "Scanning $image..."
  docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(pwd)/security/reports:/reports \
    aquasec/trivy:latest image \
    --format json \
    --output "/reports/${image//\//_}.json" \
    "$image"
done
```

## 4. Secret Management

### Docker Secrets Configuration
```yaml
secrets:
  # Database Credentials
  db_root_password:
    external: true
  db_user_password:
    external: true
  
  # API Keys
  jellyfin_api_key:
    external: true
  sonarr_api_key:
    external: true
  radarr_api_key:
    external: true
  
  # VPN Credentials
  vpn_private_key:
    external: true
  
  # SSL Certificates
  ssl_cert:
    file: ./secrets/ssl/cert.pem
  ssl_key:
    file: ./secrets/ssl/key.pem
```

### Service Secret Usage
```yaml
jellyfin:
  image: jellyfin/jellyfin:latest
  secrets:
    - jellyfin_api_key
  environment:
    - API_KEY_FILE=/run/secrets/jellyfin_api_key
```

### Secret Rotation Script
```bash
#!/bin/bash
# rotate-secrets.sh
# Automated secret rotation with zero downtime

# Generate new secret
NEW_SECRET=$(openssl rand -base64 32)

# Create new Docker secret
echo "$NEW_SECRET" | docker secret create new_jellyfin_api_key -

# Update service to use new secret
docker service update \
  --secret-rm jellyfin_api_key \
  --secret-add source=new_jellyfin_api_key,target=jellyfin_api_key \
  jellyfin

# Remove old secret after verification
docker secret rm old_jellyfin_api_key
```

## 5. Network Segmentation and Firewalling

### Network Architecture
```yaml
networks:
  # Public-facing services (reverse proxy only)
  dmz:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.name: br-dmz
    ipam:
      config:
        - subnet: 172.30.0.0/24
          gateway: 172.30.0.1
    labels:
      - "security.zone=dmz"
      - "security.firewall=strict"
  
  # Frontend services (web interfaces)
  frontend:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.30.1.0/24
    labels:
      - "security.zone=frontend"
      - "security.firewall=moderate"
  
  # Backend services (databases, cache)
  backend:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.30.2.0/24
    labels:
      - "security.zone=backend"
      - "security.firewall=strict"
  
  # Download services (isolated)
  downloads:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.30.3.0/24
    labels:
      - "security.zone=downloads"
      - "security.firewall=paranoid"
```

### iptables Rules
```bash
#!/bin/bash
# network-firewall.sh

# DMZ rules - only allow HTTP/HTTPS
iptables -A DOCKER-USER -i br-dmz -p tcp --dport 80 -j ACCEPT
iptables -A DOCKER-USER -i br-dmz -p tcp --dport 443 -j ACCEPT
iptables -A DOCKER-USER -i br-dmz -j DROP

# Frontend - allow specific ports
iptables -A DOCKER-USER -i br-frontend -p tcp --dport 8096 -j ACCEPT  # Jellyfin
iptables -A DOCKER-USER -i br-frontend -p tcp --dport 32400 -j ACCEPT # Plex
iptables -A DOCKER-USER -i br-frontend -j DROP

# Backend - restrict to internal only
iptables -A DOCKER-USER -i br-backend -s 172.30.0.0/16 -j ACCEPT
iptables -A DOCKER-USER -i br-backend -j DROP

# Downloads - VPN only egress
iptables -A DOCKER-USER -i br-downloads -o tun0 -j ACCEPT
iptables -A DOCKER-USER -i br-downloads -j DROP
```

## 6. AppArmor/SELinux Profiles

### AppArmor Profile for Media Services
```
# /etc/apparmor.d/docker-media-server
#include <tunables/global>

profile docker-media-server flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>
  
  # Network access
  network inet tcp,
  network inet udp,
  network inet6 tcp,
  network inet6 udp,
  
  # Deny network raw
  deny network raw,
  
  # File access
  /config/** rw,
  /media/** r,
  /transcode/** rw,
  
  # Deny system access
  deny /proc/sys/** w,
  deny /sys/** w,
  deny /dev/** w,
  
  # Allow specific devices
  /dev/dri/* r,  # GPU access
  /dev/null rw,
  /dev/urandom r,
  
  # Capabilities
  capability dac_override,
  capability setuid,
  capability setgid,
  
  # Deny dangerous capabilities
  deny capability sys_admin,
  deny capability sys_module,
  deny capability sys_rawio,
}
```

### SELinux Policy
```
# media-server.te
policy_module(media_server, 1.0.0)

require {
    type container_t;
    type container_file_t;
    class file { read write create unlink };
    class dir { read write add_name remove_name };
}

# Define media server domain
type media_server_t;
type media_server_exec_t;

# File contexts
type media_config_t;
type media_content_t;
type media_transcode_t;

# Allow read-only media access
allow media_server_t media_content_t:file { read };
allow media_server_t media_content_t:dir { read };

# Allow config read/write
allow media_server_t media_config_t:file { read write create unlink };
allow media_server_t media_config_t:dir { read write add_name remove_name };

# Allow transcode read/write
allow media_server_t media_transcode_t:file { read write create unlink };
allow media_server_t media_transcode_t:dir { read write add_name remove_name };
```

### Container Security Options
```yaml
jellyfin:
  security_opt:
    - apparmor=docker-media-server
    - seccomp=./security/seccomp-media.json
    - no-new-privileges:true
  cap_drop:
    - ALL
  cap_add:
    - CHOWN
    - DAC_OVERRIDE
    - SETUID
    - SETGID
```

## 7. Image Signing and Verification

### Docker Content Trust (DCT)
```bash
# Enable DCT globally
export DOCKER_CONTENT_TRUST=1
export DOCKER_CONTENT_TRUST_SERVER=https://notary.docker.io

# Initialize repository keys
docker trust key generate media-server-key
docker trust signer add --key media-server-key.pub media-signer registry.example.com/media-server
```

### Cosign Integration
```yaml
# Image verification service
cosign-verifier:
  image: gcr.io/projectsigstore/cosign:latest
  profiles: [security]
  volumes:
    - ./security/cosign:/keys:ro
  environment:
    - COSIGN_EXPERIMENTAL=1
  command: >
    verify
    --key /keys/cosign.pub
    ${IMAGE_TO_VERIFY}
```

### Automated Verification Script
```bash
#!/bin/bash
# verify-images.sh

# List of trusted image patterns
TRUSTED_REGISTRIES=(
  "docker.io/jellyfin/*"
  "lscr.io/linuxserver/*"
  "ghcr.io/*"
)

# Verification function
verify_image() {
  local image=$1
  
  # Check Docker Content Trust
  if [[ "$DOCKER_CONTENT_TRUST" == "1" ]]; then
    docker trust inspect "$image" || return 1
  fi
  
  # Verify with Cosign if public key exists
  if [[ -f "./security/cosign/${image}.pub" ]]; then
    cosign verify --key "./security/cosign/${image}.pub" "$image" || return 1
  fi
  
  return 0
}

# Main verification loop
for image in $(docker-compose config | grep "image:" | awk '{print $2}'); do
  echo "Verifying $image..."
  if ! verify_image "$image"; then
    echo "ERROR: Failed to verify $image"
    exit 1
  fi
done
```

## 8. SBOM Generation

### Syft Integration
```yaml
# SBOM generator service
sbom-generator:
  image: anchore/syft:latest
  profiles: [security]
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock:ro
    - ./security/sbom:/sbom
  command: >
    packages
    -o json=/sbom/${IMAGE_NAME}-sbom.json
    -o spdx=/sbom/${IMAGE_NAME}-sbom.spdx
    -o cyclonedx=/sbom/${IMAGE_NAME}-sbom.xml
    ${IMAGE_TO_SCAN}
```

### Automated SBOM Pipeline
```bash
#!/bin/bash
# generate-sboms.sh

SBOM_DIR="./security/sbom"
mkdir -p "$SBOM_DIR"

# Generate SBOMs for all images
for image in $(docker images --format "{{.Repository}}:{{.Tag}}" | grep -v "<none>"); do
  echo "Generating SBOM for $image..."
  
  # Clean image name for filename
  image_file=$(echo "$image" | tr '/:' '_')
  
  # Generate SBOM in multiple formats
  docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$SBOM_DIR:/sbom" \
    anchore/syft:latest \
    packages "$image" \
    -o json="/sbom/${image_file}-sbom.json" \
    -o spdx="/sbom/${image_file}-sbom.spdx" \
    -o cyclonedx="/sbom/${image_file}-sbom.xml"
    
  # Generate vulnerability report
  docker run --rm \
    -v "$SBOM_DIR:/sbom" \
    anchore/grype:latest \
    sbom:/sbom/${image_file}-sbom.json \
    -o json > "$SBOM_DIR/${image_file}-vulns.json"
done

# Generate consolidated report
./security/scripts/consolidate-sboms.py "$SBOM_DIR" > "$SBOM_DIR/consolidated-report.html"
```

### SBOM Analysis Script
```python
#!/usr/bin/env python3
# consolidate-sboms.py

import json
import sys
from pathlib import Path
from datetime import datetime

def analyze_sboms(sbom_dir):
    """Analyze all SBOMs and generate security report"""
    sbom_files = Path(sbom_dir).glob("*-sbom.json")
    vuln_files = Path(sbom_dir).glob("*-vulns.json")
    
    report = {
        "generated": datetime.now().isoformat(),
        "images": {},
        "summary": {
            "total_images": 0,
            "total_packages": 0,
            "total_vulnerabilities": 0,
            "critical_vulns": 0,
            "high_vulns": 0
        }
    }
    
    # Process each SBOM
    for sbom_file in sbom_files:
        with open(sbom_file) as f:
            sbom_data = json.load(f)
            
        image_name = sbom_file.stem.replace("-sbom", "")
        package_count = len(sbom_data.get("artifacts", []))
        
        # Find corresponding vulnerability file
        vuln_file = sbom_file.parent / f"{image_name}-vulns.json"
        vuln_data = {}
        if vuln_file.exists():
            with open(vuln_file) as f:
                vuln_data = json.load(f)
        
        # Count vulnerabilities by severity
        vulns_by_severity = {}
        for match in vuln_data.get("matches", []):
            severity = match["vulnerability"]["severity"]
            vulns_by_severity[severity] = vulns_by_severity.get(severity, 0) + 1
        
        report["images"][image_name] = {
            "packages": package_count,
            "vulnerabilities": vulns_by_severity
        }
        
        # Update summary
        report["summary"]["total_images"] += 1
        report["summary"]["total_packages"] += package_count
        report["summary"]["critical_vulns"] += vulns_by_severity.get("Critical", 0)
        report["summary"]["high_vulns"] += vulns_by_severity.get("High", 0)
    
    return report

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: consolidate-sboms.py <sbom_directory>")
        sys.exit(1)
    
    report = analyze_sboms(sys.argv[1])
    print(json.dumps(report, indent=2))
```

## Implementation Checklist

- [ ] Configure Docker daemon for user namespace remapping
- [ ] Apply non-root user configurations to all services
- [ ] Implement read-only root filesystems where possible
- [ ] Set up automated security scanning with Trivy
- [ ] Deploy Docker secrets for sensitive data
- [ ] Configure network segmentation and firewall rules
- [ ] Create and apply AppArmor/SELinux profiles
- [ ] Enable image signing and verification
- [ ] Automate SBOM generation and analysis
- [ ] Set up security monitoring and alerting
- [ ] Document security procedures and incident response
- [ ] Schedule regular security audits and updates

## Security Monitoring

### Security Event Logging
```yaml
# Falco runtime security
falco:
  image: falcosecurity/falco:latest
  privileged: true
  profiles: [security]
  volumes:
    - /var/run/docker.sock:/host/var/run/docker.sock:ro
    - /dev:/host/dev:ro
    - /proc:/host/proc:ro
    - /boot:/host/boot:ro
    - /lib/modules:/host/lib/modules:ro
    - ./security/falco/rules:/etc/falco/rules.d:ro
  command: >
    /usr/bin/falco
    -A
    -K /host/proc/1/root/usr/src
    -k https://0.0.0.0:8765/k8s-audit
```

### Security Metrics
```yaml
# Prometheus security metrics
- job_name: 'security'
  static_configs:
    - targets:
      - 'trivy-exporter:9520'
      - 'falco-exporter:9376'
      - 'docker-exporter:9323'
  metric_relabel_configs:
    - source_labels: [__name__]
      regex: 'security_.*'
      target_label: __tmp_security
      replacement: 'true'
```

## Incident Response

### Automated Response Script
```bash
#!/bin/bash
# incident-response.sh

respond_to_incident() {
  local container=$1
  local incident_type=$2
  
  case $incident_type in
    "privilege_escalation")
      docker stop "$container"
      docker commit "$container" "quarantine/${container}:$(date +%s)"
      notify_security_team "Privilege escalation detected in $container"
      ;;
    "suspicious_network")
      docker network disconnect all "$container"
      capture_network_traffic "$container"
      ;;
    "file_integrity")
      create_forensic_snapshot "$container"
      restore_from_backup "$container"
      ;;
  esac
}
```

## Compliance and Auditing

### CIS Docker Benchmark Compliance
```bash
#!/bin/bash
# Run docker-bench-security
docker run --rm --net host --pid host --userns host --cap-add audit_control \
  -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
  -v /var/lib:/var/lib:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  -v /etc:/etc:ro \
  docker/docker-bench-security
```

### Regular Security Audits
- Weekly vulnerability scans
- Monthly security configuration reviews
- Quarterly penetration testing
- Annual security architecture review

## Conclusion

This comprehensive security strategy provides multiple layers of protection for the media server infrastructure. Regular reviews and updates ensure continued effectiveness against evolving threats.