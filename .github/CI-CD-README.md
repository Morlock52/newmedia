# CI/CD Pipeline Documentation

## Overview

This repository implements a comprehensive CI/CD pipeline using GitHub Actions, designed for containerized microservices with multi-architecture support, security scanning, and advanced deployment strategies.

## Pipeline Architecture

### 🔄 Workflow Structure

1. **Docker Build & Push** (`docker-build.yml`)
   - Multi-architecture builds (amd64, arm64, arm/v7)
   - Automated security scanning with Trivy and Snyk
   - Container signing with Cosign
   - Push to GitHub Container Registry and Docker Hub
   - SBOM generation

2. **Security Scanning** (`security-scan.yml`)
   - Daily automated security scans
   - Container vulnerability scanning
   - Code analysis with CodeQL
   - Secret detection with TruffleHog
   - Infrastructure scanning with Checkov
   - License compliance checking

3. **Dependency Updates** (`dependency-update.yml`)
   - Automated dependency updates with Renovate
   - Dependabot configuration for all ecosystems
   - Security-first update strategy
   - Grouped updates by category

4. **Release Automation** (`release.yml`)
   - Semantic versioning
   - Automated changelog generation
   - Multi-registry image publishing
   - Helm chart packaging
   - GitHub Release creation

5. **Deployment** (`deploy.yml`)
   - Multiple deployment strategies (Rolling, Blue-Green, Canary)
   - Environment-specific configurations
   - Automated rollback capabilities
   - Post-deployment validation

6. **Test Suite** (`test-suite.yml`)
   - Unit tests with coverage reporting
   - Integration tests with real services
   - E2E tests with Playwright
   - Performance tests with k6
   - Security tests with OWASP ZAP

## 🚀 Quick Start

### Prerequisites

- GitHub repository with Actions enabled
- Docker Hub account (optional)
- Container registry credentials
- Kubernetes cluster for deployments

### Initial Setup

1. **Configure Secrets**
   ```
   Required GitHub Secrets:
   - DOCKER_HUB_USERNAME (optional)
   - DOCKER_HUB_TOKEN (optional)
   - RENOVATE_TOKEN
   - SLACK_WEBHOOK_URL (optional)
   - AWS_ROLE_ARN (for deployments)
   - PRODUCTION_SSH_KEY (for deployments)
   ```

2. **Enable Workflows**
   ```bash
   # Workflows are automatically enabled when pushed to .github/workflows/
   git add .github/workflows/
   git commit -m "ci: add GitHub Actions workflows"
   git push
   ```

3. **Configure Renovate**
   - Renovate will auto-discover the configuration
   - Visit the Dependency Dashboard issue after first run

## 📋 Workflow Details

### Docker Build Workflow

**Triggers:**
- Push to main/develop branches
- Pull requests
- Git tags (v*)
- Manual dispatch

**Features:**
- Automatic service discovery (finds all Dockerfiles)
- Multi-platform builds
- Layer caching for faster builds
- Security scanning before push
- Image signing for supply chain security

**Usage:**
```yaml
# Manual trigger with custom options
gh workflow run docker-build.yml \
  --ref main \
  -f push_images=true
```

### Security Scanning

**Runs:**
- Daily at 2 AM UTC
- On every push/PR
- Manual trigger

**Scans:**
- Container images (Trivy, Grype)
- Source code (CodeQL, Semgrep)
- Secrets (TruffleHog, GitGuardian)
- Infrastructure (Checkov, Terrascan)
- Dependencies (OWASP Dependency Check)
- Licenses (FOSSA, License Finder)

### Deployment Strategies

#### Rolling Deployment
```bash
./scripts/deploy-rolling.sh \
  --environment production \
  --version v1.2.3 \
  --max-surge 1 \
  --max-unavailable 0
```

#### Blue-Green Deployment
```bash
./scripts/deploy-blue-green.sh \
  --environment production \
  --version v1.2.3 \
  --traffic-percentage 100 \
  --validation-time 300
```

#### Canary Deployment
```bash
./scripts/deploy-canary.sh \
  --environment production \
  --version v1.2.3 \
  --steps "10,25,50,75,100" \
  --step-duration 300 \
  --auto-promote
```

## 🔧 Configuration

### Renovate Configuration

Located in `.github/renovate.json`:
- Automatic dependency updates
- Grouped updates by category
- Security updates prioritized
- Automerge for patch/minor updates

### Dependabot Configuration

Located in `.github/dependabot.yml`:
- Covers all package ecosystems
- Weekly update schedule
- Grouped updates for related packages
- Security updates created immediately

### Custom Configuration

#### Adding a New Service

1. Create Dockerfile in service directory
2. Workflows will auto-discover it
3. Ensure health check endpoint exists
4. Add service-specific tests

#### Modifying Build Platforms

Edit in `docker-build.yml`:
```yaml
env:
  PLATFORMS: linux/amd64,linux/arm64,linux/arm/v7
```

#### Changing Deployment Windows

Edit in `deploy.yml`:
```bash
# Allow deployments only on weekdays 9-17 UTC
if [[ $DAY -gt 5 ]] || [[ $HOUR -lt 9 ]] || [[ $HOUR -gt 17 ]]; then
```

## 📊 Monitoring & Notifications

### Slack Notifications

Configure `SLACK_WEBHOOK_URL` secret for notifications on:
- Deployment success/failure
- Security vulnerabilities found
- Release publications

### GitHub Status Checks

All workflows update commit statuses and deployment environments.

### Metrics Collection

Performance test results are uploaded as artifacts and can be integrated with monitoring systems.

## 🛡️ Security Best Practices

1. **Image Scanning**: Every image is scanned before push
2. **Secret Management**: No secrets in code, use GitHub Secrets
3. **SBOM Generation**: Software Bill of Materials for each release
4. **Dependency Updates**: Automated with security focus
5. **Container Signing**: Images signed with Cosign
6. **Least Privilege**: Minimal permissions for workflows

## 🔍 Troubleshooting

### Common Issues

1. **Build Failures**
   - Check Docker build logs
   - Verify Dockerfile syntax
   - Ensure base images are accessible

2. **Deployment Failures**
   - Verify Kubernetes credentials
   - Check namespace exists
   - Validate image tags

3. **Test Failures**
   - Review test logs in artifacts
   - Check service dependencies
   - Verify environment variables

### Debug Mode

Enable debug logging:
```yaml
env:
  ACTIONS_RUNNER_DEBUG: true
  ACTIONS_STEP_DEBUG: true
```

## 📈 Performance Optimization

1. **Build Caching**: Uses GitHub Actions cache and registry cache
2. **Parallel Execution**: Matrix builds for multiple services
3. **Conditional Steps**: Skip unnecessary work
4. **Artifact Retention**: Configurable retention periods

## 🤝 Contributing

1. Create feature branch
2. Add/modify workflows
3. Test with `act` locally (optional)
4. Submit PR with description
5. Workflows will validate changes

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Deployment Strategies](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Renovate Documentation](https://docs.renovatebot.com/)
- [Trivy Documentation](https://aquasecurity.github.io/trivy/)

## 📝 License

This CI/CD pipeline configuration is provided as-is for use in your projects.