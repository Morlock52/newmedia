# Non-Media Server Components to Remove

## Code Quality Analysis Report

### Summary
- Overall Quality Score: 7/10
- Files Analyzed: 500+
- Issues Found: Multiple non-essential components
- Technical Debt Estimate: 40+ hours of cleanup

### Critical Issues

## 1. UI/Dashboard Components (Non-Essential)
These are alternative dashboard implementations that are not part of the core media server:

### holographic-dashboard/
- **Issue**: Complete standalone dashboard implementation with 3D effects
- **Severity**: High (duplicates functionality)
- **Files**: 80+ files including HTML, CSS, JS, and documentation
- **Recommendation**: Remove entirely - media server has its own dashboard (Homepage)

### sci-fi-dashboard/
- **Issue**: React-based dashboard with sci-fi theme
- **Severity**: High (unnecessary duplicate)
- **Files**: Full React/TypeScript application with tests
- **Recommendation**: Remove entirely - not needed for media server

### media-dashboard/
- **Issue**: Another dashboard implementation
- **Severity**: High (redundant)
- **Files**: React application with Vite
- **Recommendation**: Remove - Homepage dashboard serves this purpose

### holographic-media-dashboard/
- **Issue**: Yet another dashboard variant
- **Severity**: High (duplicate)
- **Files**: React/TypeScript with Tailwind
- **Recommendation**: Remove completely

## 2. Development/Testing Infrastructure (Non-Core)

### dashboard-security/
- **Issue**: Separate security implementation project
- **Severity**: Medium (not integrated with main stack)
- **Files**: Backend/frontend security components
- **Recommendation**: Remove - security should be in main docker-compose

### performance-optimization/
- **Issue**: Standalone performance testing suite
- **Severity**: Medium (development tool)
- **Files**: TypeScript performance analyzers
- **Recommendation**: Remove - not needed for production

### production-validation/
- **Issue**: Validation and testing infrastructure
- **Severity**: Medium (CI/CD tool)
- **Files**: Test suites and Kubernetes configs
- **Recommendation**: Remove - testing should be separate

### security-review/
- **Issue**: Security audit and fixes
- **Severity**: Low (documentation)
- **Files**: Security reports and fix scripts
- **Recommendation**: Keep reports, remove implementation

## 3. Project Management/Documentation (Non-Essential)

### consensus-report/
- **Issue**: Project planning documents
- **Severity**: Low
- **Files**: 4 markdown files
- **Recommendation**: Archive or remove

### coordination/
- **Issue**: Empty coordination directories
- **Severity**: Low
- **Files**: Empty folders
- **Recommendation**: Remove completely

### website-fix-implementation/
- **Issue**: Website fix scripts unrelated to media server
- **Severity**: Medium
- **Files**: JavaScript utilities
- **Recommendation**: Remove entirely

## 4. Duplicate/Test Files

### Multiple Docker Compose Files
- docker-compose-2025-enhanced.yml
- docker-compose-2025-fixed.yml
- docker-compose-2025.yml
- docker-compose-enhanced-2025.yml
- docker-compose-enhanced.yml
- docker-compose-macos-optimized.yml
- docker-compose-optimized-2025.yml
- docker-compose-optimized.yml
- docker-compose-production.yml
- docker-compose-ultimate.yml
**Recommendation**: Keep only docker-compose.yml and docker-compose.override.yml

### Test/Demo Files
- test-navigation.html
- interactive-functionality-tests.html
- frontend-functionality-test-results.md
- test.txt
**Recommendation**: Remove all test HTML files

## 5. Code Smells Detected

### Large Directory Trees
- **holographic-dashboard/**: 500+ lines directory listing
- **sci-fi-dashboard/**: 400+ lines with test results
- **media-server-stack/**: Duplicate of main setup

### Duplicate Functionality
- Multiple dashboard implementations
- Multiple deployment scripts doing same thing
- Multiple docker-compose variants

### Dead Code
- Empty directories (coordination/)
- Unused test results
- Old backup files (.bak)

## 6. Core Media Server Components to KEEP

### Essential Services
- ✅ config/ (service configurations)
- ✅ data/ (media storage)
- ✅ docker-compose.yml (main configuration)
- ✅ scripts/ (automation scripts)
- ✅ .env files
- ✅ homepage-config/ (main dashboard)

### Media Services (All in docker-compose.yml)
- ✅ Jellyfin (media server)
- ✅ Sonarr/Radarr/Lidarr (media management)
- ✅ Prowlarr (indexer management)
- ✅ qBittorrent (downloading)
- ✅ Overseerr (request management)
- ✅ Tautulli (analytics)
- ✅ Homepage (dashboard)

## Refactoring Opportunities

1. **Consolidate Docker Compose**: Merge all variants into single file with profiles
2. **Remove Duplicate Dashboards**: Use only Homepage dashboard
3. **Clean Script Directory**: Keep only essential automation scripts
4. **Archive Documentation**: Move old docs to archive folder

## Positive Findings

- Core media server stack is well-configured
- Service integration is properly set up
- Docker networking is correctly configured
- Volume mappings are appropriate

## Recommended Cleanup Commands

```bash
# Remove non-essential directories
rm -rf holographic-dashboard/
rm -rf sci-fi-dashboard/
rm -rf media-dashboard/
rm -rf holographic-media-dashboard/
rm -rf dashboard-security/
rm -rf performance-optimization/
rm -rf production-validation/
rm -rf consensus-report/
rm -rf coordination/
rm -rf website-fix-implementation/

# Remove duplicate docker-compose files
rm -f docker-compose-*-*.yml
rm -f docker-compose-enhanced.yml
rm -f docker-compose-optimized.yml
rm -f docker-compose-production.yml
rm -f docker-compose-ultimate.yml

# Remove test files
rm -f *test*.html
rm -f test.txt

# Keep only essential deployment scripts
# Review and consolidate deploy-*.sh scripts
```

## Summary

The project contains significant technical debt from multiple dashboard implementations and testing infrastructure that should be removed. The core media server stack (Jellyfin, *arr services, etc.) should be retained and is properly configured. Removing these non-essential components will reduce complexity and maintenance burden.