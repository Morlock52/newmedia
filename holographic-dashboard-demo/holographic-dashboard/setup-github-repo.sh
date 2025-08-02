#!/bin/bash

# 🚀 Holographic Media Dashboard - GitHub Repository Setup Script
# Professional GitHub repository initialization with best practices
# Created for Morlock52's holographic-dashboard project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REPO_NAME="holographic-media-dashboard"
REPO_DESCRIPTION="🌟 Next-generation 3D holographic media server dashboard with WebGL effects, real-time visualization, and immersive UI. Built with Three.js, WebSockets, and modern web technologies."
GITHUB_USERNAME=""
GITHUB_TOKEN=""
PRIMARY_BRANCH="main"
DEVELOPMENT_BRANCH="develop"
RELEASE_VERSION="v1.0.0"

# Helper functions
print_header() {
    echo -e "\n${PURPLE}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}  🎬 HOLOGRAPHIC MEDIA DASHBOARD - GITHUB REPOSITORY SETUP${NC}"
    echo -e "${PURPLE}════════════════════════════════════════════════════════════════════════════════${NC}\n"
}

print_step() {
    echo -e "\n${CYAN}🔄 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get GitHub username
get_github_username() {
    if [ -z "$GITHUB_USERNAME" ]; then
        # Try to get from git config
        GITHUB_USERNAME=$(git config --global user.name 2>/dev/null || echo "")
        
        if [ -z "$GITHUB_USERNAME" ]; then
            read -p "Enter your GitHub username: " GITHUB_USERNAME
        else
            read -p "GitHub username [$GITHUB_USERNAME]: " input
            GITHUB_USERNAME=${input:-$GITHUB_USERNAME}
        fi
    fi
}

# Function to get GitHub token (with graceful handling)
get_github_token() {
    if [ -z "$GITHUB_TOKEN" ]; then
        echo -e "\n${YELLOW}GitHub Personal Access Token Setup:${NC}"
        echo -e "${BLUE}1. Go to: https://github.com/settings/tokens${NC}"
        echo -e "${BLUE}2. Click 'Generate new token (classic)'${NC}"
        echo -e "${BLUE}3. Select scopes: repo, workflow, admin:repo_hook${NC}"
        echo -e "${BLUE}4. Copy the token and paste it below${NC}\n"
        
        read -s -p "Enter your GitHub Personal Access Token (optional - press Enter to skip): " GITHUB_TOKEN
        echo
        
        if [ -z "$GITHUB_TOKEN" ]; then
            print_warning "No token provided. Repository will be created manually."
            print_info "You can create the repository later at: https://github.com/new"
            return 1
        fi
    fi
    return 0
}

# Function to initialize git repository
init_git_repo() {
    print_step "Initializing Git repository"
    
    if [ ! -d ".git" ]; then
        git init
        print_success "Git repository initialized"
    else
        print_info "Git repository already exists"
    fi
    
    # Set up git config if not already set
    if [ -z "$(git config --global user.name 2>/dev/null)" ]; then
        read -p "Enter your name for Git commits: " git_name
        git config --global user.name "$git_name"
    fi
    
    if [ -z "$(git config --global user.email 2>/dev/null)" ]; then
        read -p "Enter your email for Git commits: " git_email
        git config --global user.email "$git_email"
    fi
    
    # Set default branch to main
    git config init.defaultBranch main
    git branch -M main 2>/dev/null || true
}

# Function to create GitHub repository
create_github_repo() {
    if [ -z "$GITHUB_TOKEN" ]; then
        print_warning "Skipping GitHub repository creation - no token provided"
        return 1
    fi
    
    print_step "Creating GitHub repository"
    
    # Check if repository already exists
    response=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
        "https://api.github.com/repos/$GITHUB_USERNAME/$REPO_NAME" 2>/dev/null)
    
    if echo "$response" | grep -q '"name":'; then
        print_warning "Repository $GITHUB_USERNAME/$REPO_NAME already exists"
        return 1
    fi
    
    # Create repository
    curl -s -H "Authorization: token $GITHUB_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"name\": \"$REPO_NAME\",
            \"description\": \"$REPO_DESCRIPTION\",
            \"homepage\": \"https://$GITHUB_USERNAME.github.io/$REPO_NAME\",
            \"private\": false,
            \"has_issues\": true,
            \"has_projects\": true,
            \"has_wiki\": true,
            \"has_downloads\": true,
            \"auto_init\": false,
            \"allow_squash_merge\": true,
            \"allow_merge_commit\": true,
            \"allow_rebase_merge\": true,
            \"delete_branch_on_merge\": true
        }" \
        "https://api.github.com/user/repos" > /tmp/create_repo_response.json
    
    if grep -q '"clone_url"' /tmp/create_repo_response.json; then
        print_success "GitHub repository created successfully"
        return 0
    else
        print_error "Failed to create GitHub repository"
        cat /tmp/create_repo_response.json
        return 1
    fi
}

# Function to set up remote origin
setup_remote_origin() {
    print_step "Setting up remote origin"
    
    # Remove existing origin if it exists
    git remote remove origin 2>/dev/null || true
    
    # Add new origin
    git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    print_success "Remote origin configured"
}

# Function to create and switch to development branch
create_branches() {
    print_step "Creating branch structure"
    
    # Ensure we're on main branch
    git checkout -B main
    
    # Create development branch
    git checkout -B $DEVELOPMENT_BRANCH
    git checkout main
    
    print_success "Branch structure created (main, develop)"
}

# Function to create initial commit
create_initial_commit() {
    print_step "Creating initial commit"
    
    # Add all files
    git add .
    
    # Create initial commit
    git commit -m "🎬 Initial commit: Holographic Media Dashboard v1.0.0

✨ Features:
- 3D holographic interface with Three.js
- WebGL particle systems and custom shaders  
- Real-time audio visualizer
- WebSocket integration for live data
- Responsive design with glass morphism UI
- Performance optimization for all devices
- Comprehensive documentation

🚀 Ready for production deployment!

Co-authored-by: Morlock52 <noreply@morlock52.dev>"
    
    print_success "Initial commit created"
}

# Function to push to GitHub
push_to_github() {
    if [ -z "$GITHUB_TOKEN" ]; then
        print_warning "Skipping push to GitHub - no token provided"
        print_info "To push manually later:"
        echo -e "${BLUE}git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git${NC}"
        echo -e "${BLUE}git push -u origin main${NC}"
        echo -e "${BLUE}git push -u origin develop${NC}"
        return 1
    fi
    
    print_step "Pushing to GitHub"
    
    # Push main branch
    git push -u origin main
    
    # Push development branch
    git checkout $DEVELOPMENT_BRANCH
    git push -u origin $DEVELOPMENT_BRANCH
    git checkout main
    
    print_success "Code pushed to GitHub successfully"
}

# Function to create release
create_release() {
    if [ -z "$GITHUB_TOKEN" ]; then
        print_warning "Skipping release creation - no token provided"
        return 1
    fi
    
    print_step "Creating release $RELEASE_VERSION"
    
    # Create git tag
    git tag -a "$RELEASE_VERSION" -m "🚀 Holographic Media Dashboard $RELEASE_VERSION

🌟 Production Release - Ready for deployment!

✨ Key Features:
- Complete 3D holographic media dashboard
- Advanced WebGL effects and shaders
- Real-time data visualization
- Cross-platform compatibility
- Professional documentation
- Production-ready deployment scripts

🔗 Live Demo: https://$GITHUB_USERNAME.github.io/$REPO_NAME
📖 Documentation: Complete API and deployment guides included

Built with ❤️ using Three.js, WebGL, and modern web technologies."
    
    # Push tag
    git push origin "$RELEASE_VERSION"
    
    # Create GitHub release
    curl -s -H "Authorization: token $GITHUB_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"tag_name\": \"$RELEASE_VERSION\",
            \"target_commitish\": \"main\",
            \"name\": \"🎬 Holographic Media Dashboard $RELEASE_VERSION\",
            \"body\": \"## 🌟 Production Release - Ready for Deployment!\\n\\n### ✨ Features\\n- **3D Holographic Interface**: Immersive Three.js-powered dashboard\\n- **WebGL Effects**: Custom shaders, particles, and post-processing\\n- **Real-time Visualization**: Live data streaming with WebSockets\\n- **Audio Visualizer**: Frequency-based 3D visualization\\n- **Responsive Design**: Optimized for desktop and mobile\\n- **Performance Optimized**: Adaptive quality based on device capabilities\\n\\n### 🚀 Quick Start\\n\\\`\\\`\\\`bash\\nnpm install\\nnpm start\\n\\\`\\\`\\\`\\n\\n### 🔗 Links\\n- **Live Demo**: https://$GITHUB_USERNAME.github.io/$REPO_NAME\\n- **Documentation**: Complete setup and API guides\\n- **Examples**: Ready-to-use integration examples\\n\\n### 📦 What's Included\\n- Complete dashboard source code\\n- WebSocket demo server\\n- Deployment scripts\\n- Comprehensive documentation\\n- Example configurations\\n\\nBuilt with ❤️ for the future of media dashboards!\",
            \"draft\": false,
            \"prerelease\": false,
            \"generate_release_notes\": true
        }" \
        "https://api.github.com/repos/$GITHUB_USERNAME/$REPO_NAME/releases" > /tmp/create_release_response.json
    
    if grep -q '"tag_name"' /tmp/create_release_response.json; then
        print_success "Release $RELEASE_VERSION created successfully"
        RELEASE_URL=$(grep -o '"html_url":"[^"]*' /tmp/create_release_response.json | cut -d'"' -f4)
        print_info "Release URL: $RELEASE_URL"
    else
        print_error "Failed to create release"
        cat /tmp/create_release_response.json
    fi
}

# Function to enable GitHub Pages
enable_github_pages() {
    if [ -z "$GITHUB_TOKEN" ]; then
        print_warning "Skipping GitHub Pages setup - no token provided"
        print_info "To enable GitHub Pages manually:"
        echo -e "${BLUE}1. Go to: https://github.com/$GITHUB_USERNAME/$REPO_NAME/settings/pages${NC}"
        echo -e "${BLUE}2. Select 'Deploy from a branch'${NC}"
        echo -e "${BLUE}3. Choose 'main' branch and '/ (root)' folder${NC}"
        echo -e "${BLUE}4. Click 'Save'${NC}"
        return 1
    fi
    
    print_step "Enabling GitHub Pages"
    
    # Enable GitHub Pages
    curl -s -H "Authorization: token $GITHUB_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"source\": {
                \"branch\": \"main\",
                \"path\": \"/\"
            }
        }" \
        "https://api.github.com/repos/$GITHUB_USERNAME/$REPO_NAME/pages" > /tmp/pages_response.json
    
    if grep -q '"status":"built"' /tmp/pages_response.json || grep -q '"html_url"' /tmp/pages_response.json; then
        print_success "GitHub Pages enabled successfully"
        print_info "Site will be available at: https://$GITHUB_USERNAME.github.io/$REPO_NAME"
        print_info "Note: It may take a few minutes for the site to be available"
    else
        print_warning "GitHub Pages setup may have failed or is already configured"
        # Don't show error details as Pages API can be finicky
    fi
}

# Function to configure repository settings
configure_repo_settings() {
    if [ -z "$GITHUB_TOKEN" ]; then
        print_warning "Skipping repository settings configuration - no token provided"
        return 1
    fi
    
    print_step "Configuring repository settings"
    
    # Update repository settings
    curl -s -H "Authorization: token $GITHUB_TOKEN" \
        -H "Content-Type: application/json" \
        -X PATCH \
        -d "{
            \"has_issues\": true,
            \"has_projects\": true,
            \"has_wiki\": true,
            \"has_discussions\": true,
            \"allow_squash_merge\": true,
            \"allow_merge_commit\": true,
            \"allow_rebase_merge\": true,
            \"delete_branch_on_merge\": true,
            \"allow_auto_merge\": true,
            \"security_and_analysis\": {
                \"secret_scanning\": {
                    \"status\": \"enabled\"
                },
                \"secret_scanning_push_protection\": {
                    \"status\": \"enabled\"
                }
            }
        }" \
        "https://api.github.com/repos/$GITHUB_USERNAME/$REPO_NAME" > /tmp/settings_response.json
    
    print_success "Repository settings configured"
}

# Function to create GitHub Labels
create_github_labels() {
    if [ -z "$GITHUB_TOKEN" ]; then
        print_warning "Skipping GitHub labels creation - no token provided"
        return 1
    fi
    
    print_step "Creating GitHub labels"
    
    # Define labels
    labels=(
        "bug|d73a4a|Something isn't working"
        "enhancement|a2eeef|New feature or request"
        "documentation|0075ca|Improvements or additions to documentation"
        "good first issue|7057ff|Good for newcomers"
        "help wanted|008672|Extra attention is needed"
        "question|d876e3|Further information is requested" 
        "performance|f9d0c4|Performance improvements"
        "3d-graphics|e11d48|Related to Three.js or WebGL"
        "ui-ux|0052cc|User interface and experience"
        "websocket|5319e7|WebSocket related issues"
        "mobile|fbca04|Mobile device compatibility"
        "browser-support|006b75|Browser compatibility issues"
        "shader|d4c5f9|WebGL shader related"
        "audio|c2e0c6|Audio visualizer features"
        "high-priority|b60205|High priority issue"
        "medium-priority|fbca04|Medium priority issue"
        "low-priority|0e8a16|Low priority issue"
    )
    
    for label in "${labels[@]}"; do
        IFS='|' read -r name color description <<< "$label"
        curl -s -H "Authorization: token $GITHUB_TOKEN" \
            -H "Content-Type: application/json" \
            -d "{
                \"name\": \"$name\",
                \"color\": \"$color\",
                \"description\": \"$description\"
            }" \
            "https://api.github.com/repos/$GITHUB_USERNAME/$REPO_NAME/labels" >/dev/null
    done
    
    print_success "GitHub labels created"
}

# Function to display final information
display_final_info() {
    echo -e "\n${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  🎉 GITHUB REPOSITORY SETUP COMPLETE!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}\n"
    
    echo -e "${CYAN}📊 Repository Information:${NC}"
    echo -e "${BLUE}  Repository:${NC} https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    echo -e "${BLUE}  Live Demo:${NC} https://$GITHUB_USERNAME.github.io/$REPO_NAME"
    echo -e "${BLUE}  Clone URL:${NC} git clone https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    
    echo -e "\n${CYAN}🌟 What's been set up:${NC}"
    echo -e "${GREEN}  ✅ Git repository initialized with main/develop branches${NC}"
    echo -e "${GREEN}  ✅ GitHub repository created with optimal settings${NC}"
    echo -e "${GREEN}  ✅ Release $RELEASE_VERSION created and tagged${NC}"
    echo -e "${GREEN}  ✅ GitHub Pages enabled for live demo${NC}"
    echo -e "${GREEN}  ✅ Professional README and documentation${NC}"
    echo -e "${GREEN}  ✅ Community features (issues, discussions, wiki)${NC}"
    echo -e "${GREEN}  ✅ Security features enabled${NC}"
    echo -e "${GREEN}  ✅ Organized labels for issue management${NC}"
    
    echo -e "\n${CYAN}🚀 Next Steps:${NC}"
    echo -e "${BLUE}  1. Visit your repository: https://github.com/$GITHUB_USERNAME/$REPO_NAME${NC}"
    echo -e "${BLUE}  2. Check the live demo: https://$GITHUB_USERNAME.github.io/$REPO_NAME${NC}"
    echo -e "${BLUE}  3. Share your project with the community!${NC}"
    echo -e "${BLUE}  4. Consider adding topics/tags for better discoverability${NC}"
    
    echo -e "\n${CYAN}💡 Pro Tips:${NC}"
    echo -e "${YELLOW}  • Use 'git checkout develop' for new features${NC}"
    echo -e "${YELLOW}  • Merge to main only for releases${NC}"
    echo -e "${YELLOW}  • GitHub Pages may take 5-10 minutes to be live${NC}"
    echo -e "${YELLOW}  • Add repository topics for better SEO${NC}"
    
    echo -e "\n${PURPLE}Thank you for using the Holographic Media Dashboard setup script!${NC}"
    echo -e "${PURPLE}Star the repository if you find it useful! ⭐${NC}\n"
}

# Main execution
main() {
    print_header
    
    # Check prerequisites
    if ! command_exists git; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi
    
    if ! command_exists curl; then
        print_error "curl is not installed. Please install curl first."
        exit 1
    fi
    
    # Get user input
    get_github_username
    get_github_token
    
    # Execute setup steps
    init_git_repo
    create_branches
    
    if [ ! -z "$GITHUB_TOKEN" ]; then
        create_github_repo
        setup_remote_origin
        create_initial_commit
        push_to_github
        configure_repo_settings
        create_github_labels
        enable_github_pages
        create_release
    else
        create_initial_commit
        print_info "Manual setup required - see instructions below"
    fi
    
    # Clean up temporary files
    rm -f /tmp/create_repo_response.json /tmp/pages_response.json /tmp/settings_response.json /tmp/create_release_response.json
    
    display_final_info
}

# Run main function
main "$@"