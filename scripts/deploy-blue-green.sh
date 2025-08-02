#!/bin/bash
# Blue-Green deployment script for Kubernetes

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=""
VERSION=""
NAMESPACE=""
TRAFFIC_SWITCH_PERCENTAGE="100"
VALIDATION_TIME="300" # 5 minutes
ROLLBACK_ON_FAILURE="true"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment)
            ENVIRONMENT="$2"
            NAMESPACE="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --traffic-percentage)
            TRAFFIC_SWITCH_PERCENTAGE="$2"
            shift 2
            ;;
        --validation-time)
            VALIDATION_TIME="$2"
            shift 2
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE="false"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$ENVIRONMENT" || -z "$VERSION" ]]; then
    echo -e "${RED}Error: --environment and --version are required${NC}"
    exit 1
fi

echo -e "${BLUE}Starting Blue-Green deployment${NC}"
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "Namespace: $NAMESPACE"
echo "Strategy: Blue-Green Deployment"

# Function to get current active color
get_active_color() {
    local service=$1
    local active_selector=$(kubectl get service "$service" -n "$NAMESPACE" -o jsonpath='{.spec.selector.version}' 2>/dev/null || echo "blue")
    
    if [[ "$active_selector" == "blue" ]]; then
        echo "blue"
    else
        echo "green"
    fi
}

# Function to get inactive color
get_inactive_color() {
    local active=$1
    if [[ "$active" == "blue" ]]; then
        echo "green"
    else
        echo "blue"
    fi
}

# Function to deploy to color
deploy_to_color() {
    local deployment=$1
    local color=$2
    local version=$3
    
    echo -e "${YELLOW}Deploying $deployment-$color with version $version${NC}"
    
    # Check if deployment exists, create if not
    if ! kubectl get deployment "$deployment-$color" -n "$NAMESPACE" > /dev/null 2>&1; then
        echo "Creating new deployment $deployment-$color"
        kubectl create deployment "$deployment-$color" \
            --image="$REGISTRY/$IMAGE_PREFIX-$deployment:$version" \
            -n "$NAMESPACE"
        
        # Set labels
        kubectl label deployment "$deployment-$color" \
            app="$deployment" \
            version="$color" \
            -n "$NAMESPACE" \
            --overwrite
    else
        # Update existing deployment
        kubectl set image deployment/"$deployment-$color" \
            "*=$REGISTRY/$IMAGE_PREFIX-$deployment:$version" \
            -n "$NAMESPACE" \
            --record
    fi
    
    # Wait for deployment to be ready
    echo "Waiting for $deployment-$color to be ready..."
    kubectl rollout status deployment/"$deployment-$color" -n "$NAMESPACE" --timeout=600s
}

# Function to run smoke tests
run_smoke_tests() {
    local deployment=$1
    local color=$2
    
    echo -e "${YELLOW}Running smoke tests on $deployment-$color...${NC}"
    
    # Get a pod from the deployment
    local pod=$(kubectl get pods -n "$NAMESPACE" -l app="$deployment",version="$color" -o jsonpath='{.items[0].metadata.name}')
    
    if [[ -z "$pod" ]]; then
        echo -e "${RED}No pods found for $deployment-$color${NC}"
        return 1
    fi
    
    # Run health check
    if kubectl exec -n "$NAMESPACE" "$pod" -- curl -f -s http://localhost/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Health check passed${NC}"
    else
        echo -e "${RED}✗ Health check failed${NC}"
        return 1
    fi
    
    # Run basic functionality tests
    if [[ -f "tests/smoke/run-tests.sh" ]]; then
        if kubectl exec -n "$NAMESPACE" "$pod" -- /app/tests/smoke/run-tests.sh; then
            echo -e "${GREEN}✓ Smoke tests passed${NC}"
        else
            echo -e "${RED}✗ Smoke tests failed${NC}"
            return 1
        fi
    fi
    
    return 0
}

# Function to switch traffic
switch_traffic() {
    local service=$1
    local target_color=$2
    local percentage=$3
    
    echo -e "${YELLOW}Switching $percentage% traffic to $target_color for $service${NC}"
    
    # Update service selector
    kubectl patch service "$service" -n "$NAMESPACE" -p '{"spec":{"selector":{"version":"'$target_color'"}}}'
    
    # If using Istio or other service mesh, update traffic rules
    if kubectl get virtualservice "$service" -n "$NAMESPACE" > /dev/null 2>&1; then
        echo "Updating Istio VirtualService for gradual traffic shift"
        # This would update Istio rules for canary-style traffic management
    fi
}

# Get list of services/deployments
SERVICES=$(kubectl get services -n "$NAMESPACE" -l app -o jsonpath='{.items[*].metadata.name}')

if [[ -z "$SERVICES" ]]; then
    echo -e "${RED}No services found in namespace $NAMESPACE${NC}"
    exit 1
fi

echo "Found services: $SERVICES"

# Deploy to inactive environment
for service in $SERVICES; do
    ACTIVE_COLOR=$(get_active_color "$service")
    INACTIVE_COLOR=$(get_inactive_color "$ACTIVE_COLOR")
    
    echo -e "${BLUE}Service $service: Active=$ACTIVE_COLOR, Deploying to=$INACTIVE_COLOR${NC}"
    
    # Deploy new version to inactive color
    deploy_to_color "$service" "$INACTIVE_COLOR" "$VERSION"
    
    # Run smoke tests on new deployment
    if ! run_smoke_tests "$service" "$INACTIVE_COLOR"; then
        echo -e "${RED}Smoke tests failed for $service-$INACTIVE_COLOR${NC}"
        
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            echo "Cleaning up failed deployment"
            kubectl delete deployment "$service-$INACTIVE_COLOR" -n "$NAMESPACE"
        fi
        exit 1
    fi
done

# Validation phase
echo -e "${YELLOW}Starting validation phase ($VALIDATION_TIME seconds)${NC}"
echo "New deployments are ready but not receiving traffic"

# Optional: Run load tests against inactive environment
if [[ -f "tests/load/run-tests.sh" ]]; then
    echo "Running load tests against inactive environment"
    for service in $SERVICES; do
        INACTIVE_COLOR=$(get_inactive_color "$(get_active_color "$service")")
        # Port-forward to test directly
        kubectl port-forward "deployment/$service-$INACTIVE_COLOR" 8080:80 -n "$NAMESPACE" &
        PF_PID=$!
        sleep 5
        
        ./tests/load/run-tests.sh --target http://localhost:8080 --duration 60
        
        kill $PF_PID 2>/dev/null || true
    done
fi

# Ask for confirmation before switching traffic
if [[ -t 0 ]]; then
    echo -e "${YELLOW}Ready to switch traffic to new version${NC}"
    read -p "Continue with traffic switch? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled"
        exit 1
    fi
fi

# Switch traffic to new version
echo -e "${BLUE}Switching traffic to new version${NC}"
for service in $SERVICES; do
    ACTIVE_COLOR=$(get_active_color "$service")
    INACTIVE_COLOR=$(get_inactive_color "$ACTIVE_COLOR")
    
    switch_traffic "$service" "$INACTIVE_COLOR" "$TRAFFIC_SWITCH_PERCENTAGE"
    
    echo -e "${GREEN}✓ Traffic switched for $service${NC}"
done

# Monitor new deployment
echo -e "${YELLOW}Monitoring new deployment for $VALIDATION_TIME seconds${NC}"
START_TIME=$(date +%s)
MONITORING_FAILED=false

while [[ $(($(date +%s) - START_TIME)) -lt $VALIDATION_TIME ]]; do
    for service in $SERVICES; do
        NEW_COLOR=$(get_active_color "$service")
        
        # Check deployment health
        READY=$(kubectl get deployment "$service-$NEW_COLOR" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Available")].status}')
        
        if [[ "$READY" != "True" ]]; then
            echo -e "${RED}✗ $service-$NEW_COLOR is not healthy${NC}"
            MONITORING_FAILED=true
            break
        fi
    done
    
    if [[ "$MONITORING_FAILED" == "true" ]]; then
        break
    fi
    
    echo -n "."
    sleep 10
done
echo

# Handle monitoring results
if [[ "$MONITORING_FAILED" == "true" ]] && [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
    echo -e "${RED}Monitoring failed, rolling back${NC}"
    
    for service in $SERVICES; do
        CURRENT_COLOR=$(get_active_color "$service")
        OLD_COLOR=$(get_inactive_color "$CURRENT_COLOR")
        
        switch_traffic "$service" "$OLD_COLOR" "100"
    done
    
    exit 1
fi

# Cleanup old deployments
echo -e "${YELLOW}Cleaning up old deployments${NC}"
for service in $SERVICES; do
    ACTIVE_COLOR=$(get_active_color "$service")
    OLD_COLOR=$(get_inactive_color "$ACTIVE_COLOR")
    
    echo "Scaling down $service-$OLD_COLOR"
    kubectl scale deployment "$service-$OLD_COLOR" --replicas=0 -n "$NAMESPACE"
    
    # Optionally delete after some time
    # kubectl delete deployment "$service-$OLD_COLOR" -n "$NAMESPACE"
done

echo -e "${GREEN}Blue-Green deployment completed successfully!${NC}"
echo "All services are now running version: $VERSION"

# Display deployment status
kubectl get deployments -n "$NAMESPACE" -o wide

exit 0