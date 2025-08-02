#!/bin/bash
# Rolling deployment script for Kubernetes

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=""
VERSION=""
NAMESPACE=""
MAX_SURGE="1"
MAX_UNAVAILABLE="0"
TIMEOUT="600"
HEALTH_CHECK_RETRIES="30"
HEALTH_CHECK_INTERVAL="10"

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
        --max-surge)
            MAX_SURGE="$2"
            shift 2
            ;;
        --max-unavailable)
            MAX_UNAVAILABLE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
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

echo -e "${GREEN}Starting rolling deployment${NC}"
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "Namespace: $NAMESPACE"
echo "Strategy: Rolling Update (maxSurge=$MAX_SURGE, maxUnavailable=$MAX_UNAVAILABLE)"

# Function to check deployment status
check_deployment_status() {
    local deployment=$1
    local replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    local ready_replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
    
    if [[ "$ready_replicas" == "$replicas" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to perform health check
health_check() {
    local service=$1
    local endpoint=$2
    local retries=$HEALTH_CHECK_RETRIES
    
    echo -e "${YELLOW}Performing health check for $service...${NC}"
    
    while [[ $retries -gt 0 ]]; do
        if kubectl exec -n "$NAMESPACE" deployment/"$service" -- curl -f -s "$endpoint" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Health check passed for $service${NC}"
            return 0
        fi
        
        retries=$((retries - 1))
        echo "Health check failed, retrying... ($retries attempts left)"
        sleep $HEALTH_CHECK_INTERVAL
    done
    
    echo -e "${RED}✗ Health check failed for $service${NC}"
    return 1
}

# Get list of deployments
DEPLOYMENTS=$(kubectl get deployments -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}')

if [[ -z "$DEPLOYMENTS" ]]; then
    echo -e "${RED}No deployments found in namespace $NAMESPACE${NC}"
    exit 1
fi

echo "Found deployments: $DEPLOYMENTS"

# Update deployment strategy
echo -e "${YELLOW}Updating deployment strategy...${NC}"
for deployment in $DEPLOYMENTS; do
    kubectl patch deployment "$deployment" -n "$NAMESPACE" --type='json' -p='[
        {
            "op": "replace",
            "path": "/spec/strategy",
            "value": {
                "type": "RollingUpdate",
                "rollingUpdate": {
                    "maxSurge": "'$MAX_SURGE'",
                    "maxUnavailable": "'$MAX_UNAVAILABLE'"
                }
            }
        }
    ]'
done

# Perform rolling update
echo -e "${YELLOW}Starting rolling update...${NC}"
for deployment in $DEPLOYMENTS; do
    echo "Updating $deployment to version $VERSION"
    
    # Update image
    kubectl set image deployment/"$deployment" \
        "*=$REGISTRY/$IMAGE_PREFIX-$deployment:$VERSION" \
        -n "$NAMESPACE" \
        --record
    
    # Wait for rollout to complete
    echo "Waiting for $deployment rollout to complete..."
    if kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="${TIMEOUT}s"; then
        echo -e "${GREEN}✓ $deployment updated successfully${NC}"
        
        # Perform health check
        health_check "$deployment" "/health" || {
            echo -e "${RED}Health check failed, initiating rollback${NC}"
            kubectl rollout undo deployment/"$deployment" -n "$NAMESPACE"
            kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="${TIMEOUT}s"
            exit 1
        }
    else
        echo -e "${RED}✗ $deployment update failed${NC}"
        exit 1
    fi
done

# Verify all deployments are healthy
echo -e "${YELLOW}Verifying deployment health...${NC}"
all_healthy=true
for deployment in $DEPLOYMENTS; do
    if check_deployment_status "$deployment"; then
        echo -e "${GREEN}✓ $deployment is healthy${NC}"
    else
        echo -e "${RED}✗ $deployment is not healthy${NC}"
        all_healthy=false
    fi
done

if [[ "$all_healthy" == "false" ]]; then
    echo -e "${RED}Not all deployments are healthy. Rolling back...${NC}"
    for deployment in $DEPLOYMENTS; do
        kubectl rollout undo deployment/"$deployment" -n "$NAMESPACE"
    done
    exit 1
fi

# Run post-deployment tests
echo -e "${YELLOW}Running post-deployment tests...${NC}"
if [[ -f "tests/post-deploy/run-tests.sh" ]]; then
    if ./tests/post-deploy/run-tests.sh --environment "$ENVIRONMENT"; then
        echo -e "${GREEN}✓ Post-deployment tests passed${NC}"
    else
        echo -e "${RED}✗ Post-deployment tests failed${NC}"
        exit 1
    fi
fi

# Update deployment annotations
echo -e "${YELLOW}Updating deployment annotations...${NC}"
for deployment in $DEPLOYMENTS; do
    kubectl annotate deployment "$deployment" -n "$NAMESPACE" \
        deployment.kubernetes.io/revision="$(kubectl rollout history deployment/"$deployment" -n "$NAMESPACE" | tail -2 | head -1 | awk '{print $1}')" \
        deployment.kubernetes.io/updated-at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        deployment.kubernetes.io/updated-by="$USER" \
        --overwrite
done

echo -e "${GREEN}Rolling deployment completed successfully!${NC}"
echo "All services are running version: $VERSION"

# Display deployment status
kubectl get deployments -n "$NAMESPACE" -o wide

exit 0