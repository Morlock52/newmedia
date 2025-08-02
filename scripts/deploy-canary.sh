#!/bin/bash
# Canary deployment script for Kubernetes

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=""
VERSION=""
NAMESPACE=""
CANARY_PERCENTAGE="10"
CANARY_STEPS="10,25,50,75,100"
STEP_DURATION="300" # 5 minutes between steps
SUCCESS_THRESHOLD="99" # Success rate percentage
RESPONSE_TIME_THRESHOLD="1000" # milliseconds
AUTO_PROMOTE="false"
METRICS_ENDPOINT="http://prometheus:9090"

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
        --percentage)
            CANARY_PERCENTAGE="$2"
            shift 2
            ;;
        --steps)
            CANARY_STEPS="$2"
            shift 2
            ;;
        --step-duration)
            STEP_DURATION="$2"
            shift 2
            ;;
        --success-threshold)
            SUCCESS_THRESHOLD="$2"
            shift 2
            ;;
        --response-time-threshold)
            RESPONSE_TIME_THRESHOLD="$2"
            shift 2
            ;;
        --auto-promote)
            AUTO_PROMOTE="true"
            shift
            ;;
        --metrics-endpoint)
            METRICS_ENDPOINT="$2"
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

echo -e "${CYAN}Starting Canary deployment${NC}"
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "Namespace: $NAMESPACE"
echo "Strategy: Canary Deployment"
echo "Steps: $CANARY_STEPS"
echo "Auto-promote: $AUTO_PROMOTE"

# Function to check if deployment exists
deployment_exists() {
    kubectl get deployment "$1" -n "$NAMESPACE" > /dev/null 2>&1
}

# Function to create canary deployment
create_canary_deployment() {
    local base_deployment=$1
    local canary_name="${base_deployment}-canary"
    
    echo -e "${YELLOW}Creating canary deployment: $canary_name${NC}"
    
    # Get current deployment spec
    kubectl get deployment "$base_deployment" -n "$NAMESPACE" -o yaml | \
    sed "s/name: $base_deployment/name: $canary_name/g" | \
    sed "s/version: stable/version: canary/g" | \
    kubectl apply -f -
    
    # Update image to new version
    kubectl set image deployment/"$canary_name" \
        "*=$REGISTRY/$IMAGE_PREFIX-${base_deployment}:$VERSION" \
        -n "$NAMESPACE" \
        --record
    
    # Add canary labels
    kubectl label deployment "$canary_name" \
        version=canary \
        track=canary \
        -n "$NAMESPACE" \
        --overwrite
    
    # Scale canary to minimal replicas initially
    kubectl scale deployment "$canary_name" --replicas=1 -n "$NAMESPACE"
    
    # Wait for canary to be ready
    kubectl rollout status deployment/"$canary_name" -n "$NAMESPACE" --timeout=300s
}

# Function to update traffic split
update_traffic_split() {
    local service=$1
    local canary_percentage=$2
    local stable_percentage=$((100 - canary_percentage))
    
    echo -e "${YELLOW}Updating traffic split: Stable=$stable_percentage%, Canary=$canary_percentage%${NC}"
    
    # If using Istio
    if kubectl get virtualservice "$service" -n "$NAMESPACE" > /dev/null 2>&1; then
        cat <<EOF | kubectl apply -f -
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: $service
  namespace: $NAMESPACE
spec:
  hosts:
  - $service
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: $service
        subset: canary
      weight: 100
  - route:
    - destination:
        host: $service
        subset: stable
      weight: $stable_percentage
    - destination:
        host: $service
        subset: canary
      weight: $canary_percentage
EOF
    else
        # Using native Kubernetes with multiple services
        # Scale deployments proportionally
        local stable_replicas=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
        local total_replicas=$stable_replicas
        local canary_replicas=$(( (total_replicas * canary_percentage + 99) / 100 ))
        local new_stable_replicas=$(( total_replicas - canary_replicas ))
        
        kubectl scale deployment "$service" --replicas="$new_stable_replicas" -n "$NAMESPACE"
        kubectl scale deployment "$service-canary" --replicas="$canary_replicas" -n "$NAMESPACE"
    fi
}

# Function to get metrics
get_metrics() {
    local deployment=$1
    local metric_type=$2
    
    case "$metric_type" in
        "success_rate")
            # Query Prometheus for success rate
            curl -s "${METRICS_ENDPOINT}/api/v1/query" \
                --data-urlencode "query=sum(rate(http_requests_total{deployment=\"$deployment\",status=~\"2..\"}[5m])) / sum(rate(http_requests_total{deployment=\"$deployment\"}[5m])) * 100" | \
                jq -r '.data.result[0].value[1] // "100"'
            ;;
        "response_time")
            # Query Prometheus for p95 response time
            curl -s "${METRICS_ENDPOINT}/api/v1/query" \
                --data-urlencode "query=histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{deployment=\"$deployment\"}[5m])) by (le)) * 1000" | \
                jq -r '.data.result[0].value[1] // "0"'
            ;;
        "error_rate")
            # Query Prometheus for error rate
            curl -s "${METRICS_ENDPOINT}/api/v1/query" \
                --data-urlencode "query=sum(rate(http_requests_total{deployment=\"$deployment\",status=~\"5..\"}[5m])) / sum(rate(http_requests_total{deployment=\"$deployment\"}[5m])) * 100" | \
                jq -r '.data.result[0].value[1] // "0"'
            ;;
    esac
}

# Function to check canary health
check_canary_health() {
    local deployment=$1
    local canary_name="${deployment}-canary"
    
    echo -e "${YELLOW}Checking canary health metrics...${NC}"
    
    # Get metrics
    local success_rate=$(get_metrics "$canary_name" "success_rate")
    local response_time=$(get_metrics "$canary_name" "response_time")
    local error_rate=$(get_metrics "$canary_name" "error_rate")
    
    echo "Success Rate: ${success_rate}%"
    echo "Response Time (p95): ${response_time}ms"
    echo "Error Rate: ${error_rate}%"
    
    # Check thresholds
    if (( $(echo "$success_rate < $SUCCESS_THRESHOLD" | bc -l) )); then
        echo -e "${RED}✗ Success rate below threshold${NC}"
        return 1
    fi
    
    if (( $(echo "$response_time > $RESPONSE_TIME_THRESHOLD" | bc -l) )); then
        echo -e "${RED}✗ Response time above threshold${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ Canary health checks passed${NC}"
    return 0
}

# Function to rollback canary
rollback_canary() {
    local deployment=$1
    local canary_name="${deployment}-canary"
    
    echo -e "${RED}Rolling back canary deployment${NC}"
    
    # Scale down canary
    kubectl scale deployment "$canary_name" --replicas=0 -n "$NAMESPACE"
    
    # Restore full traffic to stable
    update_traffic_split "$deployment" 0
    
    # Delete canary deployment
    kubectl delete deployment "$canary_name" -n "$NAMESPACE"
}

# Function to promote canary
promote_canary() {
    local deployment=$1
    local canary_name="${deployment}-canary"
    
    echo -e "${GREEN}Promoting canary to stable${NC}"
    
    # Update stable deployment with canary image
    local canary_image=$(kubectl get deployment "$canary_name" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}')
    kubectl set image deployment/"$deployment" "*=$canary_image" -n "$NAMESPACE" --record
    
    # Wait for stable rollout
    kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout=600s
    
    # Scale down and remove canary
    kubectl scale deployment "$canary_name" --replicas=0 -n "$NAMESPACE"
    kubectl delete deployment "$canary_name" -n "$NAMESPACE"
    
    # Update labels
    kubectl label deployment "$deployment" version=stable release="$VERSION" -n "$NAMESPACE" --overwrite
}

# Get list of deployments
DEPLOYMENTS=$(kubectl get deployments -n "$NAMESPACE" -l track!=canary -o jsonpath='{.items[*].metadata.name}')

if [[ -z "$DEPLOYMENTS" ]]; then
    echo -e "${RED}No deployments found in namespace $NAMESPACE${NC}"
    exit 1
fi

echo "Found deployments: $DEPLOYMENTS"

# Create canary deployments
for deployment in $DEPLOYMENTS; do
    create_canary_deployment "$deployment"
done

# Progressive rollout
IFS=',' read -ra STEPS <<< "$CANARY_STEPS"
for step in "${STEPS[@]}"; do
    echo -e "${CYAN}=== Canary Step: ${step}% ===${NC}"
    
    # Update traffic split for all services
    for deployment in $DEPLOYMENTS; do
        update_traffic_split "$deployment" "$step"
    done
    
    # Wait for traffic to stabilize
    echo "Waiting 60 seconds for traffic to stabilize..."
    sleep 60
    
    # Monitor for step duration
    echo -e "${YELLOW}Monitoring canary for $STEP_DURATION seconds${NC}"
    START_TIME=$(date +%s)
    MONITORING_FAILED=false
    
    while [[ $(($(date +%s) - START_TIME)) -lt $STEP_DURATION ]]; do
        for deployment in $DEPLOYMENTS; do
            if ! check_canary_health "$deployment"; then
                MONITORING_FAILED=true
                break
            fi
        done
        
        if [[ "$MONITORING_FAILED" == "true" ]]; then
            break
        fi
        
        echo -n "."
        sleep 30
    done
    echo
    
    # Handle monitoring results
    if [[ "$MONITORING_FAILED" == "true" ]]; then
        echo -e "${RED}Canary health checks failed${NC}"
        
        # Rollback all canaries
        for deployment in $DEPLOYMENTS; do
            rollback_canary "$deployment"
        done
        
        exit 1
    fi
    
    # Ask for manual approval if not auto-promoting
    if [[ "$AUTO_PROMOTE" != "true" ]] && [[ "$step" != "100" ]] && [[ -t 0 ]]; then
        echo -e "${YELLOW}Canary at ${step}% - metrics look good${NC}"
        read -p "Continue to next step? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Canary deployment paused by user"
            exit 1
        fi
    fi
done

# Final promotion
echo -e "${GREEN}Canary deployment successful at 100%${NC}"

if [[ "$AUTO_PROMOTE" != "true" ]] && [[ -t 0 ]]; then
    read -p "Promote canary to stable? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Promotion cancelled by user"
        exit 1
    fi
fi

# Promote all canaries
for deployment in $DEPLOYMENTS; do
    promote_canary "$deployment"
done

echo -e "${GREEN}Canary deployment completed successfully!${NC}"
echo "All services are now running version: $VERSION"

# Display deployment status
kubectl get deployments -n "$NAMESPACE" -o wide

exit 0