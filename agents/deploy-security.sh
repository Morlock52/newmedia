#!/bin/bash

# Security Manager Deployment Script
# Deploys comprehensive 2025 security infrastructure

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SECURITY_NAMESPACE="consensus-security"
CONSENSUS_NAMESPACE="consensus-system"
CERT_DIR="./certs"
SECRETS_DIR="./secrets"
CONFIG_DIR="./security-config"

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check for required commands
    local required_commands=("docker" "kubectl" "helm" "openssl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            print_error "$cmd is not installed. Please install it first."
            exit 1
        fi
    done
    
    # Check Kubernetes connectivity
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Please configure kubectl."
        exit 1
    fi
    
    print_success "All prerequisites met"
}

# Generate certificates for mTLS
generate_certificates() {
    print_info "Generating mTLS certificates..."
    
    mkdir -p "$CERT_DIR"/{ca,server,client}
    
    # Generate CA
    if [ ! -f "$CERT_DIR/ca/ca.key" ]; then
        openssl genrsa -out "$CERT_DIR/ca/ca.key" 4096
        openssl req -new -x509 -days 3650 -key "$CERT_DIR/ca/ca.key" \
            -out "$CERT_DIR/ca/ca.crt" \
            -subj "/C=US/ST=CA/L=SF/O=Consensus/CN=Consensus CA"
    fi
    
    # Generate server certificates
    for service in keycloak vault grafana consensus-api; do
        if [ ! -f "$CERT_DIR/server/$service.key" ]; then
            openssl genrsa -out "$CERT_DIR/server/$service.key" 4096
            openssl req -new -key "$CERT_DIR/server/$service.key" \
                -out "$CERT_DIR/server/$service.csr" \
                -subj "/C=US/ST=CA/L=SF/O=Consensus/CN=$service.consensus.local"
            openssl x509 -req -in "$CERT_DIR/server/$service.csr" \
                -CA "$CERT_DIR/ca/ca.crt" -CAkey "$CERT_DIR/ca/ca.key" \
                -CAcreateserial -out "$CERT_DIR/server/$service.crt" \
                -days 365 -sha256 \
                -extfile <(echo "subjectAltName=DNS:$service.consensus.local,DNS:$service")
        fi
    done
    
    # Create combined cert files for services
    for service in keycloak vault grafana; do
        cat "$CERT_DIR/server/$service.crt" "$CERT_DIR/ca/ca.crt" > "$CERT_DIR/server/$service-fullchain.crt"
    done
    
    print_success "Certificates generated"
}

# Generate secure passwords and secrets
generate_secrets() {
    print_info "Generating secrets..."
    
    mkdir -p "$SECRETS_DIR"
    
    # Generate random passwords
    local secrets=(
        "keycloak_db_password"
        "keycloak_admin_password"
        "vault_root_token"
        "grafana_admin_password"
        "splunk_password"
        "consensus_api_key"
    )
    
    for secret in "${secrets[@]}"; do
        if [ ! -f "$SECRETS_DIR/$secret" ]; then
            openssl rand -base64 32 > "$SECRETS_DIR/$secret"
        fi
    done
    
    # Generate OAuth secrets
    if [ ! -f "$SECRETS_DIR/grafana_oauth_secret" ]; then
        openssl rand -hex 32 > "$SECRETS_DIR/grafana_oauth_secret"
    fi
    
    # Create Docker secrets
    for secret_file in "$SECRETS_DIR"/*; do
        secret_name=$(basename "$secret_file")
        if ! docker secret inspect "$secret_name" &> /dev/null; then
            docker secret create "$secret_name" "$secret_file"
            print_success "Created Docker secret: $secret_name"
        fi
    done
    
    print_success "Secrets generated and stored"
}

# Create Kubernetes namespaces
create_namespaces() {
    print_info "Creating Kubernetes namespaces..."
    
    for namespace in "$SECURITY_NAMESPACE" "$CONSENSUS_NAMESPACE"; do
        if ! kubectl get namespace "$namespace" &> /dev/null; then
            kubectl create namespace "$namespace"
            kubectl label namespace "$namespace" \
                pod-security.kubernetes.io/enforce=restricted \
                pod-security.kubernetes.io/audit=restricted \
                pod-security.kubernetes.io/warn=restricted
            print_success "Created namespace: $namespace"
        fi
    done
}

# Deploy network policies
deploy_network_policies() {
    print_info "Deploying network policies..."
    
    kubectl apply -f "$CONFIG_DIR/network-policies.yaml"
    
    # Install Cilium if not present
    if ! kubectl get deployment -n kube-system cilium-operator &> /dev/null; then
        print_info "Installing Cilium CNI..."
        helm repo add cilium https://helm.cilium.io/
        helm install cilium cilium/cilium --version 1.14.0 \
            --namespace kube-system \
            --set global.encryption.enabled=true \
            --set global.encryption.type=wireguard \
            --set global.hubble.relay.enabled=true \
            --set global.hubble.ui.enabled=true
    fi
    
    print_success "Network policies deployed"
}

# Deploy Keycloak
deploy_keycloak() {
    print_info "Deploying Keycloak..."
    
    # Create TLS secret
    kubectl create secret tls keycloak-tls \
        --cert="$CERT_DIR/server/keycloak-fullchain.crt" \
        --key="$CERT_DIR/server/keycloak.key" \
        --namespace="$SECURITY_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Keycloak
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm upgrade --install keycloak bitnami/keycloak \
        --namespace "$SECURITY_NAMESPACE" \
        --values - <<EOF
auth:
  adminUser: admin
  existingSecret: keycloak-admin-secret
postgresql:
  enabled: true
  auth:
    existingSecret: keycloak-db-secret
service:
  type: ClusterIP
ingress:
  enabled: true
  hostname: auth.consensus.local
  tls: true
  existingSecret: keycloak-tls
metrics:
  enabled: true
  serviceMonitor:
    enabled: true
EOF
    
    print_success "Keycloak deployed"
}

# Deploy HashiCorp Vault
deploy_vault() {
    print_info "Deploying HashiCorp Vault..."
    
    # Create TLS secret
    kubectl create secret tls vault-tls \
        --cert="$CERT_DIR/server/vault-fullchain.crt" \
        --key="$CERT_DIR/server/vault.key" \
        --namespace="$SECURITY_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Vault
    helm repo add hashicorp https://helm.releases.hashicorp.com
    helm upgrade --install vault hashicorp/vault \
        --namespace "$SECURITY_NAMESPACE" \
        --values - <<EOF
global:
  enabled: true
  tlsDisable: false
injector:
  enabled: true
server:
  ha:
    enabled: true
    replicas: 3
    raft:
      enabled: true
  ingress:
    enabled: true
    hosts:
      - host: vault.consensus.local
    tls:
      - secretName: vault-tls
        hosts:
          - vault.consensus.local
ui:
  enabled: true
EOF
    
    # Initialize Vault if needed
    sleep 30
    if ! kubectl exec -n "$SECURITY_NAMESPACE" vault-0 -- vault status &> /dev/null; then
        print_info "Initializing Vault..."
        kubectl exec -n "$SECURITY_NAMESPACE" vault-0 -- vault operator init \
            -key-shares=5 -key-threshold=3 -format=json > "$SECRETS_DIR/vault-init.json"
    fi
    
    print_success "Vault deployed"
}

# Deploy Falco
deploy_falco() {
    print_info "Deploying Falco..."
    
    helm repo add falcosecurity https://falcosecurity.github.io/charts
    helm upgrade --install falco falcosecurity/falco \
        --namespace "$SECURITY_NAMESPACE" \
        --values - <<EOF
falco:
  grpc:
    enabled: true
  grpcOutput:
    enabled: true
  webserver:
    enabled: true
  rulesFile:
    - /etc/falco/falco_rules.yaml
    - /etc/falco/falco_rules.local.yaml
    - /etc/falco/rules.d
customRules:
  rules-consensus.yaml: |
$(cat "$CONFIG_DIR/falco-rules.yaml" | sed 's/^/    /')
metrics:
  enabled: true
  serviceMonitor:
    enabled: true
EOF
    
    print_success "Falco deployed"
}

# Deploy Trivy
deploy_trivy() {
    print_info "Deploying Trivy..."
    
    helm repo add aqua https://aquasecurity.github.io/helm-charts/
    helm upgrade --install trivy-operator aqua/trivy-operator \
        --namespace "$SECURITY_NAMESPACE" \
        --values - <<EOF
operator:
  vulnerabilityScannerEnabled: true
  configAuditScannerEnabled: true
  rbacAssessmentScannerEnabled: true
  infraAssessmentScannerEnabled: true
  exposedSecretScannerEnabled: true
serviceMonitor:
  enabled: true
EOF
    
    print_success "Trivy deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    print_info "Deploying security monitoring..."
    
    # Deploy Prometheus with security rules
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace "$SECURITY_NAMESPACE" \
        --values - <<EOF
prometheus:
  prometheusSpec:
    serviceMonitorSelectorNilUsesHelmValues: false
    podMonitorSelectorNilUsesHelmValues: false
    ruleSelectorNilUsesHelmValues: false
    retention: 30d
    storageSpec:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 50Gi
grafana:
  enabled: true
  adminPassword: $(cat "$SECRETS_DIR/grafana_admin_password")
  ingress:
    enabled: true
    hosts:
      - grafana.consensus.local
    tls:
      - secretName: grafana-tls
        hosts:
          - grafana.consensus.local
  sidecar:
    dashboards:
      enabled: true
    datasources:
      enabled: true
EOF
    
    print_success "Monitoring deployed"
}

# Configure RBAC
configure_rbac() {
    print_info "Configuring RBAC..."
    
    cat <<EOF | kubectl apply -f -
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: consensus-security-manager
  namespace: $CONSENSUS_NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: consensus-security-manager
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["networking.k8s.io"]
  resources: ["networkpolicies"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["cilium.io"]
  resources: ["ciliumnetworkpolicies"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["security.istio.io"]
  resources: ["authorizationpolicies", "peerauthentications"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: consensus-security-manager
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: consensus-security-manager
subjects:
- kind: ServiceAccount
  name: consensus-security-manager
  namespace: $CONSENSUS_NAMESPACE
EOF
    
    print_success "RBAC configured"
}

# Deploy security policies
deploy_security_policies() {
    print_info "Deploying security policies..."
    
    # Pod Security Standards
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Namespace
metadata:
  name: $CONSENSUS_NAMESPACE
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: latest
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/audit-version: latest
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/warn-version: latest
EOF
    
    # OPA Gatekeeper policies
    if ! kubectl get crd configs.config.gatekeeper.sh &> /dev/null; then
        print_info "Installing OPA Gatekeeper..."
        kubectl apply -f https://raw.githubusercontent.com/open-policy-agent/gatekeeper/v3.14.0/deploy/gatekeeper.yaml
    fi
    
    print_success "Security policies deployed"
}

# Main deployment function
main() {
    print_info "Starting security infrastructure deployment..."
    
    check_prerequisites
    generate_certificates
    generate_secrets
    create_namespaces
    deploy_network_policies
    deploy_keycloak
    deploy_vault
    deploy_falco
    deploy_trivy
    deploy_monitoring
    configure_rbac
    deploy_security_policies
    
    print_success "Security infrastructure deployed successfully!"
    
    print_info "Next steps:"
    echo "1. Configure Keycloak realm at https://auth.consensus.local"
    echo "2. Unseal Vault using keys in $SECRETS_DIR/vault-init.json"
    echo "3. Access Grafana at https://grafana.consensus.local"
    echo "4. Configure SIEM integration with your provider"
    echo "5. Run security validation: ./validate-security.sh"
}

# Run main function
main "$@"