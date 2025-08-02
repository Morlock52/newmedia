#!/bin/bash

# Generate quantum-resistant certificates for testing
# Note: In production, use proper CA-signed certificates

set -e

CERT_DIR="./nginx/ssl"
mkdir -p "$CERT_DIR"

# Generate classical RSA certificate for initial TLS handshake
# (Required until browsers support quantum algorithms natively)
echo "Generating classical RSA certificate for TLS..."
openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
    -keyout "$CERT_DIR/key.pem" \
    -out "$CERT_DIR/cert.pem" \
    -subj "/C=US/ST=State/L=City/O=Quantum Security/CN=quantum-server.local" \
    -addext "subjectAltName = DNS:localhost,DNS:quantum-server.local,IP:127.0.0.1"

# Generate chain certificate
cp "$CERT_DIR/cert.pem" "$CERT_DIR/chain.pem"

# Set appropriate permissions
chmod 600 "$CERT_DIR/key.pem"
chmod 644 "$CERT_DIR/cert.pem"
chmod 644 "$CERT_DIR/chain.pem"

echo "Classical certificates generated successfully!"
echo ""
echo "Note: The quantum-resistant certificates are generated"
echo "programmatically by the application using ML-DSA and SLH-DSA"
echo "algorithms. These classical certs are only for the initial"
echo "TLS handshake until browsers support quantum algorithms."
echo ""
echo "To trust the certificate locally:"
echo "- macOS: sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain $CERT_DIR/cert.pem"
echo "- Linux: sudo cp $CERT_DIR/cert.pem /usr/local/share/ca-certificates/quantum-server.crt && sudo update-ca-certificates"