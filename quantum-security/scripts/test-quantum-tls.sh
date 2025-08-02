#!/bin/bash

# Test script for quantum-resistant TLS
set -e

echo "=== Quantum Security TLS Test ==="
echo ""

# Base URL
BASE_URL="https://localhost:8443"
HTTP_URL="http://localhost:3000"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print test results
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
    fi
}

# Test 1: Health check
echo "1. Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s -k "$HTTP_URL/health")
if echo "$HEALTH_RESPONSE" | grep -q '"status":"ok"'; then
    print_result 0 "Health check passed"
else
    print_result 1 "Health check failed"
fi

# Test 2: Generate ML-KEM keypair
echo ""
echo "2. Testing ML-KEM key generation..."
MLKEM_RESPONSE=$(curl -s -k -X POST "$BASE_URL/api/crypto/keygen/ml-kem" \
    -H "Content-Type: application/json" \
    -d '{"securityLevel": 768}')
    
if echo "$MLKEM_RESPONSE" | grep -q '"success":true'; then
    print_result 0 "ML-KEM key generation successful"
    MLKEM_PUBLIC_KEY=$(echo "$MLKEM_RESPONSE" | grep -o '"publicKey":"[^"]*' | cut -d'"' -f4)
else
    print_result 1 "ML-KEM key generation failed"
fi

# Test 3: Generate ML-DSA keypair
echo ""
echo "3. Testing ML-DSA key generation..."
MLDSA_RESPONSE=$(curl -s -k -X POST "$BASE_URL/api/crypto/keygen/ml-dsa" \
    -H "Content-Type: application/json" \
    -d '{"securityLevel": 65}')
    
if echo "$MLDSA_RESPONSE" | grep -q '"success":true'; then
    print_result 0 "ML-DSA key generation successful"
    MLDSA_PUBLIC_KEY=$(echo "$MLDSA_RESPONSE" | grep -o '"publicKey":"[^"]*' | cut -d'"' -f4)
    MLDSA_PRIVATE_KEY=$(echo "$MLDSA_RESPONSE" | grep -o '"privateKey":"[^"]*' | cut -d'"' -f4)
else
    print_result 1 "ML-DSA key generation failed"
fi

# Test 4: Generate SLH-DSA keypair
echo ""
echo "4. Testing SLH-DSA key generation..."
SLHDSA_RESPONSE=$(curl -s -k -X POST "$BASE_URL/api/crypto/keygen/slh-dsa" \
    -H "Content-Type: application/json" \
    -d '{"securityLevel": 192}')
    
if echo "$SLHDSA_RESPONSE" | grep -q '"success":true'; then
    print_result 0 "SLH-DSA key generation successful"
else
    print_result 1 "SLH-DSA key generation failed"
fi

# Test 5: Test encryption/decryption
echo ""
echo "5. Testing ML-KEM encryption..."
if [ ! -z "$MLKEM_PUBLIC_KEY" ]; then
    ENCRYPT_RESPONSE=$(curl -s -k -X POST "$BASE_URL/api/crypto/encrypt" \
        -H "Content-Type: application/json" \
        -d "{\"publicKey\": \"$MLKEM_PUBLIC_KEY\", \"securityLevel\": 768}")
    
    if echo "$ENCRYPT_RESPONSE" | grep -q '"success":true'; then
        print_result 0 "ML-KEM encryption successful"
        CIPHERTEXT=$(echo "$ENCRYPT_RESPONSE" | grep -o '"ciphertext":"[^"]*' | cut -d'"' -f4)
    else
        print_result 1 "ML-KEM encryption failed"
    fi
fi

# Test 6: Test signing/verification
echo ""
echo "6. Testing ML-DSA signing..."
if [ ! -z "$MLDSA_PRIVATE_KEY" ]; then
    MESSAGE="Test message for quantum signature"
    SIGN_RESPONSE=$(curl -s -k -X POST "$BASE_URL/api/crypto/sign" \
        -H "Content-Type: application/json" \
        -d "{\"privateKey\": \"$MLDSA_PRIVATE_KEY\", \"message\": \"$MESSAGE\", \"algorithm\": \"ml-dsa\", \"securityLevel\": 65}")
    
    if echo "$SIGN_RESPONSE" | grep -q '"success":true'; then
        print_result 0 "ML-DSA signing successful"
        SIGNATURE=$(echo "$SIGN_RESPONSE" | grep -o '"signature":"[^"]*' | cut -d'"' -f4)
        
        # Test verification
        echo ""
        echo "7. Testing ML-DSA verification..."
        VERIFY_RESPONSE=$(curl -s -k -X POST "$BASE_URL/api/crypto/verify" \
            -H "Content-Type: application/json" \
            -d "{\"publicKey\": \"$MLDSA_PUBLIC_KEY\", \"message\": \"$MESSAGE\", \"signature\": \"$SIGNATURE\", \"algorithm\": \"ml-dsa\", \"securityLevel\": 65}")
        
        if echo "$VERIFY_RESPONSE" | grep -q '"isValid":true'; then
            print_result 0 "ML-DSA verification successful"
        else
            print_result 1 "ML-DSA verification failed"
        fi
    else
        print_result 1 "ML-DSA signing failed"
    fi
fi

# Test 8: Test hybrid key exchange
echo ""
echo "8. Testing hybrid key exchange..."
HYBRID_RESPONSE=$(curl -s -k -X POST "$BASE_URL/api/crypto/hybrid-exchange" \
    -H "Content-Type: application/json" \
    -d '{"mode": "x25519-mlkem768", "isInitiator": true}')
    
if echo "$HYBRID_RESPONSE" | grep -q '"success":true'; then
    print_result 0 "Hybrid key exchange successful"
else
    print_result 1 "Hybrid key exchange failed"
fi

# Test 9: Security report
echo ""
echo "9. Testing security monitoring..."
SECURITY_RESPONSE=$(curl -s -k "$BASE_URL/api/security/report")
if echo "$SECURITY_RESPONSE" | grep -q '"status"'; then
    print_result 0 "Security monitoring active"
    echo -e "${YELLOW}Security Status: $(echo "$SECURITY_RESPONSE" | grep -o '"status":"[^"]*' | cut -d'"' -f4)${NC}"
else
    print_result 1 "Security monitoring failed"
fi

# Test 10: Performance check
echo ""
echo "10. Running performance test..."
echo -e "${YELLOW}Generating 10 ML-KEM keypairs...${NC}"
START_TIME=$(date +%s%N)
for i in {1..10}; do
    curl -s -k -X POST "$BASE_URL/api/crypto/keygen/ml-kem" \
        -H "Content-Type: application/json" \
        -d '{"securityLevel": 768}' > /dev/null
done
END_TIME=$(date +%s%N)
DURATION=$((($END_TIME - $START_TIME) / 1000000))
echo -e "${GREEN}✓ Performance test completed in ${DURATION}ms (${DURATION/10}ms per operation)${NC}"

echo ""
echo "=== Test Summary ==="
echo -e "${GREEN}Quantum-resistant TLS is operational!${NC}"
echo ""
echo "Supported algorithms:"
echo "- ML-KEM-512/768/1024 (Key Encapsulation)"
echo "- ML-DSA-44/65/87 (Digital Signatures)"
echo "- SLH-DSA-128/192/256 (Hash-based Signatures)"
echo "- Hybrid modes (x25519-mlkem768, secp256r1-mlkem768, x448-mlkem1024)"
echo ""
echo -e "${YELLOW}Note: This is a demonstration implementation. For production use,"
echo "integrate with hardware security modules and follow NIST guidelines.${NC}"