#!/bin/bash
# Run all media server integration tests

echo "==================================="
echo "Media Server Integration Test Suite"
echo "==================================="
echo ""

# Check if services are running
echo "Checking Docker services..."
docker-compose ps

echo ""
echo "Press Enter to continue with tests..."
read

# Run API connectivity tests
echo ""
echo "=== Running API Connectivity Tests ==="
bash /Users/morlock/fun/newmedia/TEST_REPORTS/api-integration-tests.sh

echo ""
echo "Press Enter to continue..."
read

# Run Prowlarr integration test
echo ""
echo "=== Running Prowlarr Integration Test ==="
python3 /Users/morlock/fun/newmedia/TEST_REPORTS/prowlarr-integration-test.py

echo ""
echo "Press Enter to continue..."
read

# Run download client tests
echo ""
echo "=== Running Download Client Tests ==="
python3 /Users/morlock/fun/newmedia/TEST_REPORTS/download-client-integration-test.py

echo ""
echo "Press Enter to continue..."
read

# Run media server tests
echo ""
echo "=== Running Media Server Tests ==="
python3 /Users/morlock/fun/newmedia/TEST_REPORTS/media-server-integration-test.py

echo ""
echo "Press Enter to continue..."
read

# Run comprehensive integration test
echo ""
echo "=== Running Comprehensive Integration Test ==="
python3 /Users/morlock/fun/newmedia/TEST_REPORTS/integration-test-suite.py

echo ""
echo "=== All Tests Complete ==="
echo ""
echo "Generated files:"
echo "- Integration dashboard: /Users/morlock/fun/newmedia/TEST_REPORTS/integration-dashboard.html"
echo "- Test reports: /Users/morlock/fun/newmedia/TEST_REPORTS/integration_test_report_*.md"
echo "- Configuration scripts: /Users/morlock/fun/newmedia/TEST_REPORTS/*.sh"
echo ""
echo "Next steps:"
echo "1. Review the test results"
echo "2. Configure services that need API keys"
echo "3. Set up integrations following the guides"
echo "4. Test the complete media workflow"