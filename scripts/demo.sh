#!/usr/bin/env bash
# Khaslana End-to-End Smoke Test Script (POSIX bash)
# Tests the full stack against the sample repo fixture

set -euo pipefail

echo "=================================================="
echo "   KHASLANA COMPLETE END-TO-END SMOKE TEST        "
echo "=================================================="

echo ""
echo "[1/10] Checking Doctor..."
khaslana doctor

echo ""
echo "[2/10] Checking Config..."
khaslana config show

echo ""
echo "[3/10] Indexing Sample Repo..."
khaslana index tests/fixtures/sample_repo

echo ""
echo "[4/10] Testing Hybrid Search with Explain..."
khaslana search "authenticate user" --explain

echo ""
echo "[5/10] Testing Symbol Resolution (where-used)..."
khaslana where-used AuthService

echo ""
echo "[6/10] Testing Call Graph Export..."
khaslana graph authenticate_user --format mermaid

echo ""
echo "[7/10] Explaining Sample File..."
khaslana explain tests/fixtures/sample_repo/auth.py

echo ""
echo "[8/10] Testing Git Review..."
khaslana review --staged

echo ""
echo "[9/10] Testing Plugin Discovery and Inspection..."
khaslana plugins list
khaslana plugins info db-query

echo ""
echo "[10/10] Testing Daemon Lifecycle..."
khaslana daemon status

echo ""
echo "=================================================="
echo "   ALL END-TO-END CHECKS COMPLETED SUCCESSFULLY!  "
echo "=================================================="
