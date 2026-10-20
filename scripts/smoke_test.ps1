# Khaslana End-to-End Smoke Test Script (PowerShell)
# Tests the full stack against the sample repo fixture

$ErrorActionPreference = "Stop"

$venvKhaslana = Join-Path $PSScriptRoot "..\.venv\Scripts\khaslana.exe"
if (Test-Path $venvKhaslana) {
    $KhaslanaCmd = (Resolve-Path $venvKhaslana).Path
} else {
    $KhaslanaCmd = "khaslana"
}

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   KHASLANA COMPLETE END-TO-END SMOKE TEST       " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. System Doctor
Write-Host "`n[1/10] Checking Doctor..." -ForegroundColor Yellow
& $KhaslanaCmd doctor

# 2. Config Inspection
Write-Host "`n[2/10] Checking Config..." -ForegroundColor Yellow
& $KhaslanaCmd config show

# 3. Repository Indexing
Write-Host "`n[3/10] Indexing Sample Repo..." -ForegroundColor Yellow
& $KhaslanaCmd index tests/fixtures/sample_repo

# 4. Hybrid Code Search
Write-Host "`n[4/10] Testing Hybrid Search with Explain..." -ForegroundColor Yellow
& $KhaslanaCmd search "authenticate user" --explain

# 5. Code Intelligence: Symbol Cross-Referencing
Write-Host "`n[5/10] Testing Symbol Resolution (where-used)..." -ForegroundColor Yellow
& $KhaslanaCmd where-used AuthService

# 6. Call Graph Generation
Write-Host "`n[6/10] Testing Call Graph Export..." -ForegroundColor Yellow
& $KhaslanaCmd graph authenticate_user --format mermaid

# 7. Code Explanation
Write-Host "`n[7/10] Explaining Sample File..." -ForegroundColor Yellow
& $KhaslanaCmd explain tests/fixtures/sample_repo/auth.py

# 8. Git Diff & Review
Write-Host "`n[8/10] Testing Git Review..." -ForegroundColor Yellow
& $KhaslanaCmd review --staged

# 9. Plugin System
Write-Host "`n[9/10] Testing Plugin Discovery and Inspection..." -ForegroundColor Yellow
& $KhaslanaCmd plugins list
& $KhaslanaCmd plugins info db-query

# 10. Background Indexing Daemon Status
Write-Host "`n[10/10] Testing Daemon Lifecycle..." -ForegroundColor Yellow
& $KhaslanaCmd daemon status

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "   ALL END-TO-END CHECKS COMPLETED SUCCESSFULLY!  " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
