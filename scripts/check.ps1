param(
    [ValidateSet('fast', 'verify', 'release', 'legacy', 'doctor', 'sanitize', 'ubsan', 'network-parity', 'network-api', 'network-health', 'network-performance', 'network-fidelity')]
    [string]$Profile = 'fast'
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    & docker compose exec -T dev python3 scripts/local_ci.py $Profile
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
