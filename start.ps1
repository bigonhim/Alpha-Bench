param(
    [switch]$NoBrowser,
    [int]$Port = 8765
)
# Alpha Foundry launcher: installs/updates dependencies, builds the UI when needed,
# compiles the simulation engine once, then serves the app at http://127.0.0.1:$Port
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Need($cmd, $hint) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "Missing '$cmd'. $hint" -ForegroundColor Red
        exit 1
    }
}

Need "uv" "Install uv from https://docs.astral.sh/uv/ (or run: pip install uv)."

Write-Host "[1/4] Python environment" -ForegroundColor Cyan
Push-Location (Join-Path $root "backend")
uv sync --python 3.13 --quiet
if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Host "uv sync failed" -ForegroundColor Red; exit 1 }
Pop-Location

Write-Host "[2/4] Web UI" -ForegroundColor Cyan
$dist = Join-Path $root "frontend\dist\index.html"
$needBuild = -not (Test-Path $dist)
if (-not $needBuild) {
    $distTime = (Get-Item $dist).LastWriteTime
    $newer = Get-ChildItem (Join-Path $root "frontend\src") -Recurse -File | Where-Object { $_.LastWriteTime -gt $distTime } | Select-Object -First 1
    if ($newer) { $needBuild = $true }
}
if ($needBuild) {
    Need "npm" "Install Node.js 20+ from https://nodejs.org/."
    Push-Location (Join-Path $root "frontend")
    if (-not (Test-Path "node_modules")) { npm install --no-audit --no-fund }
    npm run build
    if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Host "UI build failed" -ForegroundColor Red; exit 1 }
    Pop-Location
} else {
    Write-Host "      up to date"
}

Write-Host "[3/4] Simulation engine" -ForegroundColor Cyan
$marker = Join-Path $root "runtime\.engine-warm"
if (-not (Test-Path $marker)) {
    Write-Host "      first run: compiling numba kernels (one-time, 1-3 minutes)..."
    Push-Location (Join-Path $root "backend")
    uv run alphafoundry warmup
    Pop-Location
    New-Item -ItemType Directory -Force (Join-Path $root "runtime") | Out-Null
    Set-Content -Path $marker -Value (Get-Date -Format o)
} else {
    Write-Host "      ready"
}

Write-Host "[4/4] Starting Alpha Foundry on http://127.0.0.1:$Port (Ctrl+C to stop)" -ForegroundColor Green
Push-Location (Join-Path $root "backend")
$serveArgs = @("run", "alphafoundry", "serve", "--port", "$Port")
if (-not $NoBrowser) { $serveArgs += "--open" }
uv @serveArgs
Pop-Location
