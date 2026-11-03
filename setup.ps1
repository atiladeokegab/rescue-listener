# Windows PowerShell 5.1: one uv env per piece (both pinned to Python 3.14 by .python-version; uv downloads it).
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
foreach ($d in 'biocomp-spine', 'rescue-data') {
    Write-Host "== uv sync $d"
    Push-Location $d
    try { uv sync; if ($LASTEXITCODE -ne 0) { throw "uv sync failed in $d" } } finally { Pop-Location }
}
Write-Host "== setup OK"
