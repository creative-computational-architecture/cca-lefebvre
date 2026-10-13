param(
    [string]$Package = $PSScriptRoot,
    [string]$ScanId
)

$ErrorActionPreference = 'Stop'
$packageRoot = (Resolve-Path -LiteralPath $Package).Path
$manifest = Get-Content -LiteralPath (Join-Path $packageRoot 'manifest.json') -Raw | ConvertFrom-Json
if ($manifest.state -ne 'E57_VERIFIED_RCP_PENDING') {
    throw 'E57 package is not complete and verified.'
}
$decap = Join-Path $env:ProgramFiles 'Autodesk/Autodesk ReCap/decap.exe'
if (-not (Test-Path -LiteralPath $decap -PathType Leaf)) { throw 'Autodesk ReCap is not installed.' }
# Existing desktop licence only. No paid API, credentials, activation or trial.
$licenseOutput = & $decap --checkLicense 2>&1
$licenseOutput | ForEach-Object { Write-Output $_ }
if ($LASTEXITCODE -ne 0 -or ($licenseOutput -join "`n") -match 'FAILED|Licensing Error') {
    throw 'The existing ReCap license is not active. Check it in ReCap. The E57 package is preserved.'
}
$scans = @($manifest.scans)
if ($ScanId) { $scans = @($scans | Where-Object scan_id -eq $ScanId) }
if ($scans.Count -eq 0) { throw 'No matching scans.' }
foreach ($scan in $scans) {
    if ($scan.scan_id -notmatch '^[A-Za-z0-9_-]+$') { throw 'Unsafe scan id.' }
    $observed = @($scan.outputs | Where-Object kind -eq 'observed')
    if ($observed.Count -ne 1) { throw "No observed E57 for $($scan.scan_id)." }
    if ([IO.Path]::GetFileName($observed[0].file) -ne $observed[0].file) { throw 'Unsafe E57 path.' }
    $source = Join-Path (Join-Path $packageRoot $scan.scan_id) $observed[0].file
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $observed[0].sha256) { throw 'E57 hash mismatch.' }
    $output = Join-Path (Join-Path $packageRoot 'RCP') $scan.scan_id
    if (Test-Path -LiteralPath $output) { throw "Output exists; not overwritten: $output" }
    New-Item -ItemType Directory -Path $output | Out-Null
    # Separate local projects, metre, no range clipping, no unification.
    & $decap --importWithLicense $output $scan.scan_id --inputUnitType 2 --decimation 0 $source
    if ($LASTEXITCODE -ne 0) { throw "ReCap indexing failed: $output" }
    $rcp = Get-ChildItem -LiteralPath $output -Filter '*.rcp' -File
    $rcs = Get-ChildItem -LiteralPath $output -Filter '*.rcs' -File -Recurse
    if (-not $rcp -or -not $rcs) { throw "ReCap did not produce RCP and RCS: $output" }
    Write-Output "RCP/RCS files exist: $output -- inspect in ReCap/Revit; registration not verified."
}
