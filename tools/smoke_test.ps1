# Starts Dataform.exe without a visible window and checks that it comes up.
#
# The app runs with QT_QPA_PLATFORM=offscreen for a few seconds and is then
# stopped. The check fails when the process exits early (a missing DLL or a QML
# file that did not load ends it straight away) or when dataform_debug.log,
# written next to the exe, contains a Qt plugin or QML loading error.
#
# Used by CI and before packaging a release:
#   pwsh tools/smoke_test.ps1 -Exe path\to\Dataform.exe [-Seconds 15]
#
# The exe's folder must be writable: the app creates profiles/ and its log there.

param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [int]$Seconds = 15
)

$ErrorActionPreference = "Stop"
$Exe = (Resolve-Path $Exe).Path
$dir = Split-Path $Exe -Parent
$log = Join-Path $dir "dataform_debug.log"
Remove-Item $log -ErrorAction SilentlyContinue

# Without this plugin Qt cannot start offscreen and puts up a modal error box
# instead of exiting, so check for it first.
if (-not (Test-Path (Join-Path $dir "platforms/qoffscreen.dll"))) {
    Write-Host "::error::platforms/qoffscreen.dll is missing next to the exe; copy it from <Qt>/plugins/platforms."
    exit 1
}

$env:QT_QPA_PLATFORM = "offscreen"
$proc = Start-Process -FilePath $Exe -WorkingDirectory $dir -PassThru
Start-Sleep -Seconds $Seconds

$exitedEarly = $proc.HasExited
if (-not $exitedEarly) {
    Stop-Process -Id $proc.Id -Force
    $proc.WaitForExit()
}

if (-not (Test-Path $log)) {
    Write-Host "::error::No dataform_debug.log was written; the app did not get as far as main()."
    exit 1
}

$lines = Get-Content $log
Write-Host "---- dataform_debug.log ($($lines.Count) lines) ----"
$lines | ForEach-Object { Write-Host $_ }
Write-Host "---- end of log ----"

# Errors that mean a dependency is missing from the folder or QML did not load.
$fatal = @(
    'module ".*" is not installed',
    'is not a type',
    'failed to load component',
    'QQmlApplicationEngine failed',
    'Cannot load library',
    'Could not load the Qt platform plugin',
    'Failed to load platform plugin',
    'Driver not loaded',
    '\[FTL\]'
)
$found = @($lines | Select-String -Pattern $fatal)

$qmlWarnings = @($lines | Select-String -Pattern '\[WRN\].*qrc:/')
if ($qmlWarnings.Count -gt 0) {
    Write-Host "$($qmlWarnings.Count) QML warning(s) in the log (not treated as failures)."
}

if ($exitedEarly) {
    Write-Host "::error::Dataform.exe exited after less than $Seconds s (exit code $($proc.ExitCode))."
    exit 1
}
if ($found.Count -gt 0) {
    Write-Host "::error::Loading errors in the log:"
    $found | ForEach-Object { Write-Host "  $($_.Line)" }
    exit 1
}

Write-Host "Smoke test passed: Dataform.exe ran for $Seconds s with no loading errors."
exit 0
