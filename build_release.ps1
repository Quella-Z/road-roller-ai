# build_release.ps1
# Builds the two end-user distribution packages from this single codebase:
#   RoadRollerAI-v<version>-CPU.zip
#   RoadRollerAI-v<version>-GPU.zip
#
# Both packages contain identical application code. They differ only in the
# root SETUP.cmd (CPU vs GPU installer) and the requirements lock used at
# install time (cpu.lock vs gpu.lock).

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$Version = (Get-Content -LiteralPath "VERSION" -Raw).Trim()

$Dist = Join-Path $ProjectRoot "dist"
if (-not (Test-Path -LiteralPath $Dist)) {
    New-Item -ItemType Directory -Path $Dist | Out-Null
}

$AppFiles = @(
    "track_roller.py",
    "analyze_compaction.py",
    "launcher.py",
    "verify_env.py",
    "VERSION",
    "README.md",
    ".gitignore",
    "RUN_ROLLER_AI.cmd"
)

$AppDirs = @(
    "requirements",
    "deployment"
)

# SAM 2.1 Tiny weights are bundled in the release packages so end users work
# offline. The file is git-ignored (Apache-2.0, re-downloadable via Ultralytics);
# if it is absent at build time it is skipped and Ultralytics downloads it at
# first tracking run instead.
$ModelFiles = @(
    "sam2.1_t.pt"
)

function New-StagingDir([string]$Name) {
    $stage = Join-Path $env:TEMP $Name
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
    New-Item -ItemType Directory -Path $stage | Out-Null
    return $stage
}

function Copy-AppFiles([string]$Stage) {
    foreach ($file in $AppFiles) {
        Copy-Item -LiteralPath (Join-Path $ProjectRoot $file) -Destination $Stage
    }
    foreach ($dir in $AppDirs) {
        Copy-Item -LiteralPath (Join-Path $ProjectRoot $dir) -Destination $Stage -Recurse
    }
    foreach ($file in $ModelFiles) {
        $src = Join-Path $ProjectRoot $file
        if (Test-Path -LiteralPath $src) {
            Copy-Item -LiteralPath $src -Destination $Stage
        } else {
            Write-Host "  NOTE: $file not found; it will download automatically at first run." -ForegroundColor Yellow
        }
    }
}

function Write-SetupCmd([string]$Stage, [string]$Installer) {
    $content = @"
@echo off
setlocal
rem First-time setup for the Road Roller Compaction Analyzer ($Installer profile).
call "%~dp0deployment\SETUP_$Installer.cmd"
"@
    Set-Content -LiteralPath (Join-Path $Stage "SETUP.cmd") -Value $content -Encoding Ascii
}

function Build-Package([string]$Profile) {
    $stageName = "RoadRollerAI_${Profile}_stage_$([Guid]::NewGuid().ToString('N'))"
    $stage = New-StagingDir $stageName

    Write-Host ""
    Write-Host "Building $Profile package..." -ForegroundColor Cyan
    Copy-AppFiles $stage
    Write-SetupCmd $stage $Profile

    $zipName = "RoadRollerAI-$Version-$Profile.zip"
    $zipPath = Join-Path $Dist $zipName

    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }

    Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath
    Remove-Item -LiteralPath $stage -Recurse -Force

    Write-Host "Created: $zipPath" -ForegroundColor Green
    return $zipPath
}

Build-Package "CPU" | Out-Null
Build-Package "GPU" | Out-Null

Write-Host ""
Write-Host "Release packages built in: $Dist" -ForegroundColor Green
