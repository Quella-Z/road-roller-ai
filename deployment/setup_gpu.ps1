# setup_gpu.ps1
# NVIDIA GPU profile installer for the Road Roller Compaction Analyzer.
# No administrator privileges required. Installs a project-local environment
# using uv and the official PyTorch CUDA 12.1 wheel index.
#
# The installer NEVER modifies, downgrades, bypasses, or installs NVIDIA system
# drivers. It only verifies that a compatible driver is already present and
# otherwise stops cleanly.

$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$MinDriver = @(527, 41)  # CUDA 12.1 on Windows requires NVIDIA driver >= 527.41

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host ("[SETUP] " + $Message) -ForegroundColor Cyan
}

function Get-VersionParts([string]$Version) {
    $parts = ($Version -split "\." | ForEach-Object {
        if ($_ -match "^\d+$") { [int]$_ } else { 0 }
    })
    return $parts
}

function Compare-Version([string]$A, [string]$B) {
    $pa = Get-VersionParts $A
    $pb = Get-VersionParts $B
    $len = [Math]::Max($pa.Count, $pb.Count)
    for ($i = 0; $i -lt $len; $i++) {
        $va = if ($i -lt $pa.Count) { $pa[$i] } else { 0 }
        $vb = if ($i -lt $pb.Count) { $pb[$i] } else { 0 }
        if ($va -gt $vb) { return 1 }
        if ($va -lt $vb) { return -1 }
    }
    return 0
}

function Ensure-Uv {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Step "uv found: $((uv --version) 2>$null)"
        return
    }
    Write-Step "uv not found. Installing via the official installer (no admin required)..."
    try {
        & powershell -NoProfile -ExecutionPolicy RemoteSigned -Command `
            "irm https://astral.sh/uv/install.ps1 | iex"
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    } catch {
        throw "Failed to install uv. Please install uv manually from https://docs.astral.sh/uv and re-run setup."
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv is still not available after install. Re-open the terminal and re-run setup."
    }
}

function Install-OpenCvGui {
    Write-Step "Resolving OpenCV GUI/headless conflict..."
    & uv pip uninstall opencv-python-headless opencv-python *>&1 | Out-Null
    & uv pip install --no-deps "opencv-python==4.10.0.84" *>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install the OpenCV GUI build."
    }
}

Write-Step "Road Roller AI - GPU Setup"
Write-Step "Project root: $ProjectRoot"

# --- GPU / driver compatibility checks --------------------------------------
Write-Step "Checking NVIDIA GPU and driver..."

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    Write-Host "nvidia-smi was not found." -ForegroundColor Red
    Write-Host "No NVIDIA GPU/driver detected on this machine." -ForegroundColor Red
    Write-Host "Use the CPU package (RoadRollerAI-v1.0.0-CPU.zip) or contact IT." -ForegroundColor Yellow
    exit 1
}

$driverVersion = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
if (-not $driverVersion) {
    Write-Host "nvidia-smi ran but returned no driver version." -ForegroundColor Red
    Write-Host "Use the CPU package or contact IT." -ForegroundColor Yellow
    exit 1
}

$driverVersion = $driverVersion.Trim()
Write-Step "Detected NVIDIA driver version: $driverVersion"

$cmp = Compare-Version $driverVersion ($MinDriver -join ".")
if ($cmp -lt 0) {
    Write-Host "Driver $driverVersion is below the required $($MinDriver -join '.') for CUDA 12.1." -ForegroundColor Red
    Write-Host "This installer will NOT modify system drivers." -ForegroundColor Red
    Write-Host "Please update the NVIDIA driver via IT, or use the CPU package." -ForegroundColor Yellow
    exit 1
}
Write-Step "Driver $driverVersion meets the CUDA 12.1 requirement."

try {
    Ensure-Uv

    Write-Step "Creating project-local Python 3.11 environment (.venv)..."
    & uv venv ".venv" --python 3.11 --clear
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the virtual environment."
    }

    Write-Step "Installing PyTorch CUDA 12.1 runtime..."
    & uv pip install -r "requirements\gpu.lock"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install the PyTorch CUDA 12.1 runtime."
    }

    Write-Step "Installing application dependencies..."
    & uv pip install -r "requirements\common.lock"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install application dependencies."
    }

    Install-OpenCvGui

    Write-Step "Verifying environment (CUDA required)..."
    & "$ProjectRoot\.venv\Scripts\python.exe" "$ProjectRoot\verify_env.py" --require-cuda
    $verifyCode = $LASTEXITCODE
} catch {
    Write-Host ""
    Write-Host "SETUP FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Review the messages above, then re-run SETUP_GPU.cmd." -ForegroundColor Red
    exit 1
}

if ($verifyCode -eq 0) {
    $deviceName = & "$ProjectRoot\.venv\Scripts\python.exe" -c "import torch; print(torch.cuda.get_device_name(0))" 2>$null
    Write-Host ""
    Write-Host "SETUP SUCCESSFUL" -ForegroundColor Green
    Write-Host "GPU device: $deviceName" -ForegroundColor Green
    Write-Host "You can now double-click RUN_ROLLER_AI.cmd to start the application."
    Write-Host ""
    $answer = Read-Host "Launch the application now? (Y/N)"
    if ($answer -match "^[Yy]") {
        & "$ProjectRoot\.venv\Scripts\python.exe" "$ProjectRoot\launcher.py"
    }
    exit 0
} else {
    Write-Host ""
    Write-Host "SETUP FAILED: CUDA is not available after installation." -ForegroundColor Red
    Write-Host "This installer will not silently fall back to CPU." -ForegroundColor Red
    Write-Host "Check the NVIDIA driver, then re-run SETUP_GPU.cmd, or use the CPU package." -ForegroundColor Yellow
    exit 1
}
