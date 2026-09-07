# setup_cpu.ps1
# CPU profile installer for the Road Roller Compaction Analyzer.
# No administrator privileges required. Installs a project-local environment
# using uv and the official PyTorch CPU wheel index.

$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host ("[SETUP] " + $Message) -ForegroundColor Cyan
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
    Write-Step "uv installed: $((uv --version) 2>$null)"
}

function Install-OpenCvGui {
    # EasyOCR pulls opencv-python-headless, but the Road Roller tracker needs the
    # GUI build (cv2.imshow / cv2.selectROI). Remove both OpenCV distributions and
    # install only the GUI build so the two are never left in an unstable state.
    Write-Step "Resolving OpenCV GUI/headless conflict..."
    & uv pip uninstall opencv-python-headless opencv-python *>&1 | Out-Null
    & uv pip install --no-deps "opencv-python==4.10.0.84" *>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install the OpenCV GUI build."
    }
}

Write-Step "Road Roller AI - CPU Setup"
Write-Step "Project root: $ProjectRoot"

try {
    Ensure-Uv

    Write-Step "Creating project-local Python 3.11 environment (.venv)..."
    & uv venv ".venv" --python 3.11 --clear
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the virtual environment."
    }

    Write-Step "Installing PyTorch CPU runtime..."
    & uv pip install -r "requirements\cpu.lock"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install the PyTorch CPU runtime."
    }

    Write-Step "Installing application dependencies..."
    & uv pip install -r "requirements\common.lock"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install application dependencies."
    }

    Install-OpenCvGui

    Write-Step "Verifying environment..."
    & "$ProjectRoot\.venv\Scripts\python.exe" "$ProjectRoot\verify_env.py"
    $verifyCode = $LASTEXITCODE
} catch {
    Write-Host ""
    Write-Host "SETUP FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Review the messages above, then re-run SETUP_CPU.cmd." -ForegroundColor Red
    exit 1
}

if ($verifyCode -eq 0) {
    Write-Host ""
    Write-Host "SETUP SUCCESSFUL" -ForegroundColor Green
    Write-Host "You can now double-click RUN_ROLLER_AI.cmd to start the application."
    Write-Host ""
    $answer = Read-Host "Launch the application now? (Y/N)"
    if ($answer -match "^[Yy]") {
        & "$ProjectRoot\.venv\Scripts\python.exe" "$ProjectRoot\launcher.py"
    }
    exit 0
} else {
    Write-Host ""
    Write-Host "SETUP FAILED: environment verification did not pass." -ForegroundColor Red
    Write-Host "Review the verification output above and re-run SETUP_CPU.cmd." -ForegroundColor Red
    exit 1
}
