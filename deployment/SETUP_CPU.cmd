@echo off
setlocal
set "DEPLOY_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy RemoteSigned -File "%DEPLOY_DIR%setup_cpu.ps1"
echo.
echo Setup finished. See the result above.
pause
