@echo off
setlocal
rem Locate the project using this CMD file's own directory.
set "SELF=%~dp0"
pushd "%SELF%.."
set "ROOT=%CD%"
popd

set "PYTHON=%ROOT%\.venv\Scripts\pythonw.exe"
if not exist "%PYTHON%" set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo.
    echo [ERROR] The environment has not been set up yet.
    echo Please double-click SETUP.cmd first, then run this again.
    echo.
    pause
    exit /b 1
)

"%PYTHON%" "%ROOT%\launcher.py"
