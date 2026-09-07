@echo off
setlocal
rem Convenience launcher. Delegates to the canonical script in deployment/.
call "%~dp0deployment\RUN_ROLLER_AI.cmd"
