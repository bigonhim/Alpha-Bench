@echo off
rem Double-click to launch Alpha Foundry (runs start.ps1 without changing your execution policy).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if errorlevel 1 pause
