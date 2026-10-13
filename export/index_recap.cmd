@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0INDEX-RECAP.ps1" -Package "%~dp0."
pause
