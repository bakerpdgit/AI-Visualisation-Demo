@echo off
rem Starts the Cat or Alligator demo on this PC (works offline).
rem Leave the black window open while the demo is running.
title Cat or Alligator demo
cd /d "%~dp0"

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 7) else 1)" >nul 2>nul
if %errorlevel%==0 (
  python tools\serve.py 8000
  goto :end
)
py -3 -c "import sys" >nul 2>nul
if %errorlevel%==0 (
  py -3 tools\serve.py 8000
  goto :end
)
echo Python not found - using the built-in Windows web server instead.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\serve.ps1" -Port 8000

:end
pause
