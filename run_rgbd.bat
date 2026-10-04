@echo off
rem One-click live RGB + depth for the OAK-FFC-3P, running natively on Windows (no WSL needed; works on USB2).
rem Left: RGB camera. Right: stereo depth aligned to the RGB view. Click to measure, 'o' overlays depth on the RGB.
rem Extra arguments are passed through, e.g.:  run_rgbd.bat --near   or   run_rgbd.bat --fps 25 --max-m 3
cd /d "%~dp0"
echo Starting RGB + depth viewer... (first start takes ~10 s while the camera boots)
".venv\Scripts\python.exe" view_rgbd.py %*
if errorlevel 1 pause
