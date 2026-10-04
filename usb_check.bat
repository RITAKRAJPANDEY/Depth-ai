@echo off
rem Shows the speed the OAK camera (03e7) is connected at. "Super(5G) = USB3" is what you want.
rem Run it while the camera is in use (e.g. with run_depth.bat open), since it only switches to USB3 after booting.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0usb_check.ps1"
pause
