@echo off
rem Record a synchronized RGB + full-resolution stereo + IMU dataset (native Windows, USB2 is fine).
rem Usage:  record_rgbd.bat              (until you press q in the preview window)
rem         record_rgbd.bat 20           (20 seconds)
rem Then run process_rgbd.bat to compute FoundationStereo RGB-D for the newest recording.
cd /d "%~dp0"
set "ARGS="
if not "%~1"=="" set "ARGS=--seconds %~1"
".venv\Scripts\python.exe" rgbd_record.py %ARGS%
if errorlevel 1 pause
