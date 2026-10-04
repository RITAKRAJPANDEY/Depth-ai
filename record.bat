@echo off
rem Record a stereo clip for FoundationStereo: lossless rectified left/right PNGs at 30 fps + the camera's own depth.
rem Usage:  record.bat            (10 seconds)
rem         record.bat 30         (30 seconds)
rem The live window shows what is being recorded and closes when the recording is done.
rem Then run fs_process.bat to compute FoundationStereo depth for it.
set "SECONDS=%~1"
if "%SECONDS%"=="" set "SECONDS=10"
echo Recording %SECONDS% seconds once the camera starts...
call "%~dp0run_depth.bat" --record-seconds %SECONDS%
