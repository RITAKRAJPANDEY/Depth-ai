@echo off
rem FoundationStereo RGB-D processing (WSL, GPU) of a recording from record_rgbd.bat, then opens the preview.
rem Default is best quality: ViT-L, 32 iterations, left-right check, full 1280x800 (~13 s per frame on the RTX 5050).
rem   process_rgbd.bat                               newest recording
rem   process_rgbd.bat 20261003_213551               a specific recording (folder name in recordings\)
rem   process_rgbd.bat 20261003_213551 --resume      continue an interrupted run (keeps the finished frames)
rem Extra arguments go to rgbd_process.py, e.g.:  --stride 4  (every 4th frame)   --model vits --iters 16 --no-lr
rem Keep this window open while it runs - closing it stops the processing (use --resume to continue).
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "REC="
set "FIRST=%~1"
if defined FIRST if not "!FIRST:~0,1!"=="-" if exist "recordings\%~1\frames.jsonl" (
    set "REC=%~1"
    shift
)
if not defined REC (
    for /f "delims=" %%d in ('dir /b /ad /o-d "recordings"') do (
        if not defined REC if exist "recordings\%%d\frames.jsonl" set "REC=%%d"
    )
)
if not defined REC (
    echo No recording from record_rgbd.bat found in recordings\
    pause & exit /b 1
)
set "ARGS="
:collect
if "%~1"=="" goto run
set "ARGS=!ARGS! %1"
shift
goto collect
:run
echo Processing recordings\%REC% %ARGS%
echo Keep this window open - closing it stops the processing (continue later with: process_rgbd.bat %REC% --resume)
rem keep Windows awake until the processing ends (standby pauses the GPU work in WSL)
start "" /b powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%~dp0keep_awake.ps1"
wsl.exe -d Ubuntu-22.04 -- bash -lc "cd /mnt/c/Users/Ritik/Desktop/oakffc3p && ~/oakffc3p/.venv-fs/bin/python -u rgbd_process.py recordings/%REC% %ARGS% 2>&1 | grep --line-buffered -vE 'Warning|warnings.warn|autocast'"
if errorlevel 1 (
    echo Processing failed.
    pause & exit /b 1
)
start "" "recordings\%REC%\processed\preview.avi"
pause
