@echo off
rem Run FoundationStereo on the newest recording and open the comparison video
rem (rectified left | camera's own depth | FoundationStereo depth).
rem Default: ViT-small, 16 iterations, every frame (~0.9 s per frame on the RTX 5050).
rem Examples:  fs_process.bat --model vitl            (best quality, ~1.1-1.6 s per frame)
rem            fs_process.bat --stride 2               (every 2nd frame, twice as fast)
cd /d "%~dp0"
wsl.exe -d Ubuntu-22.04 -- bash "/mnt/c/Users/Ritik/Desktop/oakffc3p/fs_process.sh" %*
if errorlevel 1 (
    echo FoundationStereo processing failed.
    pause & exit /b 1
)
for /f "delims=" %%d in ('dir /b /ad /o-d "recordings"') do (
    for %%f in ("recordings\%%d\compare_*.avi") do start "" "%%f"
    goto :done
)
:done
pause
