@echo off
rem One-click live camera feed for the OAK-FFC-3P: left mono | RGB (middle) | right mono.
rem Shares the camera with WSL (usbipd), keeps it attached across reboots of the device, and opens view_feed.py.
rem Extra arguments are passed through, e.g.:  run_feed.bat --cams BC
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "USBIPD=C:\Program Files\usbipd-win\usbipd.exe"
set "PATH=C:\Program Files\usbipd-win;%PATH%"

if not exist "%USBIPD%" (
    echo usbipd-win is not installed. Run:  winget install usbipd
    pause & exit /b 1
)

rem Find the camera (Luxonis vendor id 03e7) and its state
set "BUSID=" & set "STATE="
for /f "tokens=1,*" %%a in ('usbipd list ^| findstr /i "03e7:"') do (
    if not defined BUSID (
        set "BUSID=%%a"
        set "LINE=%%b"
    )
)
if not defined BUSID (
    echo OAK camera not found on USB. Plug it in and run this again.
    pause & exit /b 1
)
echo !LINE! | findstr /c:"Not shared" >nul && set "STATE=notshared"
echo !LINE! | findstr /c:"Attached" >nul && set "STATE=attached"
echo Camera on USB bus %BUSID%

if "%STATE%"=="notshared" (
    echo Sharing the camera with WSL - accept the administrator prompt...
    powershell -NoProfile -Command "Start-Process -FilePath '%USBIPD%' -ArgumentList 'bind','--busid','%BUSID%' -Verb RunAs -Wait -WindowStyle Hidden"
)

rem Make sure WSL is up, then keep the camera attached (the device re-plugs itself when it boots)
wsl.exe -d Ubuntu-22.04 -- true
if not "%STATE%"=="attached" (
    start "OAK usbipd auto-attach (leave open)" /min "%USBIPD%" attach --wsl --busid %BUSID% --auto-attach
    timeout /t 4 /nobreak >nul
)

rem The booted camera re-plugs as 03e7:f63b; on a new port or a new USB speed (e.g. first time on USB3) Windows sees it
rem as a new device that is not shared yet. Watch for that for 90 s and share it (one administrator prompt).
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "$u='%USBIPD%'; $end=(Get-Date).AddSeconds(90); while ((Get-Date) -lt $end) { $l = & $u list | Select-String '03e7:f63b'; if ($l -and $l -match 'Not shared') { Start-Process $u -ArgumentList 'bind','--hardware-id','03e7:f63b' -Verb RunAs -Wait -WindowStyle Hidden; break }; Start-Sleep -Milliseconds 400 }"

echo Starting camera feed... (first start takes ~15 s while the camera boots)
".venv\Scripts\python.exe" view_feed.py %*
if errorlevel 1 pause
