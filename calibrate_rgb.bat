@echo off
rem Recalibrate the OAK-FFC-3P: left/right OV9282 (7.5 cm baseline) + IMX577 RGB in the middle.
rem   1. Capture: hold the charuco board (charuco_11_8.pdf) in each shown pose, it auto-captures. Press "s" to stop early.
rem   2. Process: computes intrinsics + left->right and right->rgb extrinsics, saves depthai-main\resources\calib_rgb.json.
rem   3. Review the SUMMARY, then flash with:  calibrate_rgb.bat flash
rem Square / marker size in cm of the PRINTED board - measure yours and edit if it differs.
set "SQUARE=3.8"
set "MARKER=2.85"
cd /d "%~dp0depthai-main"
set "PY=..\.venv\Scripts\python.exe"

if /i "%~1"=="flash" (
    "%PY%" process_calib.py -s %SQUARE% -ms %MARKER% -dst dataset_rgb -o resources\calib_rgb.json --flash
    pause & exit /b
)
if /i not "%~1"=="process" (
    echo y| "%PY%" calibrate.py -s %SQUARE% -ms %MARKER% -nx 11 -ny 8 -brd OAK-FFC-3P-OV9282-IMX577.json -dst dataset_rgb -m capture -rgbs 2028x1520 -ebp
    if errorlevel 1 ( pause & exit /b 1 )
)
"%PY%" process_calib.py -s %SQUARE% -ms %MARKER% -dst dataset_rgb -o resources\calib_rgb.json
pause
