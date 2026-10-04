@echo off
rem Camera-IMU calibration: guided capture, then solve (rotation IMU->camera, time offset, gyro/accel bias, accel scale,
rem IMU position). Show charuco_11_8 (3.8 cm squares) full-screen on the monitor before starting.
rem   imu_calibrate.bat                      capture + solve
rem   imu_calibrate.bat flash                solve the newest capture again and write the extrinsics to the camera
rem   imu_calibrate.bat flash 2.0 -1.5 0.8   same, with the IMU position measured by hand (cm, left camera frame)
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if /i "%~1"=="flash" (
    if "%~4"=="" ( "%PY%" imu_calibrate.py solve --flash ) else ( "%PY%" imu_calibrate.py solve --flash --lever-cm %2 %3 %4 )
    pause & exit /b
)
"%PY%" imu_calibrate.py capture
if errorlevel 1 ( pause & exit /b 1 )
"%PY%" imu_calibrate.py solve
pause
