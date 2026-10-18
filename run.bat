@echo off
setlocal
title Neural World Model

if /i "%~1"=="--help" goto :help
if /i "%~1"=="app" goto :app
set "NWM_MODE=smoke"
if not "%~1"=="" set "NWM_MODE=%~1"
if /i "%NWM_MODE%"=="smoke" goto :run
if /i "%NWM_MODE%"=="full" goto :run
echo Unknown experiment: %NWM_MODE%
echo Use run.bat smoke or run.bat full.
exit /b 1

:run
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "NWM_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%NWM_PYTHON%" goto :missing_python

echo Running the %NWM_MODE% experiment. Saved training will resume.
echo Results: %~dp0runs\%NWM_MODE%
echo.

"%NWM_PYTHON%" -u -m nwm generate --config "configs\%NWM_MODE%.json"
if errorlevel 1 goto :failed
"%NWM_PYTHON%" -u -m nwm run-experiment --config "configs\%NWM_MODE%.json" --device auto --resume
if errorlevel 1 goto :failed

echo.
echo Experiment complete. Opening the results folder.
start "" "%~dp0runs\%NWM_MODE%"
popd
exit /b 0

:missing_python
echo Project Python was not found at:
echo %NWM_PYTHON%
echo Follow the setup instructions in README.md, then run this launcher again.
goto :failed

:failed
echo.
echo The experiment stopped. See the error above and README.md for setup help.
pause
popd
exit /b 1

:help
echo Neural World Model launcher
echo Usage: run.bat [app^|smoke^|full]
echo Default: smoke. Existing training checkpoints are resumed.
echo Full training can take substantial time. Results open when complete.
exit /b 0

:app
pushd "%~dp0"
if errorlevel 1 exit /b 1
echo Starting the local app at http://127.0.0.1:8000
echo Keep this window open while using the app. Press Ctrl+C to stop.
"%~dp0.venv\Scripts\python.exe" -m nwm serve --config configs\app.json --open-browser
if errorlevel 1 goto :failed
popd
exit /b 0
