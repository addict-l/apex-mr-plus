@echo off
setlocal
cd /d "%~dp0"
echo Abstract lifecycle baseline only: no robot paths or dynamics.
python src\lifecycle.py
set "SCENE3_EXIT=%ERRORLEVEL%"
pause
exit /b %SCENE3_EXIT%
