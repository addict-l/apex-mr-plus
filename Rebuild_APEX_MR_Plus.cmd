@echo off
setlocal
cd /d "%~dp0"
python src\make_assets.py
if errorlevel 1 exit /b 1
python src\build_scene.py
pause
endlocal
