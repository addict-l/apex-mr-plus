@echo off
setlocal
cd /d "%~dp0"
python src\open_scene.py --preview
if errorlevel 1 pause
