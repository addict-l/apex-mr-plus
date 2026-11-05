@echo off
setlocal
cd /d "%~dp0"
python src\open_scene.py
if errorlevel 1 pause
