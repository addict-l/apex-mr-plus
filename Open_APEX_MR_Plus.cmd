@echo off
setlocal
cd /d "%~dp0"
if exist "%LocalAppData%\Programs\Python\Python311\pythonw.exe" (
    start "" "%LocalAppData%\Programs\Python\Python311\pythonw.exe" "%~dp0src\open_scene.py"
) else (
    start "" pythonw "%~dp0src\open_scene.py"
)
endlocal
