@echo off
setlocal
cd /d "%~dp0"
echo Build legacy five-state design preview only.
echo Output: scenes\00_Scene3_Design_Preview.ttt
echo This does not overwrite apex_mr_scene3.ttt or the archived design input.
python src\build_scene.py
if errorlevel 1 goto done
python src\verify_scene.py --scene scenes\00_Scene3_Design_Preview.ttt
:done
set "SCENE3_EXIT=%ERRORLEVEL%"
pause
exit /b %SCENE3_EXIT%
