@echo off
setlocal
cd /d "%~dp0"
echo Build Scene 3 dynamic source only: scenes\10_Scene3_Dynamic.ttt
echo The main delivered scene will not be overwritten.
echo Use Plan_Scene3.cmd to plan, or Run_Scene3_APEX.cmd to execute and audit.
python -m apex.scene
set "SCENE3_EXIT=%ERRORLEVEL%"
pause
exit /b %SCENE3_EXIT%
