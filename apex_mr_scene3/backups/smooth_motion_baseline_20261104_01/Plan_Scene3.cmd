@echo off
setlocal
cd /d "%~dp0"
echo Formal Scene 3 planning: qualified ILP, RRT-Connect, multimodal TPG.
echo Plan-only: no actual execution or main-scene promotion.
python -m apex.pipeline --replan --plan-only %*
set "SCENE3_EXIT=%ERRORLEVEL%"
pause
exit /b %SCENE3_EXIT%
