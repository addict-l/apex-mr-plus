@echo off
setlocal
cd /d "%~dp0"
set "APEX_PYTHON=python"
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "APEX_PYTHON=%LocalAppData%\Programs\Python\Python311\python.exe"
if not exist "scenes\10_Normal_Assembly.ttt" (
    echo Normal dynamic source scene is missing: scenes\10_Normal_Assembly.ttt
    echo See normal\README.md for scene construction and requirements.
    pause
    exit /b 1
)
echo APEX MR Plus - full normal-mode replanning and dynamic execution
echo Recomputing candidate IK, ILP, RRT-Connect reference paths and TPG.
echo This takes longer than Run_Normal_Mode.cmd.
echo Results: %CD%\results\normal
"%APEX_PYTHON%" -m normal.pipeline --replan %*
set "APEX_REPLAN_EXIT=%ERRORLEVEL%"
if not "%APEX_REPLAN_EXIT%"=="0" (
    echo.
    echo Run did not pass. Inspect the stage logs and results\normal diagnostics.
) else (
    echo.
    echo Command completed. Check the current summary.json before interpreting the result.
)
pause
exit /b %APEX_REPLAN_EXIT%
