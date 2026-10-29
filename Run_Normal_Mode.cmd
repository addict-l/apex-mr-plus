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
echo APEX MR Plus - normal dynamic execution
echo Reusing compatible ILP/reference and TPG caches when available.
echo Nominal geometry validation and actual dynamics execution still run.
echo Results: %CD%\results\normal
"%APEX_PYTHON%" -m normal.pipeline --reuse-reference --reuse-tpg %*
set "APEX_RUN_EXIT=%ERRORLEVEL%"
if not "%APEX_RUN_EXIT%"=="0" (
    echo.
    echo Run did not pass. Check results\normal\execution.json and summary.json if present.
) else (
    echo.
    echo Command completed. Check the current summary.json before interpreting the result.
)
pause
exit /b %APEX_RUN_EXIT%
