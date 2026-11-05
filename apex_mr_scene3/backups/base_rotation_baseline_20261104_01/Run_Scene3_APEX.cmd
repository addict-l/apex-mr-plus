@echo off
setlocal
cd /d "%~dp0"
set "SCENE3_PYTHON=python"
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "SCENE3_PYTHON=%LocalAppData%\Programs\Python\Python311\python.exe"
echo Scene 3 APEX-MR: compatible cache, geometry validation, actual dynamic execution.
echo Results: %CD%\results\apex
"%SCENE3_PYTHON%" -m apex.pipeline --reuse-reference --reuse-tpg %*
set "SCENE3_EXIT=%ERRORLEVEL%"
if not "%SCENE3_EXIT%"=="0" echo Run did not pass. Read results\apex\pipeline_manifest.json and summary.json.
if "%SCENE3_EXIT%"=="0" echo Command completed. Read summary.json to distinguish plan-only from actual acceptance.
pause
exit /b %SCENE3_EXIT%
