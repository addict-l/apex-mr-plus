@echo off
setlocal
cd /d "%~dp0"
set "SCENE3_PYTHON=python"
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "SCENE3_PYTHON=%LocalAppData%\Programs\Python\Python311\python.exe"
echo Scene 3 APEX-MR: recompute ILP, reference motions and TPG, then execute.
echo Add --rebuild-scene after editing scene geometry or physical parameters.
"%SCENE3_PYTHON%" -m apex.pipeline --replan %*
set "SCENE3_EXIT=%ERRORLEVEL%"
if not "%SCENE3_EXIT%"=="0" echo Run did not pass. Read results\apex\pipeline_manifest.json and summary.json.
if "%SCENE3_EXIT%"=="0" echo Command completed. Read summary.json to distinguish plan-only from actual acceptance.
pause
exit /b %SCENE3_EXIT%
