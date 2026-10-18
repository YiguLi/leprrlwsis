@echo off
rem ===========================================================================
rem  One-shot runner: four diagnostic stages, sequential, 70k steps each.
rem    S1 fixed wind + fixed price
rem    S2 varying wind + fixed price
rem    S3 varying wind + varying price
rem    E2 1-D uniform derate (lambda=20, noisy pool)
rem
rem  Usage:
rem    run_diag_all.cmd                  all four stages (resume: done stages skipped)
rem    run_diag_all.cmd --stages S1      only S1
rem    run_diag_all.cmd --no-resume      force re-run all
rem    run_diag_all.cmd --smoke          400 steps per stage (self-test)
rem
rem  NOTE: keep this file ASCII-only. cmd.exe parses .cmd in the OEM codepage;
rem        UTF-8 Chinese comments break the parser (garbled lines become commands).
rem ===========================================================================
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"

set "PYEXE=C:\Users\yigul\anaconda3\envs\llmenhancedwsis\python.exe"
if not exist "%PYEXE%" (
  echo [ERROR] python not found: %PYEXE%
  echo         edit PYEXE at the top of this file
  pause
  exit /b 1
)

echo [run] %DATE% %TIME%  starting four diagnostic stages (70k steps each, ~10.5 h)
"%PYEXE%" agentutil\diag_launch.py %*
if errorlevel 1 echo [warn] launcher returned non-zero (see output\diag\logs\)

echo.
echo ============================== status.json ==============================
if exist output\diag\status.json (
  type output\diag\status.json
) else (
  echo (no status.json)
)
echo.
echo All logs: output\diag\logs\   Summary: output\diag\summary.csv
pause
