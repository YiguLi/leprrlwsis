@echo off
rem ===========================================================================
rem  一条命令顺序跑完四段诊断实验（S1 恒风恒价 -> S2 变风恒价 -> S3 变风变价 -> E2 一维降载）
rem  用法:
rem      run_diag_all.cmd                四段顺序，各 70k 步（可续跑：已完成的段自动跳过）
rem      run_diag_all.cmd --stages S1    只跑 S1
rem      run_diag_all.cmd --no-resume    全部重跑
rem      run_diag_all.cmd --smoke        每段 400 步自检
rem  产物: output/DIAG-<段>/AIFFLOR-run*/ 、output/diag/{logs,status.json,summary.csv}
rem ===========================================================================
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

set "PYEXE=C:\Users\yigul\anaconda3\envs\llmenhancedwsis\python.exe"
if not exist "%PYEXE%" (
  echo [ERROR] 找不到解释器: %PYEXE%
  echo         请改本文件里的 PYEXE 为可用的 python.exe
  pause
  exit /b 1
)

echo [run] %DATE% %TIME%  开始四段诊断实验（每段 70k 步，预计 ~10.5 h）
"%PYEXE%" agentutil\diag_launch.py %*

echo.
echo ============================== summary.csv ==============================
if exist output\diag\summary.csv (
  "%PYEXE%" -c "import pandas as pd;df=pd.read_csv(r'output/diag/summary.csv');cols=[c for c in ['run','task','eta','pattern_err','viol_pct','a_std_step','r2_dua','regret_pct','s_mean','s_star_mean'] if c in df.columns];print(df[cols].to_string(index=False))"
) else (
  echo （无 summary.csv）
)
echo ============================== status.json ==============================
if exist output\diag\status.json type output\diag\status.json
echo.
pause
