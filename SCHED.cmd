@echo off
setlocal

pushd "%~dp0" >nul || exit /b 1
set "SCHED_CLI_PROG=SCHED"

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python -m interfaces.cli %*
) else (
    py -3 -m interfaces.cli %*
)
set "_SCHED_EXIT=%ERRORLEVEL%"

popd >nul
exit /b %_SCHED_EXIT%
