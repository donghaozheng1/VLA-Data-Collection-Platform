@echo off
REM ============================================================
REM  One-click launcher for HKC Data Platform  (ASCII only!)
REM  NOTE: this file must stay ASCII-only, because cmd.exe reads
REM        .cmd files using the system ANSI codepage (GBK on
REM        Chinese Windows), and UTF-8 Chinese would break it.
REM ============================================================
chcp 65001 >nul 2>nul
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%CD%\.vendor"
set "PYTHONIOENCODING=utf-8"

where python >nul 2>nul
if errorlevel 1 (
    echo [x] python not found on PATH.
    echo     Please activate conda base first, then run this again.
    echo.
    pause
    exit /b 1
)

if not exist ".vendor\fastapi" (
    echo [x] Missing ".vendor\fastapi" -- dependency folder absent or moved.
    echo     Expected: %CD%\.vendor
    echo.
    pause
    exit /b 1
)

echo.
echo   HKC Data Platform starting...
echo   URL    : http://127.0.0.1:8000
echo   Login  : admin / admin123
echo   Stop   : press Ctrl+C in this window
echo.

start "" "http://127.0.0.1:8000"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
