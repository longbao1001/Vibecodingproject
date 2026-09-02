@echo off
chcp 65001 >nul
title 智能故障预警系统 - 关闭
cd /d "%~dp0"
echo ============================================================
echo   面向车间设备运维的智能故障预警系统 - 一键关闭
echo ============================================================
echo.

if not exist "E:\anaconda\python.exe" (
    echo [ERROR] 未找到 Python: E:\anaconda\python.exe
    echo 请手动关闭占用5000端口的进程
    pause
    exit /b 1
)

E:\anaconda\python.exe stop_platform.py
