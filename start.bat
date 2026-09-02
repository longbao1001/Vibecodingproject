@echo off
chcp 65001 >nul
title 智能故障预警系统 - 启动
cd /d "%~dp0"
echo ============================================================
echo   面向车间设备运维的智能故障预警系统 - 一键启动
echo ============================================================
echo.

if not exist "E:\anaconda\python.exe" (
    echo [ERROR] 未找到 Python: E:\anaconda\python.exe
    echo 请修改 start_platform.py 中的 PYTHON_EXE 路径
    pause
    exit /b 1
)

E:\anaconda\python.exe start_platform.py
