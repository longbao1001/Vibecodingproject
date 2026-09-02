# -*- coding: utf-8 -*-
"""
面向车间设备运维的智能故障预警系统 - 一键关闭脚本
功能：查找占用5000端口的进程 -> 终止进程 -> 确认关闭
"""

import os
import sys
import subprocess
import time

# ============ 配置 ============
PORT = 5000
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def info(msg):    print(f"[INFO] {msg}")
def success(msg): print(f"[OK]   {msg}")
def warn(msg):    print(f"[WARN] {msg}")
def error(msg):   print(f"[ERROR]{msg}")

def find_pid_by_port(port):
    """通过端口号查找进程PID"""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True
        )
        pids = set()
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.strip().split()
                if parts:
                    pid = parts[-1]
                    if pid.isdigit():
                        pids.add(pid)
        return list(pids)
    except Exception as e:
        error(f"查找端口进程失败: {e}")
        return []

def kill_process(pid):
    """终止指定PID的进程"""
    try:
        result = subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return True
        # 尝试用Python的os.kill
        try:
            os.kill(int(pid), 9)
            return True
        except Exception:
            pass
        return False
    except Exception as e:
        error(f"终止进程失败: {e}")
        return False

def is_port_still_in_use(port):
    """检查端口是否仍被占用"""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                return True
    except Exception:
        pass
    return False

def cleanup_log():
    """可选：清理server.log"""
    log_file = os.path.join(PROJECT_DIR, "server.log")
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
            info("已清理 server.log")
        except Exception:
            pass

def main():
    print("=" * 60)
    print("  面向车间设备运维的智能故障预警系统 - 一键关闭")
    print("=" * 60)
    print()

    info(f"正在查找占用端口 {PORT} 的进程...")
    pids = find_pid_by_port(PORT)

    if not pids:
        success(f"端口 {PORT} 未被占用，服务未在运行")
        print()
        input("按回车键退出...")
        return

    info(f"找到 {len(pids)} 个进程: {', '.join(pids)}")
    print()

    all_killed = True
    for pid in pids:
        info(f"正在终止进程 PID={pid}...")
        if kill_process(pid):
            success(f"进程 PID={pid} 已终止")
        else:
            warn(f"进程 PID={pid} 终止失败，请手动关闭")
            all_killed = False

    # 等待端口释放
    time.sleep(1.5)

    print()
    if is_port_still_in_use(PORT):
        warn(f"端口 {PORT} 仍被占用，可能有其他进程使用该端口")
        warn("请手动检查: netstat -ano | findstr :5000")
    else:
        success(f"端口 {PORT} 已释放，服务已完全关闭")

    # 询问是否清理日志
    print()
    try:
        choice = input("是否清理 server.log 日志文件？(y/N): ").strip().lower()
        if choice == "y":
            cleanup_log()
    except (EOFError, KeyboardInterrupt):
        pass

    print()
    print("=" * 60)
    success("系统关闭完成")
    print("  如需重新启动，请运行: start.bat")
    print("=" * 60)
    print()
    input("按回车键退出...")

if __name__ == "__main__":
    main()
