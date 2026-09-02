# -*- coding: utf-8 -*-
"""
面向车间设备运维的智能故障预警系统 - 一键启动脚本
功能：检查环境 -> 初始化数据库/模型 -> 启动Flask服务 -> 自动打开浏览器
"""

import os
import sys
import subprocess
import time
import webbrowser

# ============ 配置 ============
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = r"E:\anaconda\python.exe"
APP_FILE = "app.py"
DB_FILE = os.path.join(PROJECT_DIR, "data", "equipment.db")
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "saved")
REQUIRED_MODELS = ["anomaly_iforest.pkl", "fault_rf.pkl"]
HOST = "127.0.0.1"
PORT = 5000
URL = f"http://{HOST}:{PORT}"

# 颜色输出（Windows CMD）
def info(msg):    print(f"[INFO] {msg}")
def success(msg): print(f"[OK]   {msg}")
def warn(msg):    print(f"[WARN] {msg}")
def error(msg):   print(f"[ERROR]{msg}")

def check_python():
    """检查Python解释器"""
    if not os.path.exists(PYTHON_EXE):
        error(f"Python解释器不存在: {PYTHON_EXE}")
        error("请修改脚本中的 PYTHON_EXE 路径为你的Python安装路径")
        input("按回车键退出...")
        sys.exit(1)
    success(f"Python解释器: {PYTHON_EXE}")

def check_dependencies():
    """检查核心依赖是否安装"""
    info("检查核心依赖...")
    deps = ["flask", "pandas", "sklearn", "numpy"]
    missing = []
    for dep in deps:
        try:
            result = subprocess.run(
                [PYTHON_EXE, "-c", f"import {dep}"],
                capture_output=True, text=True, cwd=PROJECT_DIR
            )
            if result.returncode != 0:
                missing.append(dep)
        except Exception:
            missing.append(dep)
    if missing:
        warn(f"缺少依赖: {', '.join(missing)}")
        info("正在自动安装依赖...")
        subprocess.run(
            [PYTHON_EXE, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=PROJECT_DIR
        )
        success("依赖安装完成")
    else:
        success("核心依赖已全部安装")

def check_database():
    """检查数据库是否存在，不存在则初始化"""
    if os.path.exists(DB_FILE):
        success(f"数据库已存在: data/equipment.db")
        return
    warn("数据库不存在，正在初始化...")
    if not os.path.exists(os.path.join(PROJECT_DIR, "database.py")):
        error("database.py 不存在，无法初始化数据库")
        return
    result = subprocess.run(
        [PYTHON_EXE, "database.py"],
        cwd=PROJECT_DIR, capture_output=True, text=True
    )
    if result.returncode == 0 and os.path.exists(DB_FILE):
        success("数据库初始化完成")
    else:
        warn("database.py 执行完毕，请确认数据库是否已生成")
        if result.stderr:
            print(result.stderr[-500:])

def check_models():
    """检查模型文件是否存在，不存在则训练"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    all_exist = all(
        os.path.exists(os.path.join(MODEL_DIR, m)) for m in REQUIRED_MODELS
    )
    if all_exist:
        success(f"模型文件已存在: {', '.join(REQUIRED_MODELS)}")
        return
    warn("模型文件不存在，正在训练模型...")
    if not os.path.exists(os.path.join(PROJECT_DIR, "train_model.py")):
        error("train_model.py 不存在，无法训练模型")
        return
    result = subprocess.run(
        [PYTHON_EXE, "train_model.py"],
        cwd=PROJECT_DIR, capture_output=True, text=True
    )
    if result.returncode == 0:
        success("模型训练完成")
    else:
        warn("train_model.py 执行完毕，请确认模型是否已生成")
        if result.stderr:
            print(result.stderr[-500:])

def is_port_in_use():
    """检查5000端口是否已被占用"""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if f":{PORT}" in line and "LISTENING" in line:
                return True
    except Exception:
        pass
    return False

def start_server():
    """启动Flask服务（后台运行）"""
    if is_port_in_use():
        warn(f"端口 {PORT} 已被占用，服务可能已在运行")
        return None
    info("正在启动Flask服务...")
    # 用新的控制台窗口启动，方便查看日志
    log_file = os.path.join(PROJECT_DIR, "server.log")
    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            [PYTHON_EXE, APP_FILE],
            cwd=PROJECT_DIR,
            stdout=f,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
    # 等待服务启动
    for i in range(15):
        time.sleep(1)
        if is_port_in_use():
            success(f"Flask服务已启动 (PID={proc.pid})")
            return proc
    warn("服务启动超时，请检查 server.log 日志")
    return proc

def open_browser():
    """自动打开浏览器"""
    info(f"正在打开浏览器: {URL}")
    try:
        webbrowser.open(URL)
        success("浏览器已打开")
    except Exception as e:
        warn(f"自动打开浏览器失败: {e}")
        print(f"请手动访问: {URL}")

def main():
    print("=" * 60)
    print("  面向车间设备运维的智能故障预警系统 - 一键启动")
    print("=" * 60)
    print()

    os.chdir(PROJECT_DIR)

    check_python()
    check_dependencies()
    check_database()
    check_models()

    print()
    proc = start_server()

    if proc or is_port_in_use():
        time.sleep(1)
        open_browser()
        print()
        print("=" * 60)
        success("系统启动完成！")
        print(f"  访问地址: {URL}")
        print(f"  项目目录: {PROJECT_DIR}")
        print(f"  日志文件: server.log")
        print()
        print("  关闭系统请运行: stop.bat")
        print("=" * 60)
    else:
        error("系统启动失败，请检查日志")

    print()
    input("按回车键关闭此窗口（服务将继续在后台运行）...")

if __name__ == "__main__":
    main()
