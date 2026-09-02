# -*- coding: utf-8 -*-
# ============================================================
# 文件名：start_platform.py
# 作用：系统一键启动脚本（start.bat双击后实际调用的就是这个文件）
# 通俗解释：
#   你可以把这个文件想象成一个"开机向导"，它按顺序做5件事：
#     步骤1：检查Python解释器在不在
#     步骤2：检查需要的第三方库装没装（没装就自动装）
#     步骤3：检查数据库在不在（不在就自动建）
#     步骤4：检查AI模型文件在不在（不在就自动训练）
#     步骤5：后台启动网站服务，并自动打开浏览器
#   全部检查通过后，你就能在浏览器里访问系统了。
# 编码说明（重要）：
#   Windows中文系统的命令行默认用GBK编码，而Python内部用UTF-8，
#   如果不处理好编码，读取netstat等系统命令的输出时会报UnicodeDecodeError。
#   本脚本统一用 run_sys_cmd() 执行系统命令（GBK解码+容错），
#   用 UTF8_ENV 让被调用的Python子进程强制用UTF-8输出，避免乱码崩溃。
# ============================================================

import os
import sys
import subprocess
import time
import webbrowser

# ============ 配置区（如果你的Python装在别的位置，改这里） ============
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # 项目所在文件夹
PYTHON_EXE = r"E:\anaconda\python.exe"  # Python解释器的完整路径
APP_FILE = "app.py"                     # 网站主程序文件名
DB_FILE = os.path.join(PROJECT_DIR, "data", "equipment.db")        # 数据库文件路径
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "saved")           # 模型文件夹路径
REQUIRED_MODELS = ["anomaly_iforest.pkl", "fault_rf.pkl"]          # 必须存在的两个模型文件
HOST = "127.0.0.1"   # 网站访问地址（本机）
PORT = 5000          # 网站端口号
URL = f"http://{HOST}:{PORT}"  # 完整访问网址

# ============ 编码处理：让Python子进程强制用UTF-8输出，避免中文乱码 ============
# 复制当前系统环境变量，再追加两个Python专用变量
UTF8_ENV = os.environ.copy()
UTF8_ENV["PYTHONIOENCODING"] = "utf-8"  # 让Python的print用UTF-8编码输出
UTF8_ENV["PYTHONUTF8"] = "1"            # 开启Python的UTF-8模式


# ============ 彩色提示函数（让启动过程看得更清楚） ============
def info(msg):    print(f"[INFO] {msg}")    # 蓝色提示：正在做什么
def success(msg): print(f"[OK]   {msg}")    # 绿色提示：这一步成功了
def warn(msg):    print(f"[WARN] {msg}")    # 黄色提示：有小问题但不致命
def error(msg):   print(f"[ERROR]{msg}")    # 红色提示：出错了


def run_sys_cmd(cmd, **kwargs):
    """
    执行系统命令（如netstat/taskkill）的统一函数
    通俗解释：Windows系统命令说的是"GBK方言"，Python默认用"UTF-8普通话"去听，
              听不懂就会崩溃。这个函数让Python用GBK去听，听不懂的字直接跳过，
              保证不会因为编码问题报错。
    :param cmd: 命令列表，例如 ["netstat", "-ano"]
    :return: subprocess.CompletedResults 对象
    """
    return subprocess.run(
        cmd,
        capture_output=True,          # 把命令输出抓回来（不直接显示在屏幕上）
        encoding="gbk",               # 用GBK解码系统命令输出（中文Windows的原生编码）
        errors="replace",             # 万一有解不了的字符，用?代替，绝不崩溃
        **kwargs
    )


def check_python():
    """步骤1：检查Python解释器是否存在"""
    if not os.path.exists(PYTHON_EXE):
        error(f"Python解释器不存在: {PYTHON_EXE}")
        error("请修改脚本开头的 PYTHON_EXE 路径为你的Python安装路径")
        input("按回车键退出...")
        sys.exit(1)
    success(f"Python解释器: {PYTHON_EXE}")


def check_dependencies():
    """步骤2：检查核心依赖库是否安装，缺了就自动pip安装"""
    info("检查核心依赖...")
    deps = ["flask", "pandas", "sklearn", "numpy"]  # 系统运行必需的4个库
    missing = []
    for dep in deps:
        try:
            # 运行 python -c "import xxx" 测试库能不能导入
            result = subprocess.run(
                [PYTHON_EXE, "-c", f"import {dep}"],
                capture_output=True, encoding="utf-8", errors="replace",
                cwd=PROJECT_DIR, env=UTF8_ENV
            )
            if result.returncode != 0:
                missing.append(dep)  # 导入失败说明没装
        except Exception:
            missing.append(dep)
    if missing:
        warn(f"缺少依赖: {', '.join(missing)}")
        info("正在自动安装依赖...")
        subprocess.run(
            [PYTHON_EXE, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=PROJECT_DIR, env=UTF8_ENV
        )
        success("依赖安装完成")
    else:
        success("核心依赖已全部安装")


def check_database():
    """步骤3：检查数据库文件是否存在，不存在就运行database.py自动建库"""
    if os.path.exists(DB_FILE):
        success("数据库已存在: data/equipment.db")
        return
    warn("数据库不存在，正在初始化...")
    if not os.path.exists(os.path.join(PROJECT_DIR, "database.py")):
        error("database.py 不存在，无法初始化数据库")
        return
    result = subprocess.run(
        [PYTHON_EXE, "database.py"],
        cwd=PROJECT_DIR, capture_output=True,
        encoding="utf-8", errors="replace", env=UTF8_ENV
    )
    if result.returncode == 0 and os.path.exists(DB_FILE):
        success("数据库初始化完成")
    else:
        warn("database.py 执行完毕，请确认数据库是否已生成")
        if result.stderr:
            print(result.stderr[-500:])


def check_models():
    """步骤4：检查两个AI模型文件是否存在，不存在就运行train_model.py自动训练"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    # 两个模型文件都存在才算通过
    all_exist = all(
        os.path.exists(os.path.join(MODEL_DIR, m)) for m in REQUIRED_MODELS
    )
    if all_exist:
        success(f"模型文件已存在: {', '.join(REQUIRED_MODELS)}")
        return
    warn("模型文件不存在，正在训练模型（可能需要几分钟）...")
    if not os.path.exists(os.path.join(PROJECT_DIR, "train_model.py")):
        error("train_model.py 不存在，无法训练模型")
        return
    result = subprocess.run(
        [PYTHON_EXE, "train_model.py"],
        cwd=PROJECT_DIR, capture_output=True,
        encoding="utf-8", errors="replace", env=UTF8_ENV
    )
    if result.returncode == 0:
        success("模型训练完成")
    else:
        warn("train_model.py 执行完毕，请确认模型是否已生成")
        if result.stderr:
            print(result.stderr[-500:])


def is_port_in_use():
    """检查5000端口是否已经被占用（被占用说明服务可能已经在跑了）"""
    try:
        result = run_sys_cmd(["netstat", "-ano"])  # netstat列出所有网络连接
        for line in result.stdout.splitlines():
            if "LISTENING" not in line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                # 第2列是本地地址，形如 0.0.0.0:5000 或 [::]:5000
                local = parts[1]
                # 取最后一个冒号后面的端口号，精确比较（避免50005/50006被误判成5000）
                port_part = local.rsplit(":", 1)[-1]
                if port_part == str(PORT):
                    return True  # 找到5000端口正在监听
    except Exception:
        pass
    return False


def start_server():
    """步骤5：后台启动Flask网站服务"""
    if is_port_in_use():
        warn(f"端口 {PORT} 已被占用，服务可能已在运行")
        return None
    info("正在启动Flask服务...")
    # 把网站运行日志写入server.log文件（用UTF-8，配合子进程的UTF-8输出）
    log_file = os.path.join(PROJECT_DIR, "server.log")
    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            [PYTHON_EXE, APP_FILE],
            cwd=PROJECT_DIR,
            stdout=f,                       # 正常输出写入日志文件
            stderr=subprocess.STDOUT,       # 报错也合并写入同一个日志文件
            env=UTF8_ENV,                   # 让Flask用UTF-8输出，和日志文件编码一致
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
    # 最多等15秒，每秒检查一次端口，端口开始监听就说明启动成功了
    for i in range(15):
        time.sleep(1)
        if is_port_in_use():
            success(f"Flask服务已启动 (PID={proc.pid})")
            return proc
    warn("服务启动超时，请查看 server.log 日志排查问题")
    return proc


def open_browser():
    """自动打开默认浏览器访问系统"""
    info(f"正在打开浏览器: {URL}")
    try:
        webbrowser.open(URL)
        success("浏览器已打开")
    except Exception as e:
        warn(f"自动打开浏览器失败: {e}")
        print(f"请手动在浏览器输入: {URL}")


def main():
    """主函数：按顺序执行5个检查步骤，然后启动服务"""
    print("=" * 60)
    print("  面向车间设备运维的智能故障预警系统 - 一键启动")
    print("=" * 60)
    print()

    os.chdir(PROJECT_DIR)  # 切换到项目目录，保证相对路径都能找到文件

    # 依次执行4项检查
    check_python()
    check_dependencies()
    check_database()
    check_models()

    print()
    proc = start_server()  # 启动网站服务

    # 服务启动成功（或端口已被占用）就打开浏览器
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
        print("  关闭系统请双击运行: stop.bat")
        print("=" * 60)
    else:
        error("系统启动失败，请检查日志")

    print()
    input("按回车键关闭此窗口（网站服务会继续在后台运行）...")


# 程序入口：直接运行这个文件时执行main函数
if __name__ == "__main__":
    main()
