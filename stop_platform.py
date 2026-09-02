# -*- coding: utf-8 -*-
# ============================================================
# 文件名：stop_platform.py
# 作用：系统一键关闭脚本（stop.bat双击后实际调用的就是这个文件）
# 通俗解释：
#   你可以把这个文件想象成一个"关机助手"，它按顺序做3件事：
#     步骤1：用netstat命令查出是谁占用了5000端口（拿到进程编号PID）
#     步骤2：用taskkill命令把这个进程强制结束掉
#     步骤3：再查一次端口，确认服务真的关闭了
#   最后还会问你要不要顺手清理运行日志server.log。
# 编码说明（重要）：
#   Windows中文系统的netstat/taskkill命令输出是GBK编码，
#   本脚本统一用 run_sys_cmd() 以GBK解码+容错方式读取，避免UnicodeDecodeError。
# ============================================================

import os
import sys
import subprocess
import time

# ============ 配置区 ============
PORT = 5000  # 要关闭的网站端口号（和start_platform.py保持一致）
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # 项目所在文件夹


# ============ 彩色提示函数 ============
def info(msg):    print(f"[INFO] {msg}")
def success(msg): print(f"[OK]   {msg}")
def warn(msg):    print(f"[WARN] {msg}")
def error(msg):   print(f"[ERROR]{msg}")


def run_sys_cmd(cmd):
    """
    执行系统命令（netstat/taskkill）的统一函数
    通俗解释：系统命令说"GBK方言"，这里用GBK去听，听不懂的字跳过，保证不崩溃。
    :param cmd: 命令列表，例如 ["netstat", "-ano"]
    :return: subprocess.CompletedResults 对象
    """
    return subprocess.run(
        cmd,
        capture_output=True,      # 抓回命令输出
        encoding="gbk",           # 用GBK解码（中文Windows原生编码）
        errors="replace"          # 解不了的字符用?代替，绝不报错崩溃
    )


def find_pid_by_port(port):
    """
    步骤1：通过端口号找到占用它的进程PID（进程编号，就像进程的身份证号）
    通俗解释：运行 netstat -ano 列出所有网络连接，从中筛出占用5000端口的行，
              每行最后一个数字就是PID。
    :param port: 端口号
    :return: PID字符串列表（可能有多个进程占用）
    """
    try:
        result = run_sys_cmd(["netstat", "-ano"])
        pids = set()  # 用集合自动去重
        for line in result.stdout.splitlines():
            if "LISTENING" not in line:
                continue
            parts = line.split()  # 按空格切分这一行
            if len(parts) >= 5:
                # 第2列是本地地址，形如 0.0.0.0:5000 或 [::]:5000
                local = parts[1]
                # 取最后一个冒号后的端口号，精确比较（避免50005/50006被误当成5000）
                port_part = local.rsplit(":", 1)[-1]
                if port_part == str(port):
                    pid = parts[-1]  # 最后一段就是PID
                    if pid.isdigit():
                        pids.add(pid)
        return list(pids)
    except Exception as e:
        error(f"查找端口进程失败: {e}")
        return []


def kill_process(pid):
    """
    步骤2：根据PID强制结束进程
    通俗解释：运行 taskkill /F /PID xxx，/F表示强制结束，/PID指定结束谁。
    :param pid: 进程编号
    :return: True=结束成功，False=结束失败
    """
    try:
        result = run_sys_cmd(["taskkill", "/F", "/PID", str(pid)])
        if result.returncode == 0:
            return True
        # taskkill失败时，尝试用Python自带的os.kill再试一次
        try:
            os.kill(int(pid), 9)  # 信号9表示强制终止
            return True
        except Exception:
            pass
        return False
    except Exception as e:
        error(f"终止进程失败: {e}")
        return False


def is_port_still_in_use(port):
    """步骤3：再次检查端口是否仍被占用（用来确认是否真的关掉了）"""
    try:
        result = run_sys_cmd(["netstat", "-ano"])
        for line in result.stdout.splitlines():
            if "LISTENING" not in line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                # 精确匹配端口号，避免50005/50006被误判
                port_part = parts[1].rsplit(":", 1)[-1]
                if port_part == str(port):
                    return True  # 还在监听说明没关掉
    except Exception:
        pass
    return False


def cleanup_log():
    """可选：删除运行日志文件server.log"""
    log_file = os.path.join(PROJECT_DIR, "server.log")
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
            info("已清理 server.log")
        except Exception:
            pass


def main():
    """主函数：查找→结束→确认，三步关闭系统"""
    print("=" * 60)
    print("  面向车间设备运维的智能故障预警系统 - 一键关闭")
    print("=" * 60)
    print()

    # 步骤1：查找占用5000端口的进程
    info(f"正在查找占用端口 {PORT} 的进程...")
    pids = find_pid_by_port(PORT)

    # 一个进程都没找到，说明服务本来就没运行
    if not pids:
        success(f"端口 {PORT} 未被占用，服务未在运行")
        print()
        input("按回车键退出...")
        return

    info(f"找到 {len(pids)} 个进程: {', '.join(pids)}")
    print()

    # 步骤2：逐个结束进程
    all_killed = True
    for pid in pids:
        info(f"正在终止进程 PID={pid}...")
        if kill_process(pid):
            success(f"进程 PID={pid} 已终止")
        else:
            warn(f"进程 PID={pid} 终止失败，请手动关闭")
            all_killed = False

    # 等待1.5秒让系统释放端口
    time.sleep(1.5)

    # 步骤3：确认端口是否已经释放
    print()
    if is_port_still_in_use(PORT):
        warn(f"端口 {PORT} 仍被占用，可能有其他进程使用该端口")
        warn("请手动检查: netstat -ano | findstr :5000")
    else:
        success(f"端口 {PORT} 已释放，服务已完全关闭")

    # 询问是否清理日志文件
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
    print("  如需重新启动，请双击运行: start.bat")
    print("=" * 60)
    print()
    input("按回车键退出...")


# 程序入口：直接运行这个文件时执行main函数
if __name__ == "__main__":
    main()
