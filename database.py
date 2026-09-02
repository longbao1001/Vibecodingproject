# ============================================================
# 文件名：database.py
# 作用：数据库管理模块，负责创建数据库表、导入数据、提供查询连接
# 通俗解释：
#   你可以把这个文件想象成一个"仓库管理员"。
#   仓库（数据库文件 data/equipment.db）里有4个货架（4张表）：
#     货架1：devices（设备档案）—— 记录每台设备的名字、位置、类型
#     货架2：sensor_data（传感器数据）—— 记录46669条传感器采集的数据
#     货架3：alerts（告警记录）—— 记录系统发现的所有告警
#     货架4：maintenance（运维记录）—— 记录运维人员处理告警的历史
#   这个文件负责：建仓库、摆货架、把货物搬进去、给其他人开仓库门
# 运行方式：python database.py（会自动建表+导入数据）
# ============================================================

import os
import sqlite3
import pandas as pd

# ---------- 路径配置 ----------
# 获取当前文件所在的文件夹路径（项目根目录）
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据库文件存放路径：项目目录/data/equipment.db
DB_PATH = os.path.join(ROOT_DIR, "data", "equipment.db")
# 预处理后的CSV数据文件路径（数据预处理脚本preprocess.py的输出）
PROCESSED_CSV = os.path.join(ROOT_DIR, "data", "processed", "out.csv")

# ---------- 传感器列名 ----------
# 8路传感器的英文名字，和数据集中的列名完全一致
# 这些名字在整个项目中统一使用，避免写错
SENSOR_COLS = [
    "Accelerometer1RMS",   # 1号振动加速度（RMS表示均方根值，反映振动强度）
    "Accelerometer2RMS",   # 2号振动加速度
    "Current",              # 电机电流
    "Pressure",             # 压力
    "Temperature",          # 温度
    "Thermocouple",         # 热电偶温度（更精确的温度测量）
    "Voltage",              # 电机电压
    "Volume Flow RateRMS"   # 流量（体积流量的均方根值）
]

# ---------- 设备映射表 ----------
# 原始数据集中有4种故障场景，我们把它们映射成4台虚拟设备
# 格式：故障场景名 → (设备编号, 设备名字, 设备位置)
DEVICE_MAP = {
    "anomaly-free": ("PUMP-001", "1号冷却水泵", "一号车间A区"),  # 正常工况，作为健康设备
    "valve1":       ("PUMP-002", "2号循环水泵", "一号车间B区"),  # 阀门1故障场景
    "valve2":       ("PUMP-003", "3号增压水泵", "二号车间A区"),  # 阀门2故障场景
    "other":        ("PUMP-004", "4号供水泵",   "二号车间B区"),  # 其他故障场景
}


# ============================================================
# 函数1：获取数据库连接
# ============================================================
def get_conn():
    """
    获取数据库连接（就像用钥匙打开仓库大门）
    返回：一个数据库连接对象，可以用它执行SQL查询
    注意：用完后要记得调用 conn.close() 关闭连接（关门省电）
    """
    conn = sqlite3.connect(DB_PATH)
    # 设置行工厂为Row，这样查询结果可以用列名访问（比如 row["device_id"]）
    # 不设置的话只能用下标访问（比如 row[0]），容易搞错
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# 函数2：创建数据库表（建仓库+摆货架）
# ============================================================
def init_db():
    """
    创建4张数据库表（如果表已经存在就跳过，不会重复创建）
    4张表：
      1. devices      —— 设备档案表
      2. sensor_data  —— 传感器数据表（最大的表，46669条数据）
      3. alerts       —— 告警记录表
      4. maintenance  —— 运维记录表
    """
    conn = get_conn()
    cur = conn.cursor()  # 创建游标（就像拿个扫描枪，用来执行SQL命令）

    # ---------- 表1：设备档案表 ----------
    # 记录每台设备的基本信息
    cur.execute("""
        CREATE TABLE IF NOT EXISTS devices (
            device_id   TEXT PRIMARY KEY,   -- 设备编号，主键（唯一标识，不能重复）
            device_name TEXT NOT NULL,       -- 设备名字，不能为空
            location    TEXT,                 -- 设备位置
            fault_type  TEXT,                 -- 对应故障场景类型
            status      TEXT DEFAULT '运行中' -- 设备状态，默认"运行中"
        )
    """)

    # ---------- 表2：传感器数据表 ----------
    # 记录每一条传感器采集的数据，是项目中最大的表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增ID，每条数据唯一编号
            device_id             TEXT,       -- 所属设备编号
            timestamp             TEXT,       -- 采集时间
            Accelerometer1RMS     REAL,       -- 1号振动加速度（REAL表示小数）
            Accelerometer2RMS     REAL,       -- 2号振动加速度
            Current               REAL,       -- 电机电流
            Pressure              REAL,       -- 压力
            Temperature           REAL,       -- 温度
            Thermocouple          REAL,       -- 热电偶温度
            Voltage               REAL,       -- 电机电压
            Volume_Flow_RateRMS   REAL,       -- 流量（注意：数据库里空格换成了下划线，因为SQL列名不能有空格）
            anomaly               INTEGER,    -- 是否异常（0=正常，1=异常）
            fault_type            TEXT        -- 故障场景类型
        )
    """)

    # ---------- 表3：告警记录表 ----------
    # 系统扫描发现异常后，自动生成的告警记录
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,  -- 告警ID
            device_id    TEXT,       -- 所属设备
            timestamp    TEXT,       -- 告警时间
            alert_type   TEXT,       -- 告警类型（如"故障预警"）
            severity     TEXT,       -- 告警级别（高危/中危）
            message      TEXT,       -- 告警消息内容（人类可读的描述）
            anomaly_score REAL,      -- 异常分数
            fault_prob   REAL,       -- 故障概率
            root_cause   TEXT,       -- 根因分析结论
            status       TEXT DEFAULT '未处理'  -- 处理状态，默认"未处理"
        )
    """)

    # ---------- 表4：运维记录表 ----------
    # 运维人员处理告警后，自动生成的处置记录
    cur.execute("""
        CREATE TABLE IF NOT EXISTS maintenance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,  -- 记录ID
            alert_id    INTEGER,    -- 关联的告警ID（这条运维记录处理的是哪条告警）
            device_id   TEXT,       -- 所属设备
            action      TEXT,       -- 处置措施（运维人员做了什么）
            operator    TEXT DEFAULT '运维人员',  -- 处理人，默认"运维人员"
            create_time TEXT DEFAULT (datetime('now','localtime'))  -- 创建时间，默认当前时间
        )
    """)

    conn.commit()  # 提交事务（把建表操作真正写入数据库文件）
    conn.close()   # 关闭连接
    print("[数据库] 表结构创建完成")


# ============================================================
# 函数3：初始化设备档案（把4台虚拟设备信息录入设备表）
# ============================================================
def load_devices():
    """
    把DEVICE_MAP中定义的4台设备信息插入到devices表中
    使用 INSERT OR REPLACE：如果设备已存在就更新，不存在就插入
    """
    conn = get_conn()
    cur = conn.cursor()
    for fault_type, (dev_id, name, loc) in DEVICE_MAP.items():
        cur.execute(
            "INSERT OR REPLACE INTO devices(device_id, device_name, location, fault_type) VALUES(?,?,?,?)",
            (dev_id, name, loc, fault_type)
        )
    conn.commit()
    conn.close()
    print("[数据库] 设备档案初始化完成，共4台设备")


# ============================================================
# 函数4：导入传感器数据（把CSV文件中的数据搬进数据库）
# ============================================================
def load_sensor_data():
    """
    将预处理后的CSV数据导入 sensor_data 表
    步骤：
      1. 读取CSV文件
      2. 根据故障场景类型映射到对应的虚拟设备
      3. 清空旧数据（避免重复导入）
      4. 批量插入所有数据
    """
    # 先检查预处理后的CSV文件是否存在
    if not os.path.exists(PROCESSED_CSV):
        raise FileNotFoundError(f"预处理数据不存在: {PROCESSED_CSV}，请先运行 preprocess.py")

    # 用pandas读取CSV文件（就像用Excel打开表格）
    df = pd.read_csv(PROCESSED_CSV)
    # 根据故障类型映射到设备编号（比如 anomaly-free → PUMP-001）
    df["device_id"] = df["fault_type"].map({k: v[0] for k, v in DEVICE_MAP.items()})
    # 删掉映射不到设备的行（理论上不会有，防御性编程）
    df = df.dropna(subset=["device_id"])

    conn = get_conn()
    # 清空旧数据（避免重复导入导致数据翻倍）
    conn.execute("DELETE FROM sensor_data")
    # 插入数据的SQL语句（? 是占位符，后面会被真实数据替换，防止SQL注入）
    insert_sql = """
        INSERT INTO sensor_data
        (device_id, timestamp, Accelerometer1RMS, Accelerometer2RMS, Current,
         Pressure, Temperature, Thermocouple, Voltage, Volume_Flow_RateRMS, anomaly, fault_type)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """
    # 从DataFrame中提取需要的列，转成列表格式
    # 注意：CSV中的时间列名叫"datetime"，数据库中叫"timestamp"，这里对应上
    records = df[
        ["device_id", "datetime"] + SENSOR_COLS + ["anomaly", "fault_type"]
    ].values.tolist()
    # 批量插入（executemany 一次执行多条插入，比一条条插快很多）
    conn.executemany(insert_sql, records)
    conn.commit()
    # 查一下导入了多少条，打印出来确认
    count = conn.execute("SELECT COUNT(*) FROM sensor_data").fetchone()[0]
    conn.close()
    print(f"[数据库] 传感器数据导入完成，共 {count} 条")


# ============================================================
# 函数5：一键初始化（建表+设备档案+数据导入，一条龙服务）
# ============================================================
def setup_database():
    """
    一键初始化数据库：
      步骤1：创建4张表
      步骤2：录入4台设备档案
      步骤3：导入46669条传感器数据
    运行 python database.py 时会自动调用这个函数
    """
    init_db()           # 步骤1：建表
    load_devices()      # 步骤2：设备档案
    load_sensor_data()  # 步骤3：导入数据
    print("[数据库] 初始化全部完成")


# ============================================================
# 程序入口：直接运行这个文件时，执行一键初始化
# ============================================================
if __name__ == "__main__":
    setup_database()
