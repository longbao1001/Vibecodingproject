"""
数据库模块：SQLite 建表、数据入库、数据查询
表结构：devices（设备表）、sensor_data（传感器数据表）、alerts（告警表）、maintenance（运维记录表）
"""
import os
import sqlite3
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT_DIR, "data", "equipment.db")
PROCESSED_CSV = os.path.join(ROOT_DIR, "data", "processed", "out.csv")

# 传感器特征列（8路传感器）
SENSOR_COLS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"
]

# 4 台虚拟设备，分别对应不同故障场景
DEVICE_MAP = {
    "anomaly-free": ("PUMP-001", "1号冷却水泵", "一号车间A区"),
    "valve1": ("PUMP-002", "2号循环水泵", "一号车间B区"),
    "valve2": ("PUMP-003", "3号增压水泵", "二号车间A区"),
    "other": ("PUMP-004", "4号供水泵", "二号车间B区"),
}


def get_conn():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """创建数据库表"""
    conn = get_conn()
    cur = conn.cursor()

    # 设备表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS devices (
            device_id   TEXT PRIMARY KEY,
            device_name TEXT NOT NULL,
            location    TEXT,
            fault_type  TEXT,
            status      TEXT DEFAULT '运行中'
        )
    """)

    # 传感器数据表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id             TEXT,
            timestamp             TEXT,
            Accelerometer1RMS     REAL,
            Accelerometer2RMS     REAL,
            Current               REAL,
            Pressure              REAL,
            Temperature           REAL,
            Thermocouple          REAL,
            Voltage               REAL,
            Volume_Flow_RateRMS   REAL,
            anomaly               INTEGER,
            fault_type            TEXT
        )
    """)

    # 告警表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id    TEXT,
            timestamp    TEXT,
            alert_type   TEXT,
            severity     TEXT,
            message      TEXT,
            anomaly_score REAL,
            fault_prob   REAL,
            root_cause   TEXT,
            status       TEXT DEFAULT '未处理'
        )
    """)

    # 运维记录表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS maintenance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id    INTEGER,
            device_id   TEXT,
            action      TEXT,
            operator    TEXT DEFAULT '运维人员',
            create_time TEXT DEFAULT (datetime('now','localtime'))
        )
    """)

    conn.commit()
    conn.close()
    print("[数据库] 表结构创建完成")


def load_devices():
    """初始化设备档案"""
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


def load_sensor_data():
    """将预处理后的 CSV 数据导入 sensor_data 表"""
    if not os.path.exists(PROCESSED_CSV):
        raise FileNotFoundError(f"预处理数据不存在: {PROCESSED_CSV}，请先运行 preprocess.py")

    df = pd.read_csv(PROCESSED_CSV)
    # 根据 fault_type 映射到虚拟设备
    df["device_id"] = df["fault_type"].map({k: v[0] for k, v in DEVICE_MAP.items()})
    df = df.dropna(subset=["device_id"])

    conn = get_conn()
    # 清空旧数据，避免重复导入
    conn.execute("DELETE FROM sensor_data")
    insert_sql = """
        INSERT INTO sensor_data
        (device_id, timestamp, Accelerometer1RMS, Accelerometer2RMS, Current,
         Pressure, Temperature, Thermocouple, Voltage, Volume_Flow_RateRMS, anomaly, fault_type)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """
    records = df[
        ["device_id", "datetime"] + SENSOR_COLS + ["anomaly", "fault_type"]
    ].values.tolist()
    conn.executemany(insert_sql, records)
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM sensor_data").fetchone()[0]
    conn.close()
    print(f"[数据库] 传感器数据导入完成，共 {count} 条")


def setup_database():
    """一键初始化数据库：建表 + 设备档案 + 数据入库"""
    init_db()
    load_devices()
    load_sensor_data()
    print("[数据库] 初始化全部完成")


if __name__ == "__main__":
    setup_database()
