"""
SKAB 工业设备异常数据集预处理脚本
功能：读取 data/SKAB-master/data 下全部故障场景 CSV，完成清洗合并，输出到 data/processed/out.csv
运行方式：python preprocess.py
"""
import os
import pandas as pd

# 项目根目录（脚本所在目录）
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "SKAB-master", "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "out.csv")

# SKAB 原始字段（分号分隔，共 11 列）
RAW_COLUMNS = [
    "datetime", "Accelerometer1RMS", "Accelerometer2RMS", "Current",
    "Pressure", "Temperature", "Thermocouple", "Voltage",
    "Volume Flow RateRMS", "anomaly", "changepoint"
]

# 预处理后保留的业务字段（含附加标签）
KEEP_COLUMNS = [
    "datetime", "Accelerometer1RMS", "Accelerometer2RMS", "Current",
    "Pressure", "Temperature", "Thermocouple", "Voltage",
    "Volume Flow RateRMS", "anomaly", "fault_type"
]


def load_all_csv(raw_dir):
    """遍历 raw_dir 下所有子目录的 CSV，合并为一个 DataFrame，附加故障类型标签"""
    all_frames = []
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"原始数据目录不存在: {raw_dir}")

    for fault_type in sorted(os.listdir(raw_dir)):
        sub_dir = os.path.join(raw_dir, fault_type)
        if not os.path.isdir(sub_dir):
            continue
        for file_name in sorted(os.listdir(sub_dir)):
            if not file_name.endswith(".csv"):
                continue
            file_path = os.path.join(sub_dir, file_name)
            df = pd.read_csv(file_path, sep=";", header=0)
            # 校验核心传感器字段（9列必须存在）
            core_cols = [c for c in RAW_COLUMNS if c not in ("anomaly", "changepoint")]
            missing_core = [c for c in core_cols if c not in df.columns]
            if missing_core:
                print(f"[警告] {file_path} 缺少核心字段: {missing_core}，跳过")
                continue
            # anomaly-free 等正常工况数据缺少标签列，填充为 0（正常）
            if "anomaly" not in df.columns:
                df["anomaly"] = 0
                print(f"[补充标签] {fault_type}/{file_name} 无anomaly列，填充为0（正常工况）")
            if "changepoint" not in df.columns:
                df["changepoint"] = 0
            df["fault_type"] = fault_type  # 附加故障类型标签（valve1/valve2/other/anomaly-free）
            all_frames.append(df)
            print(f"[读取] {fault_type}/{file_name}  样本数: {len(df)}")

    if not all_frames:
        raise RuntimeError("未读取到任何有效 CSV 文件")
    return pd.concat(all_frames, ignore_index=True)


def preprocess(df):
    """数据清洗与预处理"""
    # 1. 时间戳转换
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # 2. 删除时间戳无效的行
    df = df.dropna(subset=["datetime"])

    # 3. 去除完全重复行
    before = len(df)
    df = df.drop_duplicates()
    print(f"[去重] 删除重复行: {before - len(df)}")

    # 4. 数值列缺失值处理：先向前填充，残余缺失用该列均值填充
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].ffill()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # 5. anomaly 标签转为整数
    df["anomaly"] = df["anomaly"].astype(int)

    # 6. 按时间排序
    df = df.sort_values("datetime").reset_index(drop=True)

    # 7. 保留业务字段
    keep = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[keep]

    # 8. 保留字段后再次去重（确保最终输出无重复行）
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if before - len(df) > 0:
        print(f"[二次去重] 字段筛选后删除重复行: {before - len(df)}")
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("开始 SKAB 数据集预处理")
    print("=" * 60)

    # 1. 加载全部原始 CSV
    df_raw = load_all_csv(RAW_DATA_DIR)
    print(f"\n[合并完成] 原始总样本数: {len(df_raw)}")
    print(f"[故障类型分布]\n{df_raw['fault_type'].value_counts()}")
    print(f"[异常标签分布]\n{df_raw['anomaly'].value_counts()}")

    # 2. 预处理
    df_processed = preprocess(df_raw)
    print(f"\n[预处理完成] 输出样本数: {len(df_processed)}")
    print(f"[字段列表] {list(df_processed.columns)}")
    print(f"\n[预处理后数据质量统计]")
    print(f"  总样本数: {len(df_processed)}")
    print(f"  缺失值总数: {df_processed.isnull().sum().sum()}")
    print(f"  重复行数: {df_processed.duplicated().sum()}")
    print(f"  时间范围: {df_processed['datetime'].min()} ~ {df_processed['datetime'].max()}")
    print(f"  异常标签分布: {df_processed['anomaly'].value_counts().to_dict()}")
    print(f"  故障类型分布: {df_processed['fault_type'].value_counts().to_dict()}")

    # 3. 输出
    df_processed.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n[输出文件] {OUTPUT_FILE}")
    print("=" * 60)
    print("预处理全部完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
