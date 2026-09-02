# ============================================================
# 文件名：preprocess.py
# 作用：数据预处理脚本 —— 把原始的SKAB数据集清洗整理成算法能用的干净数据
# 通俗解释：
#   你可以把这个文件想象成一个"数据清洁工"。
#   原始数据集（data/SKAB-master/data/）里有38个CSV文件，散落在4个文件夹里，
#   每个文件的格式可能不太一样（有的缺标签列、有的有重复行、有的有缺失值）。
#   这个"清洁工"要做的事：
#     1. 把38个文件全部读出来，合并成一个大表格
#     2. 给每条数据贴上"故障类型"标签（它来自哪个文件夹）
#     3. 清洗：删重复行、填缺失值、转时间格式、按时间排序
#     4. 输出一个干净的CSV文件（data/processed/out.csv，共46669条）
#   后面的模型训练和数据库导入，都用这个干净的文件。
# 运行方式：python preprocess.py
# 运行后会生成 data/processed/out.csv
# ============================================================

import os
import pandas as pd

# ---------- 路径配置 ----------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "SKAB-master", "data")  # 原始数据文件夹
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed")  # 输出文件夹
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "out.csv")  # 输出文件路径

# ---------- 原始数据列名 ----------
# SKAB原始CSV文件有11列（用分号分隔，不是逗号）
RAW_COLUMNS = [
    "datetime",                # 采集时间
    "Accelerometer1RMS",       # 1号振动加速度
    "Accelerometer2RMS",       # 2号振动加速度
    "Current",                  # 电机电流
    "Pressure",                 # 压力
    "Temperature",              # 温度
    "Thermocouple",             # 热电偶温度
    "Voltage",                  # 电机电压
    "Volume Flow RateRMS",      # 流量
    "anomaly",                  # 是否异常（0=正常，1=异常）
    "changepoint"               # 变化点标记（本项目不用，预处理后会删掉）
]

# ---------- 预处理后保留的列 ----------
# 删掉changepoint列，加上fault_type列（故障类型标签）
KEEP_COLUMNS = [
    "datetime", "Accelerometer1RMS", "Accelerometer2RMS", "Current",
    "Pressure", "Temperature", "Thermocouple", "Voltage",
    "Volume Flow RateRMS", "anomaly", "fault_type"
]


# ============================================================
# 函数1：读取所有CSV文件并合并（"清洁工"把散落的文件收集到一起）
# ============================================================
def load_all_csv(raw_dir):
    """
    遍历原始数据文件夹下所有子目录的CSV文件，合并成一个大表格
    通俗解释：
      原始数据文件夹里有4个子文件夹（anomaly-free/valve1/valve2/other），
      每个文件夹里有若干个CSV文件。这个函数把它们全部读出来，
      给每条数据加上"fault_type"标签（它来自哪个文件夹），然后合并成一个大表。
    :param raw_dir: 原始数据文件夹路径
    :return: 合并后的DataFrame
    """
    all_frames = []  # 用来存放每个CSV文件的DataFrame，最后一起合并

    # 检查原始数据文件夹是否存在
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"原始数据目录不存在: {raw_dir}")

    # 遍历4个故障类型文件夹（按名字排序，保证顺序一致）
    for fault_type in sorted(os.listdir(raw_dir)):
        sub_dir = os.path.join(raw_dir, fault_type)
        if not os.path.isdir(sub_dir):
            continue  # 跳过非文件夹（比如隐藏文件）

        # 遍历这个文件夹里的所有文件
        for file_name in sorted(os.listdir(sub_dir)):
            if not file_name.endswith(".csv"):
                continue  # 只处理CSV文件，跳过其他文件

            file_path = os.path.join(sub_dir, file_name)
            # 读取CSV文件（注意：SKAB的CSV用分号分隔，不是逗号，所以要指定sep=";"）
            df = pd.read_csv(file_path, sep=";", header=0)

            # 校验核心字段是否齐全（9个传感器+时间列必须有）
            core_cols = [c for c in RAW_COLUMNS if c not in ("anomaly", "changepoint")]
            missing_core = [c for c in core_cols if c not in df.columns]
            if missing_core:
                print(f"[警告] {file_path} 缺少核心字段: {missing_core}，跳过")
                continue

            # anomaly-free（正常工况）的文件没有anomaly标签列，手动填充为0（正常）
            if "anomaly" not in df.columns:
                df["anomaly"] = 0
                print(f"[补充标签] {fault_type}/{file_name} 无anomaly列，填充为0（正常工况）")
            # changepoint列如果没有也填充0（这个列本项目不用）
            if "changepoint" not in df.columns:
                df["changepoint"] = 0

            # 给每条数据加上故障类型标签（它来自哪个文件夹）
            df["fault_type"] = fault_type
            all_frames.append(df)
            print(f"[读取] {fault_type}/{file_name}  样本数: {len(df)}")

    # 如果一个有效文件都没读到，报错
    if not all_frames:
        raise RuntimeError("未读取到任何有效 CSV 文件")

    # 把所有DataFrame合并成一个大表（ignore_index=True表示重新编行号）
    return pd.concat(all_frames, ignore_index=True)


# ============================================================
# 函数2：数据清洗与预处理（"清洁工"开始打扫卫生）
# ============================================================
def preprocess(df):
    """
    数据清洗与预处理
    通俗解释：对合并后的大表格做8步清洗：
      1. 把时间字符串转成真正的时间格式
      2. 删掉时间无效的行
      3. 删掉完全重复的行
      4. 填充缺失值（先用前一行的值填，剩下的用平均值填）
      5. 把异常标签转成整数
      6. 按时间排序
      7. 只保留需要的列
      8. 保留列后再去重一次（确保最终干净）
    :param df: 合并后的原始DataFrame
    :return: 清洗后的DataFrame
    """
    # 步骤1：把时间字符串转成pandas的datetime格式（方便后续排序和查询）
    # errors="coerce"表示转不了的就设为空值（后面会删掉）
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # 步骤2：删掉时间无效的行（时间为空的行没法用）
    df = df.dropna(subset=["datetime"])

    # 步骤3：删除完全重复的行（所有列都一样的行，可能是采集重复了）
    before = len(df)
    df = df.drop_duplicates()
    print(f"[去重] 删除重复行: {before - len(df)}")

    # 步骤4：数值列缺失值处理
    # 先找出所有数值列
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # ffill()：向前填充（用前一行的值填当前行的空值，适合时序数据）
    df[numeric_cols] = df[numeric_cols].ffill()
    # 剩下还没填上的，用该列的平均值填充
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # 步骤5：把异常标签转成整数（可能读进来是浮点数，转成0/1整数）
    df["anomaly"] = df["anomaly"].astype(int)

    # 步骤6：按时间从早到晚排序（时序数据必须按时间排序）
    df = df.sort_values("datetime").reset_index(drop=True)

    # 步骤7：只保留业务需要的列（删掉changepoint等不用的列）
    keep = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[keep]

    # 步骤8：保留列后再去重一次（删列后可能出现新的重复行）
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if before - len(df) > 0:
        print(f"[二次去重] 字段筛选后删除重复行: {before - len(df)}")

    return df


# ============================================================
# 函数3：主函数（"清洁工"的工作流程）
# ============================================================
def main():
    # 确保输出文件夹存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("开始 SKAB 数据集预处理")
    print("=" * 60)

    # 步骤1：加载全部原始CSV（38个文件合并成一个大表）
    df_raw = load_all_csv(RAW_DATA_DIR)
    print(f"\n[合并完成] 原始总样本数: {len(df_raw)}")
    print(f"[故障类型分布]\n{df_raw['fault_type'].value_counts()}")
    print(f"[异常标签分布]\n{df_raw['anomaly'].value_counts()}")

    # 步骤2：数据清洗预处理
    df_processed = preprocess(df_raw)
    print(f"\n[预处理完成] 输出样本数: {len(df_processed)}")
    print(f"[字段列表] {list(df_processed.columns)}")

    # 打印数据质量统计（确认清洗效果）
    print(f"\n[预处理后数据质量统计]")
    print(f"  总样本数: {len(df_processed)}")
    print(f"  缺失值总数: {df_processed.isnull().sum().sum()}")  # 应该是0
    print(f"  重复行数: {df_processed.duplicated().sum()}")        # 应该是0
    print(f"  时间范围: {df_processed['datetime'].min()} ~ {df_processed['datetime'].max()}")
    print(f"  异常标签分布: {df_processed['anomaly'].value_counts().to_dict()}")
    print(f"  故障类型分布: {df_processed['fault_type'].value_counts().to_dict()}")

    # 步骤3：输出干净的CSV文件
    # encoding="utf-8-sig" 保证Excel打开中文不乱码（带BOM的UTF-8）
    df_processed.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n[输出文件] {OUTPUT_FILE}")
    print("=" * 60)
    print("预处理全部完成")
    print("=" * 60)


# 程序入口：直接运行这个文件时执行main函数
if __name__ == "__main__":
    main()
