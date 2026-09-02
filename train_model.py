# ============================================================
# 文件名：train_model.py
# 作用：模型训练总脚本 —— 一键完成数据库初始化 + 两个模型训练
# 通俗解释：
#   你可以把这个文件想象成一个"开工总指挥"。
#   它按顺序做4件事：
#     步骤1：建仓库（初始化数据库，把46669条数据搬进去）
#     步骤2：搬货物（加载预处理好的数据）
#     步骤3：训练异常哨兵（训练孤立森林异常检测模型）
#     步骤4：训练故障预言家（训练随机森林故障预测模型）
#   全部做完后，系统就可以启动使用了。
# 运行方式：python train_model.py
# 注意：运行前需要先运行 preprocess.py 生成预处理数据 data/processed/out.csv
# ============================================================

import os
import json
import pandas as pd

# 导入我们自己写的模块
from database import setup_database  # 数据库一键初始化
from models.anomaly_detection import train_anomaly_model  # 训练异常检测模型
from models.fault_prediction import train_fault_model      # 训练故障预测模型

# ---------- 路径配置 ----------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_CSV = os.path.join(ROOT_DIR, "data", "processed", "out.csv")  # 预处理后的数据
METRICS_PATH = os.path.join(ROOT_DIR, "models", "saved", "model_metrics.json")  # 模型评估指标保存路径


def main():
    """主函数：按顺序执行4个步骤"""

    # ===== 步骤1：初始化数据库并导入数据 =====
    print("=" * 60)
    print("步骤1：初始化数据库并导入数据")
    print("=" * 60)
    # 调用database.py中的setup_database，一键完成建表+设备档案+数据导入
    setup_database()

    # ===== 步骤2：加载预处理数据 =====
    print("\n" + "=" * 60)
    print("步骤2：加载预处理数据")
    print("=" * 60)
    # 用pandas读取CSV文件（就像用Excel打开表格）
    df = pd.read_csv(PROCESSED_CSV)
    print(f"数据加载完成，样本数: {len(df)}")
    # 打印异常标签分布（正常多少条，异常多少条）
    print(f"异常标签分布: {df['anomaly'].value_counts().to_dict()}")

    # ===== 步骤3：训练孤立森林异常检测模型 =====
    print("\n" + "=" * 60)
    print("步骤3：训练孤立森林异常检测模型")
    print("=" * 60)
    # 只用正常样本训练，学习正常工况边界
    train_anomaly_model(df)

    # ===== 步骤4：训练随机森林故障预测模型 =====
    print("\n" + "=" * 60)
    print("步骤4：训练随机森林故障预测模型")
    print("=" * 60)
    # 训练模型，返回模型对象和评估指标
    _, metrics = train_fault_model(df)

    # 保存模型评估指标到JSON文件（前端/api/model_info接口会读取这个文件展示模型效果）
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        # ensure_ascii=False 保证中文能正常写入（不转成\uXXXX编码）
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n模型评估指标已保存到 {METRICS_PATH}")

    # ===== 全部完成 =====
    print("\n" + "=" * 60)
    print("全部模型训练与数据库初始化完成")
    print("=" * 60)
    print("接下来可以运行 python app.py 启动系统，")
    print("或者双击 start.bat 一键启动。")


# 程序入口：直接运行这个文件时执行main函数
if __name__ == "__main__":
    main()
