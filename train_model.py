"""
模型训练脚本：训练异常检测模型 + 故障预测模型，并初始化数据库
运行方式：python train_model.py
"""
import os
import json
import pandas as pd

from database import setup_database
from models.anomaly_detection import train_anomaly_model
from models.fault_prediction import train_fault_model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_CSV = os.path.join(ROOT_DIR, "data", "processed", "out.csv")
METRICS_PATH = os.path.join(ROOT_DIR, "models", "saved", "model_metrics.json")


def main():
    print("=" * 60)
    print("步骤1：初始化数据库并导入数据")
    print("=" * 60)
    setup_database()

    print("\n" + "=" * 60)
    print("步骤2：加载预处理数据")
    print("=" * 60)
    df = pd.read_csv(PROCESSED_CSV)
    print(f"数据加载完成，样本数: {len(df)}")
    print(f"异常标签分布: {df['anomaly'].value_counts().to_dict()}")

    print("\n" + "=" * 60)
    print("步骤3：训练孤立森林异常检测模型")
    print("=" * 60)
    train_anomaly_model(df)

    print("\n" + "=" * 60)
    print("步骤4：训练随机森林故障预测模型")
    print("=" * 60)
    _, metrics = train_fault_model(df)

    # 保存模型评估指标
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n模型评估指标已保存到 {METRICS_PATH}")

    print("\n" + "=" * 60)
    print("全部模型训练与数据库初始化完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
