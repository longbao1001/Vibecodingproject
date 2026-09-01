"""
模块一：时序异常检测
算法：IsolationForest（孤立森林）
原理：通过随机划分特征空间，异常点更容易被孤立（划分路径更短），适合多传感器高维时序异常检测
课程技术方向：时序数据挖掘与异常检测
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "models", "saved")
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_iforest.pkl")

SENSOR_COLS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"
]

# 传感器中文名称映射（根因分析展示用）
SENSOR_NAMES_CN = {
    "Accelerometer1RMS": "1号振动加速度",
    "Accelerometer2RMS": "2号振动加速度",
    "Current": "电机电流",
    "Pressure": "压力",
    "Temperature": "温度",
    "Thermocouple": "热电偶温度",
    "Voltage": "电机电压",
    "Volume Flow RateRMS": "流量",
}


def train_anomaly_model(df):
    """
    用正常工况数据训练孤立森林异常检测模型
    :param df: 预处理后的完整 DataFrame
    :return: 训练好的模型
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    # 仅用正常样本训练（anomaly=0），学习正常工况边界
    normal_data = df[df["anomaly"] == 0][SENSOR_COLS]
    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(normal_data)
    joblib.dump(model, ANOMALY_MODEL_PATH)
    print(f"[异常检测] 孤立森林模型训练完成，正常训练样本: {len(normal_data)}，已保存到 {ANOMALY_MODEL_PATH}")
    return model


def load_anomaly_model():
    """加载训练好的异常检测模型"""
    if not os.path.exists(ANOMALY_MODEL_PATH):
        raise FileNotFoundError("异常检测模型不存在，请先运行 train_model.py")
    return joblib.load(ANOMALY_MODEL_PATH)


def detect_anomaly(model, sensor_values):
    """
    对单条/多条传感器数据进行异常检测
    :param model: 孤立森林模型
    :param sensor_values: dict 或 DataFrame，包含8路传感器值
    :return: dict: is_anomaly(是否异常), anomaly_score(异常分数，越大越异常)
    """
    if isinstance(sensor_values, dict):
        X = pd.DataFrame([[sensor_values[c] for c in SENSOR_COLS]], columns=SENSOR_COLS)
    else:
        X = sensor_values[SENSOR_COLS]

    # decision_function: 越大越正常，取负后越大越异常
    raw_score = model.decision_function(X)
    anomaly_score = (-raw_score).flatten()
    # predict: 1=正常, -1=异常
    pred = model.predict(X).flatten()

    if len(anomaly_score) == 1:
        return {
            "is_anomaly": int(pred[0] == -1),
            "anomaly_score": round(float(anomaly_score[0]), 4)
        }
    return [
        {"is_anomaly": int(p == -1), "anomaly_score": round(float(s), 4)}
        for p, s in zip(pred, anomaly_score)
    ]
