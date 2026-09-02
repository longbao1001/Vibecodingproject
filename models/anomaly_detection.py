# ============================================================
# 文件名：models/anomaly_detection.py
# 作用：核心技术一 —— 时序异常检测模块
# 通俗解释：
#   你可以把这个模块想象成一个"异常哨兵"。
#   它先学习设备正常运行时8路传感器的数据长什么样（训练阶段），
#   然后每当有新数据进来，它就判断"这个数据和正常情况差得远不远"。
#   如果差得远，就判定为异常，给一个异常分数（分数越大越异常）。
# 算法：IsolationForest（孤立森林）
# 原理大白话：
#   想象你在一群人中找一个外星人。正常人都长得差不多，要区分他们需要很多次提问；
#   但外星人长得很特别，可能问一两个问题就能把他"孤立"出来。
#   孤立森林就是随机问很多次问题（随机选一个特征、随机选一个阈值来切分数据），
#   异常点因为"与众不同"，平均需要更少的切分次数就能被孤立，所以更容易被发现。
# 课程技术方向：时序数据挖掘与异常检测
# ============================================================

import os
import joblib          # 用来保存和加载训练好的模型（就像把训练好的"哨兵"存档）
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest  # 孤立森林算法，来自scikit-learn库

# ---------- 路径配置 ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
MODEL_DIR = os.path.join(ROOT_DIR, "models", "saved")  # 模型保存文件夹
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_iforest.pkl")  # 异常检测模型文件路径

# 8路传感器列名（和database.py中一致，全项目统一）
SENSOR_COLS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"
]

# 传感器中文名字映射（前端展示根因分析时用，把英文名字翻译成中文让人看得懂）
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


# ============================================================
# 函数1：训练异常检测模型（让"哨兵"学习正常数据长什么样）
# ============================================================
def train_anomaly_model(df):
    """
    用正常工况数据训练孤立森林异常检测模型
    通俗解释：只给"哨兵"看正常设备的数据，让它记住正常的样子；
              以后看到不像正常的，就报警。
    :param df: 预处理后的完整DataFrame（包含所有传感器数据和异常标签）
    :return: 训练好的模型对象
    """
    # 确保模型保存文件夹存在（不存在就创建）
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 关键：只用正常样本（anomaly=0）来训练，学习正常工况的边界
    # 这是孤立森林的特点：不需要异常样本，只学正常的，偏离正常的就是异常
    normal_data = df[df["anomaly"] == 0][SENSOR_COLS]

    # 创建孤立森林模型
    # 参数解释：
    #   n_estimators=100：种100棵树（树越多越准，但越慢，100是常用的平衡点）
    #   contamination=0.1：预期异常比例约10%（用来调整判定阈值）
    #   random_state=42：随机种子，设成固定值保证每次训练结果一样（方便复现）
    #   n_jobs=-1：用所有CPU核心并行计算（跑得快）
    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    # 训练模型（让它学习正常数据的分布）
    model.fit(normal_data)

    # 保存训练好的模型到文件（以后直接加载，不用重新训练）
    joblib.dump(model, ANOMALY_MODEL_PATH)
    print(f"[异常检测] 孤立森林模型训练完成，正常训练样本: {len(normal_data)}，已保存到 {ANOMALY_MODEL_PATH}")
    return model


# ============================================================
# 函数2：加载已训练好的模型（从存档中取出"哨兵"）
# ============================================================
def load_anomaly_model():
    """
    加载训练好的异常检测模型
    如果模型文件不存在，报错提示先运行 train_model.py
    """
    if not os.path.exists(ANOMALY_MODEL_PATH):
        raise FileNotFoundError("异常检测模型不存在，请先运行 train_model.py")
    return joblib.load(ANOMALY_MODEL_PATH)


# ============================================================
# 函数3：执行异常检测（"哨兵"上岗，判断新数据是否异常）
# ============================================================
def detect_anomaly(model, sensor_values):
    """
    对单条或多条传感器数据进行异常检测
    通俗解释：把新数据拿给"哨兵"看，它告诉你"这个正常吗？异常分数多少？"
    :param model: 训练好的孤立森林模型
    :param sensor_values: 传感器数据，可以是字典（单条）或DataFrame（多条）
    :return: 字典（单条）或列表（多条），包含：
              is_anomaly: 是否异常（0=正常，1=异常）
              anomaly_score: 异常分数（越大越异常，正常数据通常为负数或接近0）
    """
    # 如果传入的是字典（单条数据），转成DataFrame格式（模型只认DataFrame）
    if isinstance(sensor_values, dict):
        X = pd.DataFrame([[sensor_values[c] for c in SENSOR_COLS]], columns=SENSOR_COLS)
    else:
        # 已经是DataFrame，只取需要的8列传感器数据
        X = sensor_values[SENSOR_COLS]

    # decision_function：计算异常分数
    # 注意：sklearn的decision_function返回值是"越大越正常"（正常样本分数高，异常样本分数低/负）
    # 所以我们取负数，变成"越大越异常"，符合直觉
    raw_score = model.decision_function(X)
    anomaly_score = (-raw_score).flatten()

    # predict：直接预测分类
    # sklearn的孤立森林返回 1=正常，-1=异常（和我们习惯的0/1不一样，需要转换）
    pred = model.predict(X).flatten()

    # 如果只有一条数据，返回字典格式
    if len(anomaly_score) == 1:
        return {
            "is_anomaly": int(pred[0] == -1),  # -1转成1（异常），1转成0（正常）
            "anomaly_score": round(float(anomaly_score[0]), 4)  # 保留4位小数
        }
    # 多条数据，返回列表格式
    return [
        {"is_anomaly": int(p == -1), "anomaly_score": round(float(s), 4)}
        for p, s in zip(pred, anomaly_score)
    ]
