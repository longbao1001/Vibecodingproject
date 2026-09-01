"""
模块二：设备故障预测
算法：RandomForestClassifier（随机森林分类器）
原理：基于8路传感器特征监督学习，预测设备发生故障的概率，实现故障预警
课程技术方向：机器学习故障预测模型
"""
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "models", "saved")
FAULT_MODEL_PATH = os.path.join(MODEL_DIR, "fault_rf.pkl")

SENSOR_COLS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"
]


def train_fault_model(df):
    """
    训练随机森林故障预测模型
    :param df: 预处理后的完整 DataFrame
    :return: 模型 + 评估报告
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    X = df[SENSOR_COLS]
    y = df["anomaly"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 模型评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"[故障预测] 随机森林模型训练完成，测试集准确率: {acc:.4f}")
    print(f"[故障预测] 混淆矩阵: {cm}")

    joblib.dump(model, FAULT_MODEL_PATH)
    print(f"[故障预测] 模型已保存到 {FAULT_MODEL_PATH}")
    return model, {"accuracy": round(acc, 4), "report": report, "confusion_matrix": cm}


def load_fault_model():
    """加载训练好的故障预测模型"""
    if not os.path.exists(FAULT_MODEL_PATH):
        raise FileNotFoundError("故障预测模型不存在，请先运行 train_model.py")
    return joblib.load(FAULT_MODEL_PATH)


def predict_fault(model, sensor_values):
    """
    预测设备故障概率
    :param model: 随机森林模型
    :param sensor_values: dict 或 DataFrame
    :return: dict: fault_label(0正常/1故障), fault_prob(故障概率)
    """
    if isinstance(sensor_values, dict):
        X = pd.DataFrame([[sensor_values[c] for c in SENSOR_COLS]], columns=SENSOR_COLS)
    else:
        X = sensor_values[SENSOR_COLS]

    # predict_proba: [正常概率, 故障概率]
    proba = model.predict_proba(X)
    label = model.predict(X)

    if len(proba) == 1:
        fault_prob = float(proba[0][1])
        return {
            "fault_label": int(label[0]),
            "fault_prob": round(fault_prob, 4),
            "normal_prob": round(float(proba[0][0]), 4)
        }
    return [
        {
            "fault_label": int(l),
            "fault_prob": round(float(p[1]), 4),
            "normal_prob": round(float(p[0]), 4)
        }
        for l, p in zip(label, proba)
    ]
