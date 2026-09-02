# ============================================================
# 文件名：models/fault_prediction.py
# 作用：核心技术二 —— 设备故障预测模块
# 通俗解释：
#   你可以把这个模块想象成一个"故障预言家"。
#   它看过大量"正常数据→没故障"和"异常数据→有故障"的例子（训练阶段），
#   然后给它一组新的传感器数据，它就能预测"这台设备有多大可能要出故障"。
# 算法：RandomForestClassifier（随机森林分类器）
# 原理大白话：
#   想象你要判断一个水果是不是苹果。你找100个朋友，每个人只看一个特征
#   （比如颜色、形状、大小、重量），然后每个人投一票。最后数票数，
#   多数人说是苹果，那就判定是苹果。
#   随机森林就是这个思路：种很多棵决策树（每棵树就是一个"朋友"），
#   每棵树根据不同的特征做判断，最后所有树投票，得票多的就是预测结果。
#   这样比单棵树更准，因为单棵树可能"偏见"，多棵树投票能减少错误。
# 课程技术方向：机器学习故障预测模型
# ============================================================

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # 模型评估工具

# ---------- 路径配置 ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "models", "saved")
FAULT_MODEL_PATH = os.path.join(MODEL_DIR, "fault_rf.pkl")  # 故障预测模型文件路径

# 8路传感器列名（全项目统一）
SENSOR_COLS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"
]


# ============================================================
# 函数1：训练故障预测模型（让"预言家"学习历史数据）
# ============================================================
def train_fault_model(df):
    """
    训练随机森林故障预测模型
    通俗解释：给"预言家"看大量历史数据（8路传感器值→是否故障），
              让它学会"什么样的数据组合意味着要出故障"。
    :param df: 预处理后的完整DataFrame
    :return: (模型对象, 评估指标字典)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # X = 特征（8路传感器数据），y = 标签（0=正常，1=故障）
    X = df[SENSOR_COLS]
    y = df["anomaly"]

    # 划分训练集和测试集（80%用来训练，20%用来考试）
    # test_size=0.2：20%做测试集
    # random_state=42：固定随机种子，保证每次划分结果一样
    # stratify=y：按标签分层抽样，保证训练集和测试集中正常/故障比例一样
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 创建随机森林分类器
    # 参数解释：
    #   n_estimators=100：种100棵决策树（树越多越准，但越慢）
    #   max_depth=12：每棵树最多长12层（限制树的深度，防止过拟合——死记硬背训练数据）
    #   random_state=42：固定随机种子，结果可复现
    #   n_jobs=-1：用所有CPU核心并行训练
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    # 训练模型（用训练集数据"学习"）
    model.fit(X_train, y_train)

    # ===== 模型评估（用测试集"考试"，看模型学得怎么样）=====
    y_pred = model.predict(X_test)  # 用模型预测测试集的结果
    acc = accuracy_score(y_test, y_pred)  # 准确率：预测对的比例（本项目约93.04%）
    # 分类报告：精确率、召回率、F1分数等详细指标
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    # 混淆矩阵：[[真正例预测为正常, 真正例预测为故障], [真故障预测为正常, 真故障预测为故障]]
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"[故障预测] 随机森林模型训练完成，测试集准确率: {acc:.4f}")
    print(f"[故障预测] 混淆矩阵: {cm}")

    # 保存模型到文件
    joblib.dump(model, FAULT_MODEL_PATH)
    print(f"[故障预测] 模型已保存到 {FAULT_MODEL_PATH}")

    # 返回模型和评估指标
    return model, {"accuracy": round(acc, 4), "report": report, "confusion_matrix": cm}


# ============================================================
# 函数2：加载已训练好的模型
# ============================================================
def load_fault_model():
    """加载训练好的故障预测模型"""
    if not os.path.exists(FAULT_MODEL_PATH):
        raise FileNotFoundError("故障预测模型不存在，请先运行 train_model.py")
    return joblib.load(FAULT_MODEL_PATH)


# ============================================================
# 函数3：执行故障预测（"预言家"上岗，预测设备故障概率）
# ============================================================
def predict_fault(model, sensor_values):
    """
    预测设备故障概率
    通俗解释：把当前传感器数据给"预言家"看，它告诉你"有多大可能要出故障"。
    :param model: 训练好的随机森林模型
    :param sensor_values: 传感器数据，字典（单条）或DataFrame（多条）
    :return: 字典（单条）或列表（多条），包含：
              fault_label: 故障标签（0=正常，1=故障）
              fault_prob: 故障概率（0到1之间，越大越可能故障，比如0.85就是85%概率）
              normal_prob: 正常概率
    """
    # 字典转DataFrame
    if isinstance(sensor_values, dict):
        X = pd.DataFrame([[sensor_values[c] for c in SENSOR_COLS]], columns=SENSOR_COLS)
    else:
        X = sensor_values[SENSOR_COLS]

    # predict_proba：预测每个类别的概率
    # 返回 [[正常概率, 故障概率], ...]，比如 [[0.1, 0.9]] 表示10%正常，90%故障
    proba = model.predict_proba(X)
    # predict：直接预测分类（0或1）
    label = model.predict(X)

    # 单条数据返回字典
    if len(proba) == 1:
        fault_prob = float(proba[0][1])  # 取故障类别的概率
        return {
            "fault_label": int(label[0]),
            "fault_prob": round(fault_prob, 4),       # 故障概率，保留4位小数
            "normal_prob": round(float(proba[0][0]), 4)  # 正常概率
        }
    # 多条数据返回列表
    return [
        {
            "fault_label": int(l),
            "fault_prob": round(float(p[1]), 4),
            "normal_prob": round(float(p[0]), 4)
        }
        for l, p in zip(label, proba)
    ]
