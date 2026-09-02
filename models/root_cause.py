# ============================================================
# 文件名：models/root_cause.py
# 作用：核心技术三 —— 故障根因分析模块
# 通俗解释：
#   你可以把这个模块想象成一个"故障侦探"。
#   当故障预测模型说"这台设备可能要出故障"时，
#   侦探就要进一步查："到底是哪个传感器/哪个部件导致的？该怎么修？"
#   它的办法是：看随机森林模型里，哪几个传感器对故障判断的"贡献度"最大，
#   贡献度最大的那几个，就是最可能的"罪魁祸首"，然后给出对应的维修建议。
# 算法：基于随机森林特征重要度（Feature Importance）
# 原理大白话：
#   随机森林里有很多棵决策树，每棵树在做判断时会问一系列问题
#   （比如"振动加速度大于0.5吗？"）。某个传感器被问得越多、
#   每次问完对判断结果影响越大，它的"重要度"就越高。
#   重要度高的传感器，就是对故障判断影响最大的，也就是最可能的根因。
# 课程技术方向：智能决策-故障根因分析
# ============================================================

import pandas as pd
from models.fault_prediction import SENSOR_COLS  # 导入8路传感器列名
from models.anomaly_detection import SENSOR_NAMES_CN  # 导入传感器中文名字


# ============================================================
# 函数1：分析故障根因（"侦探"查案，找出最可能的故障原因）
# ============================================================
def analyze_root_cause(model, sensor_values, top_k=3):
    """
    分析当前传感器数据中最可能导致故障的根因特征
    通俗解释：
      步骤1：从随机森林模型里拿出8路传感器的"重要度排名"
      步骤2：取排名前top_k（默认前3）的传感器作为"嫌疑犯"
      步骤3：根据传感器类型，给出对应的维修建议
      步骤4：综合成一段人类可读的结论
    :param model: 训练好的随机森林模型（里面有feature_importances_属性）
    :param sensor_values: 当前8路传感器值（字典格式）
    :param top_k: 返回贡献度最高的前K个特征，默认3个
    :return: 字典，包含：
              root_causes: TopK根因列表（每个含排名、传感器名、中文、贡献度、当前值、维修建议）
              conclusion: 综合结论文本
    """
    # ===== 步骤1：获取全局特征重要度 =====
    # model.feature_importances_ 是随机森林训练后自动计算好的
    # 是一个长度为8的数组，每个值对应一个传感器的重要度（加起来=1）
    importances = model.feature_importances_
    # 转成DataFrame并按重要度从高到低排序
    feat_imp = pd.DataFrame({
        "feature": SENSOR_COLS,       # 传感器英文名
        "importance": importances      # 重要度数值
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ===== 步骤2：取TopK根因，生成详细信息 =====
    # 把当前传感器值转成DataFrame（虽然这里主要用模型的全局重要度，但保留当前值供展示）
    X = pd.DataFrame([[sensor_values[c] for c in SENSOR_COLS]], columns=SENSOR_COLS)

    root_causes = []
    # 遍历排名前top_k的传感器
    for i, row in feat_imp.head(top_k).iterrows():
        feat = row["feature"]           # 传感器英文名
        imp = float(row["importance"])   # 重要度
        cur_val = float(sensor_values[feat])  # 当前传感器读数
        root_causes.append({
            "rank": i + 1,                                    # 排名（1,2,3...）
            "feature": feat,                                  # 传感器英文名
            "feature_cn": SENSOR_NAMES_CN.get(feat, feat),   # 传感器中文名
            "importance": round(imp, 4),                       # 重要度（保留4位小数）
            "current_value": round(cur_val, 4),                # 当前传感器读数
            "suggestion": _gen_suggestion(feat, cur_val)       # 对应的维修建议
        })

    # ===== 步骤3：生成综合结论文本 =====
    # 取排名第一的传感器作为主要根因，生成一段人类可读的结论
    top_feat = root_causes[0]
    conclusion = (
        f"根因分析：{top_feat['feature_cn']}（{top_feat['feature']}）贡献度最高"
        f"（{top_feat['importance']*100:.1f}%），建议优先检查该传感器对应部件。"
    )

    return {
        "root_causes": root_causes,
        "conclusion": conclusion
    }


# ============================================================
# 函数2：根据传感器类型生成维修建议（"侦探"的维修手册）
# ============================================================
def _gen_suggestion(feat, value):
    """
    根据传感器类型生成对应的运维建议
    通俗解释：这是一个"故障→维修建议"的对照表，
              比如振动异常就建议查轴承，流量异常就建议查管路。
    :param feat: 传感器英文名
    :param value: 当前传感器值（目前未使用，未来可以根据值的大小给出更具体的建议）
    :return: 维修建议文本
    """
    # 传感器类型 → 维修建议 的映射表
    suggestions = {
        "Accelerometer1RMS": "检查1号振动测点轴承磨损、松动情况",
        "Accelerometer2RMS": "检查2号振动测点轴承磨损、对中情况",
        "Current": "检查电机负载是否异常增大、是否存在堵转",
        "Pressure": "检查管路压力是否异常，排查泄漏、堵塞、阀门故障",
        "Temperature": "检查设备散热、冷却系统是否正常",
        "Thermocouple": "检查热电偶测点温升，排查摩擦过热",
        "Voltage": "检查供电电压稳定性，排查电气故障",
        "Volume Flow RateRMS": "检查流量是否异常，排查管路堵塞或泄漏",
    }
    # 如果传感器不在表里，返回通用建议
    return suggestions.get(feat, "检查对应传感器及关联部件运行状态")


# ============================================================
# 函数3：获取全部特征重要度（前端画图表用）
# ============================================================
def get_feature_importance(model):
    """
    获取全部8路传感器的特征重要度
    通俗解释：把8个传感器的重要度都列出来，前端用来画横向条形图，
              让用户一眼看到哪个传感器对故障判断影响最大。
    :param model: 训练好的随机森林模型
    :return: 列表，每个元素是一个字典（传感器英文名、中文名、重要度）
    """
    importances = model.feature_importances_
    return [
        {
            "feature": col,                                        # 传感器英文名
            "feature_cn": SENSOR_NAMES_CN.get(col, col),          # 传感器中文名
            "importance": round(float(imp), 4)                     # 重要度
        }
        for col, imp in zip(SENSOR_COLS, importances)
    ]
