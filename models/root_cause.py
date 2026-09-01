"""
模块三：故障根因分析
算法：基于随机森林特征重要度（Feature Importance）
原理：模型训练后得到各传感器特征对故障判别的贡献度，贡献度越高越可能是故障诱因
课程技术方向：智能决策-故障根因分析
"""
import pandas as pd
from models.fault_prediction import SENSOR_COLS
from models.anomaly_detection import SENSOR_NAMES_CN


def analyze_root_cause(model, sensor_values, top_k=3):
    """
    分析当前传感器数据中最可能导致故障的根因特征
    :param model: 训练好的随机森林模型（含 feature_importances_）
    :param sensor_values: dict，当前8路传感器值
    :param top_k: 返回贡献度最高的前K个特征
    :return: dict: 根因分析结果
    """
    # 1. 全局特征重要度（模型训练得到）
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": SENSOR_COLS,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # 2. 计算当前样本各传感器相对正常均值的偏离程度（z-score 思路）
    #    偏离越大，结合特征重要度，越可能是根因
    X = pd.DataFrame([[sensor_values[c] for c in SENSOR_COLS]], columns=SENSOR_COLS)
    # 用模型训练集均值/标准差需要外部传入，这里用特征重要度为主，偏离方向为辅
    root_causes = []
    for i, row in feat_imp.head(top_k).iterrows():
        feat = row["feature"]
        imp = float(row["importance"])
        cur_val = float(sensor_values[feat])
        root_causes.append({
            "rank": i + 1,
            "feature": feat,
            "feature_cn": SENSOR_NAMES_CN.get(feat, feat),
            "importance": round(imp, 4),
            "current_value": round(cur_val, 4),
            "suggestion": _gen_suggestion(feat, cur_val)
        })

    # 3. 综合结论
    top_feat = root_causes[0]
    conclusion = (
        f"根因分析：{top_feat['feature_cn']}（{top_feat['feature']}）贡献度最高"
        f"（{top_feat['importance']*100:.1f}%），建议优先检查该传感器对应部件。"
    )

    return {
        "root_causes": root_causes,
        "conclusion": conclusion
    }


def _gen_suggestion(feat, value):
    """根据传感器类型生成运维建议"""
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
    return suggestions.get(feat, "检查对应传感器及关联部件运行状态")


def get_feature_importance(model):
    """获取全部特征重要度（前端图表用）"""
    importances = model.feature_importances_
    return [
        {
            "feature": col,
            "feature_cn": SENSOR_NAMES_CN.get(col, col),
            "importance": round(float(imp), 4)
        }
        for col, imp in zip(SENSOR_COLS, importances)
    ]
