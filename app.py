"""
Flask 主应用：面向车间设备运维的智能故障预警系统
启动方式：python app.py，浏览器访问 http://127.0.0.1:5000
"""
import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify

from database import get_conn, SENSOR_COLS
from models.anomaly_detection import load_anomaly_model, detect_anomaly
from models.fault_prediction import load_fault_model, predict_fault
from models.root_cause import analyze_root_cause, get_feature_importance

app = Flask(__name__)

# 数据库列名（Volume Flow RateRMS 在建表时转为 Volume_Flow_RateRMS）
DB_SENSOR_COLS = [c.replace(" ", "_") for c in SENSOR_COLS]


def row_to_sensor_values(row):
    """将数据库行转换为算法模块所需的传感器字典（统一字段名）"""
    return {sc: row[dc] for sc, dc in zip(SENSOR_COLS, DB_SENSOR_COLS)}

# 启动时加载模型（全局单例）
print("[系统] 正在加载算法模型...")
anomaly_model = load_anomaly_model()
fault_model = load_fault_model()
print("[系统] 模型加载完成，系统启动")


# ==================== 页面路由 ====================

@app.route("/")
def index():
    """设备总览页面"""
    return render_template("index.html")


@app.route("/monitor")
def monitor():
    """传感器监控页面"""
    return render_template("monitor.html")


@app.route("/alerts")
def alerts():
    """告警面板页面"""
    return render_template("alerts.html")


@app.route("/history")
def history():
    """历史记录页面"""
    return render_template("history.html")


# ==================== 数据查询 API ====================

@app.route("/api/devices")
def api_devices():
    """获取设备列表及当前状态统计"""
    conn = get_conn()
    devices = conn.execute("SELECT * FROM devices ORDER BY device_id").fetchall()
    result = []
    for d in devices:
        # 统计每台设备的总样本数和异常数
        stat = conn.execute(
            "SELECT COUNT(*) AS total, SUM(anomaly) AS abnormal FROM sensor_data WHERE device_id=?",
            (d["device_id"],)
        ).fetchone()
        # 最新一条数据
        latest = conn.execute(
            "SELECT * FROM sensor_data WHERE device_id=? ORDER BY timestamp DESC LIMIT 1",
            (d["device_id"],)
        ).fetchone()
        result.append({
            "device_id": d["device_id"],
            "device_name": d["device_name"],
            "location": d["location"],
            "fault_type": d["fault_type"],
            "status": d["status"],
            "total_samples": stat["total"],
            "abnormal_samples": stat["abnormal"] or 0,
            "latest_time": latest["timestamp"] if latest else None
        })
    conn.close()
    return jsonify({"code": 0, "data": result})


@app.route("/api/sensor_data")
def api_sensor_data():
    """获取指定设备的传感器时序数据"""
    device_id = request.args.get("device_id", "PUMP-001")
    limit = int(request.args.get("limit", 200))
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM sensor_data WHERE device_id=? ORDER BY timestamp DESC LIMIT ?",
        (device_id, limit)
    ).fetchall()
    conn.close()
    # 反转为时间正序
    data = [dict(r) for r in reversed(rows)]
    return jsonify({"code": 0, "data": data})


@app.route("/api/latest_data")
def api_latest_data():
    """获取指定设备最新一条传感器数据"""
    device_id = request.args.get("device_id", "PUMP-001")
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM sensor_data WHERE device_id=? ORDER BY timestamp DESC LIMIT 1",
        (device_id,)
    ).fetchone()
    conn.close()
    return jsonify({"code": 0, "data": dict(row) if row else None})


# ==================== 算法分析 API ====================

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    对单条传感器数据执行三大算法分析：异常检测 + 故障预测 + 根因分析
    请求体: {device_id, sensor_values(可选，不传则取数据库最新一条)}
    """
    body = request.get_json(force=True)
    device_id = body.get("device_id", "PUMP-001")
    sensor_values = body.get("sensor_values")

    # 未传入传感器值则从数据库取最新一条
    if sensor_values is None:
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM sensor_data WHERE device_id=? ORDER BY timestamp DESC LIMIT 1",
            (device_id,)
        ).fetchone()
        conn.close()
        if not row:
            return jsonify({"code": 1, "msg": "无传感器数据"}), 404
        sensor_values = row_to_sensor_values(row)
        timestamp = row["timestamp"]
    else:
        timestamp = body.get("timestamp", "实时采集")

    # 模块一：时序异常检测
    anomaly_result = detect_anomaly(anomaly_model, sensor_values)
    # 模块二：故障预测
    fault_result = predict_fault(fault_model, sensor_values)
    # 模块三：根因分析
    root_result = analyze_root_cause(fault_model, sensor_values, top_k=3)

    # 综合告警级别判定
    fault_prob = fault_result["fault_prob"]
    if fault_prob >= 0.7 or anomaly_result["is_anomaly"] == 1:
        severity = "高危"
        alert_type = "故障预警"
    elif fault_prob >= 0.4:
        severity = "中危"
        alert_type = "异常关注"
    else:
        severity = "正常"
        alert_type = "状态正常"

    return jsonify({
        "code": 0,
        "data": {
            "device_id": device_id,
            "timestamp": timestamp,
            "sensor_values": sensor_values,
            "anomaly_detection": anomaly_result,
            "fault_prediction": fault_result,
            "root_cause": root_result,
            "severity": severity,
            "alert_type": alert_type
        }
    })


@app.route("/api/scan", methods=["POST"])
def api_scan():
    """
    批量扫描指定设备最近N条数据，自动生成告警记录入库
    """
    body = request.get_json(force=True)
    device_id = body.get("device_id", "PUMP-001")
    limit = int(body.get("limit", 50))

    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM sensor_data WHERE device_id=? ORDER BY timestamp DESC LIMIT ?",
        (device_id, limit)
    ).fetchall()

    new_alerts = 0
    for row in reversed(rows):
        sensor_values = row_to_sensor_values(row)
        anomaly_result = detect_anomaly(anomaly_model, sensor_values)
        fault_result = predict_fault(fault_model, sensor_values)
        fault_prob = fault_result["fault_prob"]

        # 仅对高危/中危生成告警
        if fault_prob >= 0.4 or anomaly_result["is_anomaly"] == 1:
            severity = "高危" if (fault_prob >= 0.7 or anomaly_result["is_anomaly"] == 1) else "中危"
            root_result = analyze_root_cause(fault_model, sensor_values, top_k=3)
            message = (
                f"{device_id} 检测到异常，故障概率{fault_prob*100:.1f}%，"
                f"{root_result['conclusion']}"
            )
            # 避免重复告警：同一时间戳只存一条
            exists = conn.execute(
                "SELECT id FROM alerts WHERE device_id=? AND timestamp=?",
                (device_id, row["timestamp"])
            ).fetchone()
            if not exists:
                conn.execute(
                    """INSERT INTO alerts(device_id, timestamp, alert_type, severity, message,
                       anomaly_score, fault_prob, root_cause) VALUES(?,?,?,?,?,?,?,?)""",
                    (device_id, row["timestamp"], "故障预警", severity, message,
                     anomaly_result["anomaly_score"], fault_prob, root_result["conclusion"])
                )
                new_alerts += 1
    conn.commit()
    conn.close()
    return jsonify({"code": 0, "msg": f"扫描完成，新增 {new_alerts} 条告警", "new_alerts": new_alerts})


@app.route("/api/model_info")
def api_model_info():
    """获取模型信息：准确率、特征重要度"""
    metrics_path = os.path.join("models", "saved", "model_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
    feature_importance = get_feature_importance(fault_model)
    return jsonify({
        "code": 0,
        "data": {
            "accuracy": metrics.get("accuracy"),
            "confusion_matrix": metrics.get("confusion_matrix"),
            "feature_importance": feature_importance
        }
    })


# ==================== 告警与运维 API ====================

@app.route("/api/alerts")
def api_alerts():
    """获取告警列表，支持按设备和状态筛选"""
    device_id = request.args.get("device_id", "")
    status = request.args.get("status", "")
    sql = "SELECT * FROM alerts WHERE 1=1"
    params = []
    if device_id:
        sql += " AND device_id=?"
        params.append(device_id)
    if status:
        sql += " AND status=?"
        params.append(status)
    sql += " ORDER BY timestamp DESC LIMIT 500"
    conn = get_conn()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return jsonify({"code": 0, "data": [dict(r) for r in rows]})


@app.route("/api/alerts/handle", methods=["POST"])
def api_handle_alert():
    """处理告警：更新告警状态并创建运维记录"""
    body = request.get_json(force=True)
    alert_id = body.get("alert_id")
    action = body.get("action", "现场巡检确认，设备恢复正常")
    operator = body.get("operator", "运维人员")

    conn = get_conn()
    alert = conn.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
    if not alert:
        conn.close()
        return jsonify({"code": 1, "msg": "告警不存在"}), 404

    conn.execute("UPDATE alerts SET status='已处理' WHERE id=?", (alert_id,))
    conn.execute(
        "INSERT INTO maintenance(alert_id, device_id, action, operator) VALUES(?,?,?,?)",
        (alert_id, alert["device_id"], action, operator)
    )
    conn.commit()
    conn.close()
    return jsonify({"code": 0, "msg": "告警已处理，运维记录已创建"})


@app.route("/api/maintenance")
def api_maintenance():
    """获取运维处理记录"""
    conn = get_conn()
    rows = conn.execute("""
        SELECT m.*, a.message AS alert_message, a.severity
        FROM maintenance m LEFT JOIN alerts a ON m.alert_id = a.id
        ORDER BY m.create_time DESC
    """).fetchall()
    conn.close()
    return jsonify({"code": 0, "data": [dict(r) for r in rows]})


@app.route("/api/stats")
def api_stats():
    """首页统计数据"""
    conn = get_conn()
    total_data = conn.execute("SELECT COUNT(*) c FROM sensor_data").fetchone()["c"]
    total_alerts = conn.execute("SELECT COUNT(*) c FROM alerts").fetchone()["c"]
    unhandled = conn.execute("SELECT COUNT(*) c FROM alerts WHERE status='未处理'").fetchone()["c"]
    handled = conn.execute("SELECT COUNT(*) c FROM alerts WHERE status='已处理'").fetchone()["c"]
    high_risk = conn.execute("SELECT COUNT(*) c FROM alerts WHERE severity='高危' AND status='未处理'").fetchone()["c"]
    mid_risk = conn.execute("SELECT COUNT(*) c FROM alerts WHERE severity='中危' AND status='未处理'").fetchone()["c"]
    conn.close()
    return jsonify({"code": 0, "data": {
        "total_data": total_data,
        "total_alerts": total_alerts,
        "unhandled": unhandled,
        "handled": handled,
        "high_risk": high_risk,
        "mid_risk": mid_risk
    }})


@app.route("/api/dashboard")
def api_dashboard():
    """管理驾驶舱聚合数据：设备健康度、状态分布、各设备异常对比、告警级别分布"""
    conn = get_conn()
    devices = conn.execute("SELECT * FROM devices ORDER BY device_id").fetchall()
    device_health = []
    status_dist = {"healthy": 0, "warning": 0, "critical": 0}
    for d in devices:
        stat = conn.execute(
            "SELECT COUNT(*) total, SUM(anomaly) abnormal FROM sensor_data WHERE device_id=?",
            (d["device_id"],)
        ).fetchone()
        total = stat["total"] or 1
        abnormal = stat["abnormal"] or 0
        # 健康度 = 正常样本占比 * 100
        health_score = round((total - abnormal) / total * 100, 1)
        if health_score >= 85:
            health_status, status_dist["healthy"] = "健康", status_dist["healthy"] + 1
        elif health_score >= 60:
            health_status, status_dist["warning"] = "关注", status_dist["warning"] + 1
        else:
            health_status, status_dist["critical"] = "故障", status_dist["critical"] + 1
        # 该设备未处理告警数
        unhandled = conn.execute(
            "SELECT COUNT(*) c FROM alerts WHERE device_id=? AND status='未处理'",
            (d["device_id"],)
        ).fetchone()["c"]
        device_health.append({
            "device_id": d["device_id"],
            "device_name": d["device_name"],
            "location": d["location"],
            "total": total,
            "abnormal": abnormal,
            "health_score": health_score,
            "health_status": health_status,
            "unhandled_alerts": unhandled
        })

    # 告警级别分布
    severity_dist = {}
    for sev in ["高危", "中危"]:
        severity_dist[sev] = conn.execute(
            "SELECT COUNT(*) c FROM alerts WHERE severity=?", (sev,)
        ).fetchone()["c"]
    severity_dist["已处理"] = conn.execute(
        "SELECT COUNT(*) c FROM alerts WHERE status='已处理'"
    ).fetchone()["c"]

    # 最近5条告警
    recent_alerts = [dict(r) for r in conn.execute(
        "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 5"
    ).fetchall()]
    conn.close()

    # 全厂综合健康率
    avg_health = round(sum(d["health_score"] for d in device_health) / len(device_health), 1) if device_health else 0
    return jsonify({"code": 0, "data": {
        "device_health": device_health,
        "status_dist": status_dist,
        "severity_dist": severity_dist,
        "avg_health": avg_health,
        "recent_alerts": recent_alerts
    }})


@app.route("/api/alert_trend")
def api_alert_trend():
    """告警时间趋势：按日期聚合高危/中危告警数量"""
    conn = get_conn()
    rows = conn.execute("""
        SELECT substr(timestamp,1,10) AS day, severity, COUNT(*) AS cnt
        FROM alerts GROUP BY day, severity ORDER BY day
    """).fetchall()
    conn.close()
    days = sorted(set(r["day"] for r in rows))
    high = [sum(r["cnt"] for r in rows if r["day"] == d and r["severity"] == "高危") for d in days]
    mid = [sum(r["cnt"] for r in rows if r["day"] == d and r["severity"] == "中危") for d in days]
    return jsonify({"code": 0, "data": {"days": days, "high": high, "mid": mid}})


@app.route("/api/device_health_detail")
def api_device_health_detail():
    """单设备健康度详情：各传感器均值、正常/异常样本、用于雷达图"""
    device_id = request.args.get("device_id", "PUMP-001")
    conn = get_conn()
    # 正常样本各传感器均值
    normal_row = conn.execute(
        f"SELECT AVG({c}) m FROM sensor_data WHERE device_id=? AND anomaly=0",
        (device_id,)
    ).fetchall() if False else None
    # 用一条SQL取全部传感器均值
    avg_sql = "SELECT " + ", ".join([f"AVG({c.replace(' ','_')}) AS {c.replace(' ','_')}" for c in SENSOR_COLS])
    normal_avg = conn.execute(
        avg_sql + " FROM sensor_data WHERE device_id=? AND anomaly=0", (device_id,)
    ).fetchone()
    abnormal_avg = conn.execute(
        avg_sql + " FROM sensor_data WHERE device_id=? AND anomaly=1", (device_id,)
    ).fetchone()
    conn.close()
    return jsonify({"code": 0, "data": {
        "normal_avg": {c: round(normal_avg[c.replace(' ', '_')] or 0, 3) for c in SENSOR_COLS},
        "abnormal_avg": {c: round(abnormal_avg[c.replace(' ', '_')] or 0, 3) for c in SENSOR_COLS}
    }})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
