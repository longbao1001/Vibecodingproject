# ============================================================
# 文件名：app.py
# 作用：整个网站的"总开关"和"前台接待员"
# 通俗解释：
#   你可以把这个文件想象成一家餐厅的大堂经理。
#   顾客（浏览器）来了，说"我要看菜单"，经理就带他去菜单区；
#   顾客说"我要下单"，经理就通知后厨（算法模块）做菜；
#   顾客说"我要结账"，经理就去收银台（数据库）查账。
#   这个文件就是负责"接收请求、分派任务、返回结果"的。
# 启动方式：双击 start.bat，或者在命令行输入 python app.py
# 启动后打开浏览器访问：http://127.0.0.1:5000
# ============================================================

# ---------- 导入工具包 ----------
# os：操作系统工具，用来读写文件、判断文件是否存在
import os
# json：JSON格式工具，用来把数据转成网页能看懂的格式
import json
# pandas：数据处理工具，像Excel一样处理表格数据
import pandas as pd
# Flask：网站框架工具，提供了搭建网站的所有基础功能
# render_template：渲染网页模板（把数据填进HTML模板里）
# request：获取浏览器发来的请求内容（比如用户选了哪台设备）
# jsonify：把数据转成JSON格式返回给浏览器
from flask import Flask, render_template, request, jsonify

# 从我们自己写的database.py文件里导入两个工具：
# get_conn：连接数据库（就像打开仓库大门）
# SENSOR_COLS：8个传感器的名字列表
from database import get_conn, SENSOR_COLS
# 从三个算法模块里导入各自的功能
from models.anomaly_detection import load_anomaly_model, detect_anomaly
from models.fault_prediction import load_fault_model, predict_fault
from models.root_cause import analyze_root_cause, get_feature_importance

# ---------- 创建网站应用 ----------
# 这一行就像"注册一家公司"，app就是这家公司的名字，后面所有功能都挂在它上面
app = Flask(__name__)

# 数据库里的列名和算法模块里的传感器名字有个小区别：
# 数据库里空格被换成了下划线（比如 "Volume Flow RateRMS" 变成了 "Volume_Flow_RateRMS"）
# 这里做一个转换，方便后面从数据库取数据后传给算法模块
DB_SENSOR_COLS = [c.replace(" ", "_") for c in SENSOR_COLS]


# ---------- 一个小工具函数 ----------
def row_to_sensor_values(row):
    """
    把数据库里查出来的一行数据，转换成算法模块能看懂的字典格式
    通俗解释：就像把仓库里的货物标签换成后厨能看懂的菜单编号
    """
    return {sc: row[dc] for sc, dc in zip(SENSOR_COLS, DB_SENSOR_COLS)}


# ---------- 启动时预加载模型 ----------
# 网站一启动就把三个算法模型加载到内存里，这样每次用户请求时不用重新加载，速度更快
# 就像餐厅开门前先把菜谱备好，客人来了直接点菜，不用现翻书
print("[系统] 正在加载算法模型...")
anomaly_model = load_anomaly_model()   # 加载异常检测模型
fault_model = load_fault_model()       # 加载故障预测模型
print("[系统] 模型加载完成，系统启动")


# ============================================================
# 第一部分：页面路由（用户访问哪个网址，就显示哪个网页）
# 通俗解释：就像餐厅的不同区域，客人走到门口说"我要去包间"，服务员就带他去包间
# ============================================================

# 访问首页 http://127.0.0.1:5000/ 时，显示设备总览页面
@app.route("/")
def index():
    """设备总览页面：展示所有设备的健康状态、告警统计等"""
    return render_template("index.html")


# 访问 http://127.0.0.1:5000/monitor 时，显示传感器监控页面
@app.route("/monitor")
def monitor():
    """传感器监控页面：实时显示8路传感器数据、仪表盘、智能诊断结果"""
    return render_template("monitor.html")


# 访问 http://127.0.0.1:5000/alerts 时，显示告警面板页面
@app.route("/alerts")
def alerts():
    """告警面板页面：展示所有告警记录、支持筛选和处置操作"""
    return render_template("alerts.html")


# 访问 http://127.0.0.1:5000/history 时，显示运维记录页面
@app.route("/history")
def history():
    """运维记录页面：展示所有历史运维处置记录"""
    return render_template("history.html")


# ============================================================
# 第二部分：数据查询API（前端网页通过这些接口从数据库取数据）
# 通俗解释：就像餐厅的"传菜口"，前端网页说"给我拿一下3号桌的菜单"，
#          这里就去数据库查出来，打包好递回去
# ============================================================

# 获取所有设备的列表和每台设备的统计信息
@app.route("/api/devices")
def api_devices():
    """
    获取设备列表及当前状态统计
    返回：每台设备的ID、名字、位置、故障类型、状态、总样本数、异常数、最新数据时间
    """
    conn = get_conn()  # 打开数据库连接（打开仓库大门）
    # 查出所有设备，按设备ID排序
    devices = conn.execute("SELECT * FROM devices ORDER BY device_id").fetchall()
    result = []
    for d in devices:
        # 统计这台设备一共有多少条数据、其中多少条是异常的
        stat = conn.execute(
            "SELECT COUNT(*) AS total, SUM(anomaly) AS abnormal FROM sensor_data WHERE device_id=?",
            (d["device_id"],)
        ).fetchone()
        # 查出这台设备最新的一条数据
        latest = conn.execute(
            "SELECT * FROM sensor_data WHERE device_id=? ORDER BY timestamp DESC LIMIT 1",
            (d["device_id"],)
        ).fetchone()
        # 把这台设备的所有信息打包成一个字典，加入结果列表
        result.append({
            "device_id": d["device_id"],           # 设备编号，如 PUMP-001
            "device_name": d["device_name"],       # 设备名字，如 1号冷却水泵
            "location": d["location"],              # 设备位置，如 一号车间A区
            "fault_type": d["fault_type"],          # 对应故障场景类型
            "status": d["status"],                   # 设备运行状态
            "total_samples": stat["total"],          # 该设备总数据条数
            "abnormal_samples": stat["abnormal"] or 0,  # 异常数据条数
            "latest_time": latest["timestamp"] if latest else None  # 最新数据时间
        })
    conn.close()  # 关闭数据库连接（关上仓库大门，省电）
    return jsonify({"code": 0, "data": result})  # 把结果转成JSON返回给前端


# 获取指定设备的传感器时序数据（最近N条）
@app.route("/api/sensor_data")
def api_sensor_data():
    """
    获取指定设备的传感器时序数据
    参数：device_id（设备编号）、limit（取多少条，默认200条）
    返回：按时间正序排列的传感器数据列表
    """
    # 从请求参数中获取设备编号，默认取PUMP-001
    device_id = request.args.get("device_id", "PUMP-001")
    # 获取要取多少条数据，默认200条
    limit = int(request.args.get("limit", 200))
    conn = get_conn()
    # 从数据库查出这台设备最近的N条数据（按时间倒序）
    rows = conn.execute(
        "SELECT * FROM sensor_data WHERE device_id=? ORDER BY timestamp DESC LIMIT ?",
        (device_id, limit)
    ).fetchall()
    conn.close()
    # 数据库查出来是倒序（最新的在前面），这里反转一下变成正序（最早的在前面）
    # 这样画折线图时时间从左到右是递增的
    data = [dict(r) for r in reversed(rows)]
    return jsonify({"code": 0, "data": data})


# 获取指定设备最新的一条传感器数据
@app.route("/api/latest_data")
def api_latest_data():
    """获取指定设备最新一条传感器数据，用于仪表盘实时显示"""
    device_id = request.args.get("device_id", "PUMP-001")
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM sensor_data WHERE device_id=? ORDER BY timestamp DESC LIMIT 1",
        (device_id,)
    ).fetchone()
    conn.close()
    return jsonify({"code": 0, "data": dict(row) if row else None})


# ============================================================
# 第三部分：算法分析API（调用三大算法模块做智能诊断）
# 通俗解释：就像餐厅的"后厨"，前端把食材（传感器数据）送过来，
#          这里调用三个厨师（三个算法）分别做菜，最后把三道菜一起端出去
# ============================================================

# 对单条传感器数据执行三大算法分析（这是最核心的接口）
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    对单条传感器数据执行三大算法分析：
      第一步：异常检测（判断当前数据是否偏离正常范围）
      第二步：故障预测（预测设备发生故障的概率）
      第三步：根因分析（找出最可能导致故障的传感器/部件）
    最后综合三个结果，给出告警级别（高危/中危/正常）
    请求体：{device_id: 设备编号, sensor_values: 传感器值（可选，不传就取数据库最新一条）}
    """
    # 获取前端发来的JSON数据
    body = request.get_json(force=True)
    device_id = body.get("device_id", "PUMP-001")
    sensor_values = body.get("sensor_values")

    # 如果前端没有传传感器数据，就从数据库里取这台设备最新的一条
    if sensor_values is None:
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM sensor_data WHERE device_id=? ORDER BY timestamp DESC LIMIT 1",
            (device_id,)
        ).fetchone()
        conn.close()
        if not row:
            return jsonify({"code": 1, "msg": "无传感器数据"}), 404
        # 把数据库行转换成算法能看懂的格式
        sensor_values = row_to_sensor_values(row)
        timestamp = row["timestamp"]
    else:
        timestamp = body.get("timestamp", "实时采集")

    # ===== 调用算法一：孤立森林异常检测 =====
    # 输入：传感器数据；输出：是否异常 + 异常分数（分数越大越异常）
    anomaly_result = detect_anomaly(anomaly_model, sensor_values)

    # ===== 调用算法二：随机森林故障预测 =====
    # 输入：传感器数据；输出：故障标签 + 故障概率（0到1之间，越大越可能故障）
    fault_result = predict_fault(fault_model, sensor_values)

    # ===== 调用算法三：特征重要度根因分析 =====
    # 输入：故障预测模型 + 传感器数据；输出：Top3最可能的根因传感器 + 运维建议
    root_result = analyze_root_cause(fault_model, sensor_values, top_k=3)

    # ===== 综合告警级别判定 =====
    # 规则：
    #   故障概率 >= 0.7，或者异常检测判定为异常 → 高危
    #   故障概率 >= 0.4 → 中危
    #   其他情况 → 正常
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

    # 把所有结果打包返回给前端
    return jsonify({
        "code": 0,
        "data": {
            "device_id": device_id,
            "timestamp": timestamp,
            "sensor_values": sensor_values,
            "anomaly_detection": anomaly_result,   # 异常检测结果
            "fault_prediction": fault_result,       # 故障预测结果
            "root_cause": root_result,               # 根因分析结果
            "severity": severity,                     # 综合告警级别
            "alert_type": alert_type                  # 告警类型
        }
    })


# 批量扫描指定设备最近N条数据，自动生成告警记录存入数据库
@app.route("/api/scan", methods=["POST"])
def api_scan():
    """
    批量扫描功能：对指定设备最近N条数据逐条跑算法，
    发现异常/故障就自动生成告警记录存入数据库
    通俗解释：就像质检员批量检查一批产品，发现不合格的就贴个标签放进不合格区
    """
    body = request.get_json(force=True)
    device_id = body.get("device_id", "PUMP-001")
    limit = int(body.get("limit", 50))  # 默认扫描最近50条

    conn = get_conn()
    # 查出最近N条数据
    rows = conn.execute(
        "SELECT * FROM sensor_data WHERE device_id=? ORDER BY timestamp DESC LIMIT ?",
        (device_id, limit)
    ).fetchall()

    new_alerts = 0  # 计数器：本次扫描新增了多少条告警
    # 反转成时间正序，逐条分析
    for row in reversed(rows):
        sensor_values = row_to_sensor_values(row)
        # 跑异常检测和故障预测
        anomaly_result = detect_anomaly(anomaly_model, sensor_values)
        fault_result = predict_fault(fault_model, sensor_values)
        fault_prob = fault_result["fault_prob"]

        # 只有中危及以上才生成告警（正常的不存，避免数据库被刷屏）
        if fault_prob >= 0.4 or anomaly_result["is_anomaly"] == 1:
            # 判断是高危还是中危
            severity = "高危" if (fault_prob >= 0.7 or anomaly_result["is_anomaly"] == 1) else "中危"
            # 跑根因分析，拿到结论
            root_result = analyze_root_cause(fault_model, sensor_values, top_k=3)
            # 拼接告警消息文本
            message = (
                f"{device_id} 检测到异常，故障概率{fault_prob*100:.1f}%，"
                f"{root_result['conclusion']}"
            )
            # 避免重复告警：同一设备同一时间戳只存一条
            exists = conn.execute(
                "SELECT id FROM alerts WHERE device_id=? AND timestamp=?",
                (device_id, row["timestamp"])
            ).fetchone()
            if not exists:
                # 插入告警记录到数据库
                conn.execute(
                    """INSERT INTO alerts(device_id, timestamp, alert_type, severity, message,
                       anomaly_score, fault_prob, root_cause) VALUES(?,?,?,?,?,?,?,?)""",
                    (device_id, row["timestamp"], "故障预警", severity, message,
                     anomaly_result["anomaly_score"], fault_prob, root_result["conclusion"])
                )
                new_alerts += 1
    conn.commit()  # 提交事务（把所有插入操作真正写入数据库文件）
    conn.close()
    return jsonify({"code": 0, "msg": f"扫描完成，新增 {new_alerts} 条告警", "new_alerts": new_alerts})


# 获取模型信息（准确率、混淆矩阵、特征重要度），用于前端展示模型效果
@app.route("/api/model_info")
def api_model_info():
    """获取模型评估指标和特征重要度，用于前端展示模型效果"""
    metrics_path = os.path.join("models", "saved", "model_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
    # 从随机森林模型中提取特征重要度
    feature_importance = get_feature_importance(fault_model)
    return jsonify({
        "code": 0,
        "data": {
            "accuracy": metrics.get("accuracy"),           # 测试集准确率
            "confusion_matrix": metrics.get("confusion_matrix"),  # 混淆矩阵
            "feature_importance": feature_importance        # 8路传感器特征重要度
        }
    })


# ============================================================
# 第四部分：告警与运维API（告警查询、处置、运维记录）
# 通俗解释：就像餐厅的"前台管理"，处理客人投诉、记录处理结果
# ============================================================

# 获取告警列表，支持按设备、状态、级别筛选
@app.route("/api/alerts")
def api_alerts():
    """
    获取告警列表
    支持三个筛选条件（都可选）：
      device_id：只看某台设备的告警
      status：只看某种状态（未处理/已处理）
      severity：只看某种级别（高危/中危）
    """
    device_id = request.args.get("device_id", "")
    status = request.args.get("status", "")
    severity = request.args.get("severity", "")
    # 基础SQL语句，WHERE 1=1 是为了方便后面拼接条件
    sql = "SELECT * FROM alerts WHERE 1=1"
    params = []
    # 根据用户传的筛选条件，动态拼接SQL
    if device_id:
        sql += " AND device_id=?"
        params.append(device_id)
    if status:
        sql += " AND status=?"
        params.append(status)
    if severity:
        sql += " AND severity=?"
        params.append(severity)
    sql += " ORDER BY timestamp DESC LIMIT 500"  # 按时间倒序，最多返回500条
    conn = get_conn()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return jsonify({"code": 0, "data": [dict(r) for r in rows]})


# 处理告警：把告警状态改成"已处理"，同时创建一条运维记录
@app.route("/api/alerts/handle", methods=["POST"])
def api_handle_alert():
    """
    处理告警
    做两件事：
      1. 把这条告警的状态从"未处理"改成"已处理"
      2. 在运维记录表中新增一条记录（谁、什么时候、做了什么处理）
    """
    body = request.get_json(force=True)
    alert_id = body.get("alert_id")                    # 要处理的告警ID
    action = body.get("action", "现场巡检确认，设备恢复正常")  # 处置措施
    operator = body.get("operator", "运维人员")          # 处理人

    conn = get_conn()
    # 先查一下这条告警是否存在
    alert = conn.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
    if not alert:
        conn.close()
        return jsonify({"code": 1, "msg": "告警不存在"}), 404

    # 第一件事：更新告警状态为已处理
    conn.execute("UPDATE alerts SET status='已处理' WHERE id=?", (alert_id,))
    # 第二件事：插入运维记录
    conn.execute(
        "INSERT INTO maintenance(alert_id, device_id, action, operator) VALUES(?,?,?,?)",
        (alert_id, alert["device_id"], action, operator)
    )
    conn.commit()
    conn.close()
    return jsonify({"code": 0, "msg": "告警已处理，运维记录已创建"})


# 获取运维处理记录列表（关联告警表，能看到每条运维记录对应的告警内容）
@app.route("/api/maintenance")
def api_maintenance():
    """获取运维处理记录，关联告警表显示告警内容和级别"""
    conn = get_conn()
    rows = conn.execute("""
        SELECT m.*, a.message AS alert_message, a.severity
        FROM maintenance m LEFT JOIN alerts a ON m.alert_id = a.id
        ORDER BY m.create_time DESC
    """).fetchall()
    conn.close()
    return jsonify({"code": 0, "data": [dict(r) for r in rows]})


# ============================================================
# 第五部分：统计聚合API（首页和驾驶舱用的汇总数据）
# 通俗解释：就像餐厅的"日报表"，把一天的营业额、客流量、好评率汇总出来
# ============================================================

# 首页基础统计数据
@app.route("/api/stats")
def api_stats():
    """首页顶部统计卡片数据：总数据量、总告警数、未处理/已处理数、高危/中危未处理数"""
    conn = get_conn()
    total_data = conn.execute("SELECT COUNT(*) c FROM sensor_data").fetchone()["c"]
    total_alerts = conn.execute("SELECT COUNT(*) c FROM alerts").fetchone()["c"]
    unhandled = conn.execute("SELECT COUNT(*) c FROM alerts WHERE status='未处理'").fetchone()["c"]
    handled = conn.execute("SELECT COUNT(*) c FROM alerts WHERE status='已处理'").fetchone()["c"]
    high_risk = conn.execute("SELECT COUNT(*) c FROM alerts WHERE severity='高危' AND status='未处理'").fetchone()["c"]
    mid_risk = conn.execute("SELECT COUNT(*) c FROM alerts WHERE severity='中危' AND status='未处理'").fetchone()["c"]
    conn.close()
    return jsonify({"code": 0, "data": {
        "total_data": total_data,      # 传感器数据总条数
        "total_alerts": total_alerts,   # 告警总条数
        "unhandled": unhandled,          # 未处理告警数
        "handled": handled,              # 已处理告警数
        "high_risk": high_risk,          # 高危未处理数
        "mid_risk": mid_risk             # 中危未处理数
    }})


# 管理驾驶舱聚合数据（设备健康度、状态分布、告警级别分布、最近告警）
@app.route("/api/dashboard")
def api_dashboard():
    """
    管理驾驶舱页面的所有数据一次性返回：
      1. 每台设备的健康度评分和状态
      2. 设备状态分布（健康/关注/故障各几台）
      3. 告警级别分布
      4. 全厂平均健康率
      5. 最近5条告警
    """
    conn = get_conn()
    devices = conn.execute("SELECT * FROM devices ORDER BY device_id").fetchall()
    device_health = []
    status_dist = {"healthy": 0, "warning": 0, "critical": 0}
    for d in devices:
        # 统计这台设备的总样本数和异常数
        stat = conn.execute(
            "SELECT COUNT(*) total, SUM(anomaly) abnormal FROM sensor_data WHERE device_id=?",
            (d["device_id"],)
        ).fetchone()
        total = stat["total"] or 1
        abnormal = stat["abnormal"] or 0
        # 健康度 = 正常样本占比 × 100（比如90%正常就是90分）
        health_score = round((total - abnormal) / total * 100, 1)
        # 根据健康度评分划分状态：>=85健康，>=60关注，<60故障
        if health_score >= 85:
            health_status, status_dist["healthy"] = "健康", status_dist["healthy"] + 1
        elif health_score >= 60:
            health_status, status_dist["warning"] = "关注", status_dist["warning"] + 1
        else:
            health_status, status_dist["critical"] = "故障", status_dist["critical"] + 1
        # 这台设备有多少条未处理告警
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

    # 告警级别分布统计
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

    # 全厂平均健康率 = 所有设备健康度的平均值
    avg_health = round(sum(d["health_score"] for d in device_health) / len(device_health), 1) if device_health else 0
    return jsonify({"code": 0, "data": {
        "device_health": device_health,    # 每台设备健康详情
        "status_dist": status_dist,         # 设备状态分布
        "severity_dist": severity_dist,     # 告警级别分布
        "avg_health": avg_health,           # 全厂平均健康率
        "recent_alerts": recent_alerts      # 最近5条告警
    }})


# 告警时间趋势：按日期统计每天的高危/中危告警数量，用于画趋势图
@app.route("/api/alert_trend")
def api_alert_trend():
    """告警时间趋势数据：按日期聚合，返回每天的高危和中危告警数量"""
    conn = get_conn()
    rows = conn.execute("""
        SELECT substr(timestamp,1,10) AS day, severity, COUNT(*) AS cnt
        FROM alerts GROUP BY day, severity ORDER BY day
    """).fetchall()
    conn.close()
    # 提取所有不重复的日期
    days = sorted(set(r["day"] for r in rows))
    # 分别统计每天的高危和中危数量
    high = [sum(r["cnt"] for r in rows if r["day"] == d and r["severity"] == "高危") for d in days]
    mid = [sum(r["cnt"] for r in rows if r["day"] == d and r["severity"] == "中危") for d in days]
    return jsonify({"code": 0, "data": {"days": days, "high": high, "mid": mid}})


# 单设备健康度详情：正常样本和异常样本各传感器的平均值，用于画雷达图对比
@app.route("/api/device_health_detail")
def api_device_health_detail():
    """
    单设备健康度详情：分别计算正常样本和异常样本中8路传感器的平均值
    用于前端画雷达图，对比正常和异常状态下各传感器的差异
    """
    device_id = request.args.get("device_id", "PUMP-001")
    conn = get_conn()
    # 动态构建SQL：一次性查出8路传感器的平均值
    avg_sql = "SELECT " + ", ".join([f"AVG({c.replace(' ','_')}) AS {c.replace(' ','_')}" for c in SENSOR_COLS])
    # 正常样本各传感器均值
    normal_avg = conn.execute(
        avg_sql + " FROM sensor_data WHERE device_id=? AND anomaly=0", (device_id,)
    ).fetchone()
    # 异常样本各传感器均值
    abnormal_avg = conn.execute(
        avg_sql + " FROM sensor_data WHERE device_id=? AND anomaly=1", (device_id,)
    ).fetchone()
    conn.close()
    return jsonify({"code": 0, "data": {
        "normal_avg": {c: round(normal_avg[c.replace(' ', '_')] or 0, 3) for c in SENSOR_COLS},
        "abnormal_avg": {c: round(abnormal_avg[c.replace(' ', '_')] or 0, 3) for c in SENSOR_COLS}
    }})


# ============================================================
# 程序入口：运行这个文件时，启动网站服务器
# ============================================================
if __name__ == "__main__":
    # host="127.0.0.1" 表示只能本机访问（安全，课程设计用这个就行）
    # port=5000 表示网站运行在5000端口
    # debug=False 表示关闭调试模式（正式运行用False，开发时可以设True看详细报错）
    app.run(host="127.0.0.1", port=5000, debug=False)
