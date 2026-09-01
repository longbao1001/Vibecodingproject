"""
制造智能技术课程设计 - 系统自动化测试脚本
覆盖：数据库、三大算法模块、核心API、业务闭环、页面路由
运行方式：python test_system.py  或  python -m unittest test_system -v
"""
import os
import sys
import json
import unittest

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, SENSOR_COLS, DB_SENSOR_COLS
from database import get_conn
from models.anomaly_detection import load_anomaly_model, detect_anomaly
from models.fault_prediction import load_fault_model, predict_fault
from models.root_cause import analyze_root_cause, get_feature_importance


# ============================================================
# 1. 数据库测试
# ============================================================
class TestDatabase(unittest.TestCase):
    """验证SQLite数据库表结构、数据量、设备映射"""

    def test_tables_exist(self):
        """4张核心表必须存在"""
        conn = get_conn()
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        for t in ["devices", "sensor_data", "alerts", "maintenance"]:
            self.assertIn(t, tables, f"缺少表: {t}")

    def test_devices_count(self):
        """必须有4台虚拟设备"""
        conn = get_conn()
        cnt = conn.execute("SELECT COUNT(*) c FROM devices").fetchone()["c"]
        conn.close()
        self.assertEqual(cnt, 4)

    def test_device_ids(self):
        """设备编号必须为 PUMP-001 ~ PUMP-004"""
        conn = get_conn()
        ids = [r["device_id"] for r in conn.execute(
            "SELECT device_id FROM devices ORDER BY device_id"
        ).fetchall()]
        conn.close()
        self.assertEqual(ids, ["PUMP-001", "PUMP-002", "PUMP-003", "PUMP-004"])

    def test_sensor_data_total(self):
        """传感器数据总量应为46669条（SKAB数据集预处理后）"""
        conn = get_conn()
        cnt = conn.execute("SELECT COUNT(*) c FROM sensor_data").fetchone()["c"]
        conn.close()
        self.assertEqual(cnt, 46669)

    def test_sensor_data_per_device(self):
        """每台设备数据量应与SKAB场景对应"""
        expected = {"PUMP-001": 9401, "PUMP-002": 18162,
                    "PUMP-003": 4312, "PUMP-004": 14794}
        conn = get_conn()
        for dev, exp in expected.items():
            cnt = conn.execute(
                "SELECT COUNT(*) c FROM sensor_data WHERE device_id=?", (dev,)
            ).fetchone()["c"]
            self.assertEqual(cnt, exp, f"{dev} 数据量不符")
        conn.close()

    def test_sensor_columns_exist(self):
        """8路传感器列必须存在于sensor_data表"""
        conn = get_conn()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sensor_data)").fetchall()]
        conn.close()
        for dc in DB_SENSOR_COLS:
            self.assertIn(dc, cols, f"缺少传感器列: {dc}")

    def test_no_null_sensor_values(self):
        """传感器数据不应有空值"""
        conn = get_conn()
        for dc in DB_SENSOR_COLS:
            cnt = conn.execute(
                f"SELECT COUNT(*) c FROM sensor_data WHERE {dc} IS NULL"
            ).fetchone()["c"]
            self.assertEqual(cnt, 0, f"{dc} 存在空值")
        conn.close()


# ============================================================
# 2. 算法模块测试
# ============================================================
class TestAlgorithms(unittest.TestCase):
    """验证三大算法模块加载、推理、返回结构"""

    @classmethod
    def setUpClass(cls):
        cls.anomaly_model = load_anomaly_model()
        cls.fault_model = load_fault_model()
        # 取PUMP-001一条正常数据作为测试样本
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM sensor_data WHERE device_id='PUMP-001' LIMIT 1"
        ).fetchone()
        conn.close()
        cls.sample = {sc: row[dc] for sc, dc in zip(SENSOR_COLS, DB_SENSOR_COLS)}

    def test_anomaly_model_loaded(self):
        """异常检测模型必须加载成功"""
        self.assertIsNotNone(self.anomaly_model)

    def test_anomaly_detect_return(self):
        """detect_anomaly 必须返回 is_anomaly(0/1) 和 anomaly_score(float)"""
        result = detect_anomaly(self.anomaly_model, self.sample)
        self.assertIn("is_anomaly", result)
        self.assertIn("anomaly_score", result)
        self.assertIn(result["is_anomaly"], [0, 1])
        self.assertIsInstance(result["anomaly_score"], float)

    def test_fault_model_loaded(self):
        """故障预测模型必须加载成功"""
        self.assertIsNotNone(self.fault_model)

    def test_fault_predict_return(self):
        """predict_fault 必须返回 fault_label/fault_prob/normal_prob，概率和为1"""
        result = predict_fault(self.fault_model, self.sample)
        self.assertIn("fault_label", result)
        self.assertIn("fault_prob", result)
        self.assertIn("normal_prob", result)
        self.assertIn(result["fault_label"], [0, 1])
        self.assertAlmostEqual(
            result["fault_prob"] + result["normal_prob"], 1.0, places=2
        )
        self.assertGreaterEqual(result["fault_prob"], 0)
        self.assertLessEqual(result["fault_prob"], 1)

    def test_root_cause_return(self):
        """analyze_root_cause 必须返回 root_causes(list) 和 conclusion(非空字符串)"""
        result = analyze_root_cause(self.fault_model, self.sample, top_k=3)
        self.assertIn("root_causes", result)
        self.assertIn("conclusion", result)
        self.assertEqual(len(result["root_causes"]), 3)
        self.assertTrue(len(result["conclusion"]) > 0)
        # 每个根因应包含特征名和贡献度
        for rc in result["root_causes"]:
            self.assertIn("feature", rc)
            self.assertIn("importance", rc)

    def test_feature_importance_length(self):
        """get_feature_importance 必须返回8个传感器的重要度"""
        fi = get_feature_importance(self.fault_model)
        self.assertEqual(len(fi), 8)

    def test_model_accuracy(self):
        """模型指标文件应存在且准确率>0.9"""
        metrics_path = os.path.join("models", "saved", "model_metrics.json")
        self.assertTrue(os.path.exists(metrics_path))
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        self.assertIn("accuracy", metrics)
        self.assertGreater(metrics["accuracy"], 0.9)


# ============================================================
# 3. API 接口测试
# ============================================================
class TestAPI(unittest.TestCase):
    """使用Flask test_client验证全部API接口"""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def _get_json(self, url):
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200, f"GET {url} 状态码异常")
        return resp.get_json()

    def _post_json(self, url, data):
        resp = self.client.post(url, json=data)
        self.assertEqual(resp.status_code, 200, f"POST {url} 状态码异常")
        return resp.get_json()

    # --- 数据查询 API ---
    def test_api_devices(self):
        data = self._get_json("/api/devices")["data"]
        self.assertEqual(len(data), 4)
        self.assertEqual(data[0]["device_id"], "PUMP-001")
        self.assertGreater(data[0]["total_samples"], 0)

    def test_api_sensor_data(self):
        data = self._get_json("/api/sensor_data?device_id=PUMP-001&limit=10")["data"]
        self.assertEqual(len(data), 10)
        self.assertIn("timestamp", data[0])

    def test_api_latest_data(self):
        data = self._get_json("/api/latest_data?device_id=PUMP-002")["data"]
        self.assertIsNotNone(data)
        self.assertEqual(data["device_id"], "PUMP-002")

    # --- 算法分析 API ---
    def test_api_analyze(self):
        data = self._post_json("/api/analyze", {"device_id": "PUMP-004"})["data"]
        self.assertIn("anomaly_detection", data)
        self.assertIn("fault_prediction", data)
        self.assertIn("root_cause", data)
        self.assertIn(data["severity"], ["高危", "中危", "正常"])
        self.assertIn("is_anomaly", data["anomaly_detection"])
        self.assertIn("fault_prob", data["fault_prediction"])

    def test_api_analyze_with_values(self):
        """传入自定义传感器值也应正常分析"""
        values = {c: 0.5 for c in SENSOR_COLS}
        data = self._post_json("/api/analyze", {
            "device_id": "PUMP-001", "sensor_values": values
        })["data"]
        self.assertIn("severity", data)

    def test_api_model_info(self):
        data = self._get_json("/api/model_info")["data"]
        self.assertIn("accuracy", data)
        self.assertGreater(data["accuracy"], 0.9)
        self.assertEqual(len(data["feature_importance"]), 8)

    # --- 统计与驾驶舱 API ---
    def test_api_stats(self):
        data = self._get_json("/api/stats")["data"]
        self.assertEqual(data["total_data"], 46669)
        self.assertIn("total_alerts", data)
        self.assertIn("unhandled", data)
        self.assertIn("high_risk", data)
        self.assertIn("mid_risk", data)

    def test_api_dashboard(self):
        data = self._get_json("/api/dashboard")["data"]
        self.assertEqual(len(data["device_health"]), 4)
        self.assertIn("healthy", data["status_dist"])
        self.assertIn("warning", data["status_dist"])
        self.assertIn("critical", data["status_dist"])
        self.assertIn("avg_health", data)
        self.assertGreater(data["avg_health"], 0)
        self.assertLessEqual(data["avg_health"], 100)

    def test_api_alert_trend(self):
        data = self._get_json("/api/alert_trend")["data"]
        self.assertIn("days", data)
        self.assertIn("high", data)
        self.assertIn("mid", data)
        self.assertEqual(len(data["days"]), len(data["high"]))
        self.assertEqual(len(data["days"]), len(data["mid"]))

    def test_api_device_health_detail(self):
        data = self._get_json("/api/device_health_detail?device_id=PUMP-002")["data"]
        self.assertIn("normal_avg", data)
        self.assertIn("abnormal_avg", data)
        self.assertEqual(len(data["normal_avg"]), 8)

    # --- 告警列表 API（含筛选） ---
    def test_api_alerts_list(self):
        data = self._get_json("/api/alerts")["data"]
        self.assertIsInstance(data, list)

    def test_api_alerts_filter_severity(self):
        """按级别筛选：高危结果必须全部为高危"""
        data = self._get_json("/api/alerts?severity=高危")["data"]
        for a in data:
            self.assertEqual(a["severity"], "高危")

    def test_api_alerts_filter_device(self):
        """按设备筛选：结果必须全部为指定设备"""
        data = self._get_json("/api/alerts?device_id=PUMP-003")["data"]
        for a in data:
            self.assertEqual(a["device_id"], "PUMP-003")

    def test_api_alerts_filter_status(self):
        """按状态筛选：未处理结果必须全部为未处理"""
        data = self._get_json("/api/alerts?status=未处理")["data"]
        for a in data:
            self.assertEqual(a["status"], "未处理")

    def test_api_alerts_combined_filter(self):
        """组合筛选：设备+级别+状态"""
        data = self._get_json(
            "/api/alerts?device_id=PUMP-004&severity=高危&status=未处理"
        )["data"]
        for a in data:
            self.assertEqual(a["device_id"], "PUMP-004")
            self.assertEqual(a["severity"], "高危")
            self.assertEqual(a["status"], "未处理")


# ============================================================
# 4. 业务闭环测试
# ============================================================
class TestBusinessFlow(unittest.TestCase):
    """验证完整业务闭环：扫描→告警→处理→运维记录"""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_01_scan_generates_alerts(self):
        """批量扫描应返回new_alerts且不报错"""
        resp = self.client.post("/api/scan", json={
            "device_id": "PUMP-001", "limit": 20
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("new_alerts", data)
        self.assertGreaterEqual(data["new_alerts"], 0)

    def test_02_alerts_queryable(self):
        """扫描后告警列表应可查询"""
        data = self.client.get("/api/alerts").get_json()["data"]
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_03_handle_alert_creates_maintenance(self):
        """处理告警：状态更新为已处理，并创建运维记录"""
        # 取一条未处理告警
        alerts = self.client.get("/api/alerts?status=未处理").get_json()["data"]
        self.assertGreater(len(alerts), 0, "没有未处理告警可测试")
        target = alerts[0]
        alert_id = target["id"]
        device_id = target["device_id"]

        # 处理告警
        resp = self.client.post("/api/alerts/handle", json={
            "alert_id": alert_id,
            "action": "自动化测试：现场巡检确认，设备恢复正常",
            "operator": "测试运维"
        })
        self.assertEqual(resp.status_code, 200)

        # 验证告警状态已更新
        updated = self.client.get("/api/alerts").get_json()["data"]
        handled = [a for a in updated if a["id"] == alert_id][0]
        self.assertEqual(handled["status"], "已处理")

        # 验证运维记录已创建
        maint = self.client.get("/api/maintenance").get_json()["data"]
        self.assertGreater(len(maint), 0)
        matched = [m for m in maint if m["alert_id"] == alert_id]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["device_id"], device_id)
        self.assertEqual(matched[0]["operator"], "测试运维")

    def test_04_handle_nonexistent_alert_returns_404(self):
        """处理不存在的告警应返回404"""
        resp = self.client.post("/api/alerts/handle", json={
            "alert_id": 999999, "action": "test"
        })
        self.assertEqual(resp.status_code, 404)

    def test_05_maintenance_records_complete(self):
        """运维记录应包含完整字段"""
        data = self.client.get("/api/maintenance").get_json()["data"]
        if data:
            m = data[0]
            self.assertIn("id", m)
            self.assertIn("alert_id", m)
            self.assertIn("device_id", m)
            self.assertIn("action", m)
            self.assertIn("operator", m)
            self.assertIn("create_time", m)


# ============================================================
# 5. 页面路由测试
# ============================================================
class TestPages(unittest.TestCase):
    """验证4个前端页面路由可正常访问"""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_page_index(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("设备运维管理驾驶舱", resp.data.decode("utf-8"))

    def test_page_monitor(self):
        resp = self.client.get("/monitor")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("传感器实时监控", resp.data.decode("utf-8"))

    def test_page_alerts(self):
        resp = self.client.get("/alerts")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("故障告警管理面板", resp.data.decode("utf-8"))

    def test_page_history(self):
        resp = self.client.get("/history")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("历史运维处置记录", resp.data.decode("utf-8"))


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("制造智能技术课程设计 - 系统自动化测试")
    print("=" * 60)
    unittest.main(verbosity=2)
