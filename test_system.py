# ============================================================
# 文件名：test_system.py
# 作用：系统自动化测试脚本 —— 自动检查整个系统有没有bug
# 通俗解释：
#   你可以把这个文件想象成一个"质检员"。
#   系统开发完了，怎么知道它有没有问题？总不能手动点一遍吧？
#   这个"质检员"会自动跑38个检查项目，每个项目验证系统的一个功能：
#     1. 检查数据库（表在不在、数据对不对、有没有空值）
#     2. 检查算法模型（能不能加载、预测结果格式对不对）
#     3. 检查API接口（每个接口能不能正常返回数据）
#     4. 检查业务闭环（扫描→告警→处理→记录，一条龙通不通）
#     5. 检查页面路由（4个网页能不能正常打开）
#   全部跑完后，告诉你"38个检查全部通过"还是"第几个检查失败了"。
# 运行方式：python test_system.py
# 运行结果：Ran 38 tests in 1.257s  OK（表示全部通过）
# ============================================================

import os
import sys
import json
import unittest  # Python自带的测试框架，就像给质检员提供的"检查清单"模板

# 把项目根目录加入Python路径，这样才能import项目里的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入要测试的模块
from app import app, SENSOR_COLS, DB_SENSOR_COLS  # 网站应用和传感器列名
from database import get_conn  # 数据库连接
from models.anomaly_detection import load_anomaly_model, detect_anomaly  # 异常检测
from models.fault_prediction import load_fault_model, predict_fault  # 故障预测
from models.root_cause import analyze_root_cause, get_feature_importance  # 根因分析


# ============================================================
# 第一组测试：数据库检查（7个检查项）
# 通俗解释：质检员先去仓库（数据库）检查，看看货架（表）在不在、
#           货物（数据）数量对不对、有没有损坏（空值）。
# ============================================================
class TestDatabase(unittest.TestCase):
    """验证SQLite数据库表结构、数据量、设备映射"""

    def test_tables_exist(self):
        """检查1：4张核心表必须都存在（devices/sensor_data/alerts/maintenance）"""
        conn = get_conn()
        # 查询数据库里所有表的名字
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        # 断言：4张表必须都在列表里，不在就报错
        for t in ["devices", "sensor_data", "alerts", "maintenance"]:
            self.assertIn(t, tables, f"缺少表: {t}")

    def test_devices_count(self):
        """检查2：设备表里必须有4台设备"""
        conn = get_conn()
        cnt = conn.execute("SELECT COUNT(*) c FROM devices").fetchone()["c"]
        conn.close()
        self.assertEqual(cnt, 4)  # 断言：数量必须等于4

    def test_device_ids(self):
        """检查3：设备编号必须是PUMP-001到PUMP-004"""
        conn = get_conn()
        ids = [r["device_id"] for r in conn.execute(
            "SELECT device_id FROM devices ORDER BY device_id"
        ).fetchall()]
        conn.close()
        self.assertEqual(ids, ["PUMP-001", "PUMP-002", "PUMP-003", "PUMP-004"])

    def test_sensor_data_total(self):
        """检查4：传感器数据总量必须是46669条（SKAB数据集预处理后的数量）"""
        conn = get_conn()
        cnt = conn.execute("SELECT COUNT(*) c FROM sensor_data").fetchone()["c"]
        conn.close()
        self.assertEqual(cnt, 46669)

    def test_sensor_data_per_device(self):
        """检查5：每台设备的数据量必须和SKAB场景对应（4台设备加起来=46669）"""
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
        """检查6：8路传感器列必须都存在于sensor_data表中"""
        conn = get_conn()
        # PRAGMA table_info可以查询表的所有列名
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sensor_data)").fetchall()]
        conn.close()
        for dc in DB_SENSOR_COLS:
            self.assertIn(dc, cols, f"缺少传感器列: {dc}")

    def test_no_null_sensor_values(self):
        """检查7：传感器数据不能有空值（预处理时已经填充过了）"""
        conn = get_conn()
        for dc in DB_SENSOR_COLS:
            cnt = conn.execute(
                f"SELECT COUNT(*) c FROM sensor_data WHERE {dc} IS NULL"
            ).fetchone()["c"]
            self.assertEqual(cnt, 0, f"{dc} 存在空值")
        conn.close()


# ============================================================
# 第二组测试：算法模块检查（8个检查项）
# 通俗解释：质检员检查三个"专家"（三个算法模型）能不能正常工作，
#           给它们一组数据，看它们返回的结果格式对不对、数值合不合理。
# ============================================================
class TestAlgorithms(unittest.TestCase):
    """验证三大算法模块加载、推理、返回结构"""

    @classmethod
    def setUpClass(cls):
        """所有测试开始前，先加载模型和取一条测试数据（只做一次，节省时间）"""
        cls.anomaly_model = load_anomaly_model()  # 加载异常检测模型
        cls.fault_model = load_fault_model()      # 加载故障预测模型
        # 从数据库取PUMP-001的一条数据作为测试样本
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM sensor_data WHERE device_id='PUMP-001' LIMIT 1"
        ).fetchone()
        conn.close()
        # 把数据库行转成算法能看懂的字典格式
        cls.sample = {sc: row[dc] for sc, dc in zip(SENSOR_COLS, DB_SENSOR_COLS)}

    def test_anomaly_model_loaded(self):
        """检查1：异常检测模型必须加载成功（不能是None）"""
        self.assertIsNotNone(self.anomaly_model)

    def test_anomaly_detect_return(self):
        """检查2：异常检测必须返回is_anomaly(0或1)和anomaly_score(小数)"""
        result = detect_anomaly(self.anomaly_model, self.sample)
        self.assertIn("is_anomaly", result)       # 必须有"是否异常"字段
        self.assertIn("anomaly_score", result)     # 必须有"异常分数"字段
        self.assertIn(result["is_anomaly"], [0, 1])  # 值必须是0或1
        self.assertIsInstance(result["anomaly_score"], float)  # 分数必须是小数

    def test_fault_model_loaded(self):
        """检查3：故障预测模型必须加载成功"""
        self.assertIsNotNone(self.fault_model)

    def test_fault_predict_return(self):
        """检查4：故障预测必须返回标签/故障概率/正常概率，且两个概率加起来=1"""
        result = predict_fault(self.fault_model, self.sample)
        self.assertIn("fault_label", result)    # 故障标签（0正常/1故障）
        self.assertIn("fault_prob", result)     # 故障概率
        self.assertIn("normal_prob", result)    # 正常概率
        self.assertIn(result["fault_label"], [0, 1])
        # 故障概率+正常概率必须约等于1（允许0.01的误差，因为浮点数精度）
        self.assertAlmostEqual(
            result["fault_prob"] + result["normal_prob"], 1.0, places=2
        )
        self.assertGreaterEqual(result["fault_prob"], 0)  # 概率不能小于0
        self.assertLessEqual(result["fault_prob"], 1)     # 概率不能大于1

    def test_root_cause_return(self):
        """检查5：根因分析必须返回3个根因和一段非空结论"""
        result = analyze_root_cause(self.fault_model, self.sample, top_k=3)
        self.assertIn("root_causes", result)   # 必须有根因列表
        self.assertIn("conclusion", result)     # 必须有结论文本
        self.assertEqual(len(result["root_causes"]), 3)  # 必须正好3个根因
        self.assertTrue(len(result["conclusion"]) > 0)    # 结论不能为空
        # 每个根因必须包含特征名和贡献度
        for rc in result["root_causes"]:
            self.assertIn("feature", rc)
            self.assertIn("importance", rc)

    def test_feature_importance_length(self):
        """检查6：特征重要度必须返回8个传感器的重要度"""
        fi = get_feature_importance(self.fault_model)
        self.assertEqual(len(fi), 8)

    def test_model_accuracy(self):
        """检查7：模型指标文件必须存在，且准确率必须大于0.9（90%）"""
        metrics_path = os.path.join("models", "saved", "model_metrics.json")
        self.assertTrue(os.path.exists(metrics_path))
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        self.assertIn("accuracy", metrics)
        self.assertGreater(metrics["accuracy"], 0.9)


# ============================================================
# 第三组测试：API接口检查（17个检查项）
# 通俗解释：质检员假装是前端网页，给每个接口发请求，
#           看接口能不能正常返回数据、返回的数据格式对不对。
#           用Flask的test_client，不需要真的启动服务器。
# ============================================================
class TestAPI(unittest.TestCase):
    """使用Flask test_client验证全部API接口"""

    @classmethod
    def setUpClass(cls):
        """创建测试客户端（模拟浏览器发请求，不需要真启动服务器）"""
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def _get_json(self, url):
        """辅助方法：发GET请求，验证状态码200，返回JSON数据"""
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200, f"GET {url} 状态码异常")
        return resp.get_json()

    def _post_json(self, url, data):
        """辅助方法：发POST请求，验证状态码200，返回JSON数据"""
        resp = self.client.post(url, json=data)
        self.assertEqual(resp.status_code, 200, f"POST {url} 状态码异常")
        return resp.get_json()

    # --- 数据查询接口测试 ---
    def test_api_devices(self):
        """设备列表接口：必须返回4台设备，每台有数据量统计"""
        data = self._get_json("/api/devices")["data"]
        self.assertEqual(len(data), 4)
        self.assertEqual(data[0]["device_id"], "PUMP-001")
        self.assertGreater(data[0]["total_samples"], 0)

    def test_api_sensor_data(self):
        """传感器数据接口：请求10条必须返回10条，每条有时间戳"""
        data = self._get_json("/api/sensor_data?device_id=PUMP-001&limit=10")["data"]
        self.assertEqual(len(data), 10)
        self.assertIn("timestamp", data[0])

    def test_api_latest_data(self):
        """最新数据接口：必须返回PUMP-002的最新一条数据"""
        data = self._get_json("/api/latest_data?device_id=PUMP-002")["data"]
        self.assertIsNotNone(data)
        self.assertEqual(data["device_id"], "PUMP-002")

    # --- 算法分析接口测试 ---
    def test_api_analyze(self):
        """智能诊断接口：必须返回三个算法结果+告警级别"""
        data = self._post_json("/api/analyze", {"device_id": "PUMP-004"})["data"]
        self.assertIn("anomaly_detection", data)
        self.assertIn("fault_prediction", data)
        self.assertIn("root_cause", data)
        self.assertIn(data["severity"], ["高危", "中危", "正常"])

    def test_api_analyze_with_values(self):
        """智能诊断接口：传入自定义传感器值也能正常分析"""
        values = {c: 0.5 for c in SENSOR_COLS}  # 构造一组全是0.5的测试数据
        data = self._post_json("/api/analyze", {
            "device_id": "PUMP-001", "sensor_values": values
        })["data"]
        self.assertIn("severity", data)

    def test_api_model_info(self):
        """模型信息接口：必须返回准确率(>0.9)和8个特征重要度"""
        data = self._get_json("/api/model_info")["data"]
        self.assertIn("accuracy", data)
        self.assertGreater(data["accuracy"], 0.9)
        self.assertEqual(len(data["feature_importance"]), 8)

    # --- 统计与驾驶舱接口测试 ---
    def test_api_stats(self):
        """首页统计接口：总数据量必须是46669，包含告警统计"""
        data = self._get_json("/api/stats")["data"]
        self.assertEqual(data["total_data"], 46669)
        self.assertIn("total_alerts", data)
        self.assertIn("unhandled", data)
        self.assertIn("high_risk", data)
        self.assertIn("mid_risk", data)

    def test_api_dashboard(self):
        """驾驶舱接口：必须返回4台设备健康度、状态分布、平均健康率"""
        data = self._get_json("/api/dashboard")["data"]
        self.assertEqual(len(data["device_health"]), 4)
        self.assertIn("healthy", data["status_dist"])
        self.assertIn("warning", data["status_dist"])
        self.assertIn("critical", data["status_dist"])
        self.assertIn("avg_health", data)
        self.assertGreater(data["avg_health"], 0)
        self.assertLessEqual(data["avg_health"], 100)  # 健康率不能超过100

    def test_api_alert_trend(self):
        """告警趋势接口：日期、高危、中危三个列表长度必须一致"""
        data = self._get_json("/api/alert_trend")["data"]
        self.assertIn("days", data)
        self.assertIn("high", data)
        self.assertIn("mid", data)
        self.assertEqual(len(data["days"]), len(data["high"]))
        self.assertEqual(len(data["days"]), len(data["mid"]))

    def test_api_device_health_detail(self):
        """设备健康详情接口：必须返回正常/异常样本各8个传感器均值"""
        data = self._get_json("/api/device_health_detail?device_id=PUMP-002")["data"]
        self.assertIn("normal_avg", data)
        self.assertIn("abnormal_avg", data)
        self.assertEqual(len(data["normal_avg"]), 8)

    # --- 告警列表接口测试（含筛选） ---
    def test_api_alerts_list(self):
        """告警列表接口：必须返回列表格式"""
        data = self._get_json("/api/alerts")["data"]
        self.assertIsInstance(data, list)

    def test_api_alerts_filter_severity(self):
        """按级别筛选：筛选高危，返回的必须全是高危"""
        data = self._get_json("/api/alerts?severity=高危")["data"]
        for a in data:
            self.assertEqual(a["severity"], "高危")

    def test_api_alerts_filter_device(self):
        """按设备筛选：筛选PUMP-003，返回的必须全是PUMP-003"""
        data = self._get_json("/api/alerts?device_id=PUMP-003")["data"]
        for a in data:
            self.assertEqual(a["device_id"], "PUMP-003")

    def test_api_alerts_filter_status(self):
        """按状态筛选：筛选未处理，返回的必须全是未处理"""
        data = self._get_json("/api/alerts?status=未处理")["data"]
        for a in data:
            self.assertEqual(a["status"], "未处理")

    def test_api_alerts_combined_filter(self):
        """组合筛选：设备+级别+状态三个条件同时用"""
        data = self._get_json(
            "/api/alerts?device_id=PUMP-004&severity=高危&status=未处理"
        )["data"]
        for a in data:
            self.assertEqual(a["device_id"], "PUMP-004")
            self.assertEqual(a["severity"], "高危")
            self.assertEqual(a["status"], "未处理")


# ============================================================
# 第四组测试：业务闭环检查（5个检查项）
# 通俗解释：质检员模拟完整的业务流程：
#           扫描设备发现异常→生成告警→处理告警→生成运维记录，
#           看一条龙能不能走通。
# ============================================================
class TestBusinessFlow(unittest.TestCase):
    """验证完整业务闭环：扫描→告警→处理→运维记录"""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_01_scan_generates_alerts(self):
        """步骤1：批量扫描PUMP-001最近20条数据，必须不报错"""
        resp = self.client.post("/api/scan", json={
            "device_id": "PUMP-001", "limit": 20
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("new_alerts", data)
        self.assertGreaterEqual(data["new_alerts"], 0)  # 新增告警数>=0

    def test_02_alerts_queryable(self):
        """步骤2：扫描后，告警列表必须能查到数据"""
        data = self.client.get("/api/alerts").get_json()["data"]
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)  # 至少有1条告警

    def test_03_handle_alert_creates_maintenance(self):
        """步骤3：处理一条告警→状态变已处理→同时生成运维记录"""
        # 先取一条未处理告警
        alerts = self.client.get("/api/alerts?status=未处理").get_json()["data"]
        self.assertGreater(len(alerts), 0, "没有未处理告警可测试")
        target = alerts[0]
        alert_id = target["id"]
        device_id = target["device_id"]

        # 调用处理接口
        resp = self.client.post("/api/alerts/handle", json={
            "alert_id": alert_id,
            "action": "自动化测试：现场巡检确认，设备恢复正常",
            "operator": "测试运维"
        })
        self.assertEqual(resp.status_code, 200)

        # 验证：告警状态已变成"已处理"
        updated = self.client.get("/api/alerts").get_json()["data"]
        handled = [a for a in updated if a["id"] == alert_id][0]
        self.assertEqual(handled["status"], "已处理")

        # 验证：运维记录表里多了一条对应记录
        maint = self.client.get("/api/maintenance").get_json()["data"]
        self.assertGreater(len(maint), 0)
        matched = [m for m in maint if m["alert_id"] == alert_id]
        self.assertEqual(len(matched), 1)  # 正好1条
        self.assertEqual(matched[0]["device_id"], device_id)
        self.assertEqual(matched[0]["operator"], "测试运维")

    def test_04_handle_nonexistent_alert_returns_404(self):
        """步骤4：处理一条不存在的告警，必须返回404错误（不能崩溃）"""
        resp = self.client.post("/api/alerts/handle", json={
            "alert_id": 999999, "action": "test"
        })
        self.assertEqual(resp.status_code, 404)

    def test_05_maintenance_records_complete(self):
        """步骤5：运维记录必须包含完整字段（ID/告警ID/设备/措施/处理人/时间）"""
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
# 第五组测试：页面路由检查（4个检查项）
# 通俗解释：质检员检查4个网页能不能正常打开，
#           并且页面里能找到对应的标题文字。
# ============================================================
class TestPages(unittest.TestCase):
    """验证4个前端页面路由可正常访问"""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_page_index(self):
        """首页：状态码200，页面包含"设备运维管理驾驶舱"标题"""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("设备运维管理驾驶舱", resp.data.decode("utf-8"))

    def test_page_monitor(self):
        """监控页：状态码200，页面包含"传感器实时监控"标题"""
        resp = self.client.get("/monitor")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("传感器实时监控", resp.data.decode("utf-8"))

    def test_page_alerts(self):
        """告警页：状态码200，页面包含"故障告警管理面板"标题"""
        resp = self.client.get("/alerts")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("故障告警管理面板", resp.data.decode("utf-8"))

    def test_page_history(self):
        """运维记录页：状态码200，页面包含"历史运维处置记录"标题"""
        resp = self.client.get("/history")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("历史运维处置记录", resp.data.decode("utf-8"))


# ============================================================
# 程序入口：运行这个文件时，自动跑所有测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("制造智能技术课程设计 - 系统自动化测试")
    print("=" * 60)
    # verbosity=2 表示详细输出每个测试的名字和结果
    unittest.main(verbosity=2)
