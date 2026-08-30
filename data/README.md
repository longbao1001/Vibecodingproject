# data 数据集目录

## 数据集名称
SKAB (Skoltech Anomaly Benchmark) 工业设备异常检测数据集

## 数据来源
开源仓库：https://github.com/KotsoevK/SKAB
> 本仓库 `data/SKAB-master/` 已存放完整原始数据集（数据量较小，可直接入库），无需额外下载。

## 数据类型
公开开源数据集，无涉密、私有敏感数据。

## 业务场景
工业水泵系统多传感器时序监测数据，包含正常工况（anomaly-free）、阀门故障（valve1/valve2）、其他异常（other）等多种设备异常场景，每条样本附带异常标签。

## 核心字段（CSV 分号分隔，原始共 11 列）
| 字段 | 含义 |
|------|------|
| datetime | 采样时间戳 |
| Accelerometer1RMS | 1号加速度传感器有效值（振动） |
| Accelerometer2RMS | 2号加速度传感器有效值（振动） |
| Current | 电机电流 |
| Pressure | 压力传感器读数 |
| Temperature | 温度传感器读数 |
| Thermocouple | 热电偶温度读数 |
| Voltage | 电机电压 |
| Volume Flow RateRMS | 流量有效值 |
| anomaly | 异常标签（0=正常工况，1=设备故障异常） |
| changepoint | 工况变化点标记 |

> 注：anomaly-free 正常工况数据原始 CSV 不含 anomaly/changepoint 列，预处理时统一填充为 0。

## 目录结构
- `SKAB-master/`：SKAB 原始数据集（含 anomaly-free、valve1、valve2、other 四类故障场景，共 38 个 CSV）
- `processed/`：经过清洗、预处理后输出的数据文件
  - `out.csv`：预处理后合并数据集（46669 条，11 列业务字段）
  - `data_report.md`：数据预处理质量报告（含预处理前后统计对比）

## 预处理说明
- **预处理脚本**：项目根目录 `preprocess.py`
- **运行方式**：`python preprocess.py`

### 预处理步骤
1. 遍历 `SKAB-master/data/` 下全部 38 个 CSV，合并为单一数据集
2. anomaly-free 数据无 anomaly 列，填充为 0（正常工况）
3. datetime 时间格式转换，删除无效时间戳行
4. 第一次全字段去重
5. 数值列缺失值向前填充，残余缺失用列均值填充
6. anomaly 标签转为整数，按时间排序
7. 保留 11 列业务字段，附加 fault_type 故障类型标签
8. 第二次业务字段去重，确保输出无重复行

### 预处理前后数据统计对比
| 指标 | 预处理前 | 预处理后 |
|------|----------|----------|
| 总样本数 | 46860 | 46669 |
| 缺失值 | 存在 | 0 |
| 重复行 | 191 | 0 |
| 正常样本（anomaly=0） | 33619 | 33428 |
| 异常样本（anomaly=1） | 13241 | 13241 |
| 故障场景数 | 4 类 | 4 类 |
| 字段数 | 11（原始） | 11（业务字段+fault_type） |

详细质量报告见 `processed/data_report.md`。

## 注意
本数据集为公开开源数据集，无涉密、私有敏感数据。
