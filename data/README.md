# data 数据集目录

## 数据集名称
SKAB (Skoltech Anomaly Benchmark) 工业设备异常检测数据集

## 数据来源
开源仓库：https://github.com/KotsoevK/SKAB
> 本仓库 `data/SKAB-master/` 已存放完整原始数据集（数据量较小，可直接入库），无需额外下载。

## 业务场景
工业水泵系统多传感器时序监测数据，包含正常工况、阀门故障（valve1/valve2）、其他异常（other）等多种设备异常场景，每条样本附带异常标签。

## 核心字段（CSV 分号分隔，共 11 列）
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

## 目录结构
- `SKAB-master/`：SKAB 原始数据集（含 valve1、valve2、other、anomaly-free 等故障场景 CSV）
- `processed/`：经过清洗、预处理后输出的 CSV 文件，可直接供算法模块与数据库入库使用

## 预处理说明
预处理脚本位于项目根目录 `preprocess.py`，运行后输出 `processed/out.csv`。
处理逻辑：合并全部故障场景 CSV、时间格式转换、去重、缺失值填充、字段保留、附加故障类型标签。

## 注意
本数据集为公开开源数据集，无涉密、私有敏感数据。
