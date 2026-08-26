 数据集说明
 数据集名称
SKAB (Skoltech Anomaly Benchmark) 工业设备异常检测数据集

 数据来源
开源仓库：https://github.com/KotsoevK/SKAB

 业务场景
工业水泵系统多传感器时序监测数据，包含正常工况、阀门故障、介质泄漏、管路堵塞等多种设备异常场景，每条样本附带异常标签。

 核心字段
| 字段 | 含义 |
|------|------|
| datetime | 采样时间戳 |
| Temperature | 温度传感器读数 |
| Pressure | 压力传感器读数 |
| Flow | 流量传感器读数 |
| Vibrations | 振动传感器读数 |
| anomaly | 异常标签（0=正常工况，1=设备故障异常） |

 目录说明
 `skab/`：原始CSV数据集
 后续数据清洗、特征工程后的处理数据、SQLite数据库文件也统一存放于本目录
