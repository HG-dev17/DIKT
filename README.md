# DIKT - Dynamic Key-Value Memory Network for Knowledge Tracing

知识追踪模型实现，基于动态键值记忆网络。

## 项目结构
DIKT-master/
├── main.py # 主训练程序
├── run.py # 训练流程控制
├── model.py # DIKT模型定义
├── load_data.py # 数据加载模块
├── config.py # 配置文件
├── glo.py # 全局变量管理
├── transformer.py # Transformer组件
├── requirements.txt # 依赖包列表
└── README.md # 项目说明    
## 环境要求
$$$
```requirements.txt
torch>=1.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
rotary-embedding-torch>=0.2.4
tqdm>=4.60.0
```
$$$
## 安装依赖