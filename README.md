# DIKT - Dynamic Key-Value Memory Network for Knowledge Tracing

Interpretable knowledge tracing with dual-level knowledge states

## 虚拟环境（推荐）
conda create -n dikt python=3.7
conda activate dikt

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
└──pre_process_data/
│   ├── {dataset}/
│   │   ├── {fold}/
│   │   │   ├── train_test/
│   │   │   │   ├── train_question.txt
│   │   │   │   └── test_question.txt
│   │   │   └── graph/
│   │   │       ├── ques_skill.csv
│   │   │       └── train_graphs.npz   

## 环境要求

```requirements.txt
torch>=1.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
rotary-embedding-torch>=0.2.4
tqdm>=4.60.0
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 方法一：使用批处理文件（推荐，Windows CMD）

1. 确保已创建conda环境并安装依赖：
```bash
conda create -n dikt python=3.7
conda activate dikt
pip install -r requirements.txt
```

2. 准备数据集（运行预处理脚本）：
```bash
python preprocess_data.py
```

3. 在CMD中运行批处理文件：
```bash
run_dikt.bat
```
或者直接双击 `run_dikt.bat` 文件

### 方法二：手动运行

1. 准备数据集
2. 运行预处理脚本（如果存在）
3. 激活conda环境并训练模型：
```bash
conda activate dikt
python main.py
```


## 许可证

MIT License

## 数据准备
### 数据集
本项目使用 ASSISTments 2009  数据集。
| 数据集 | 技能数 | 题目数 | 批次大小 |
|--------|--------|--------|----------|
| assist09 | 123 | 15,925 | 64 |
| assist12_1 | 265 | 53,091 | 50 |
| assist17 | 102 | 3,162 | 64 |
| junyi | 39 | 690 | 128 |
| eedi | 309 | 27,566 | 64 |
| ednet | 188 | 12,192 | 64 |
| xes3g5m | 865 | 7,652 | 15 |