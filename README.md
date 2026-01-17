# DIKT - Dynamic Key-Value Memory Network for Knowledge Tracing

Interpretable knowledge tracing with dual-level knowledge states

## 虚拟环境（推荐）
conda create -n dikt python=3.7
conda activate dikt

## 项目结构
```
DIKT-master/
├── main.py                  # 主训练程序
├── run.py                   # 训练流程控制
├── model.py                 # DIKT模型定义
├── load_data.py             # 数据加载模块
├── config.py                # 配置文件
├── glo.py                   # 全局变量管理
├── transformer.py           # Transformer组件
├── preprocess_data.py       # 数据预处理脚本
├── run_dikt.bat             # Windows批处理运行脚本
├── requirements.txt         # 依赖包列表
├── README.md                # 项目说明
├── baseline/                # 基线模型实现
│   ├── AKT/                 # AKT模型
│   ├── AT-DKT/              # AT-DKT模型
│   ├── ATKT/                # ATKT模型
│   ├── CL4KT/               # CL4KT模型
│   ├── DIMKT/               # DIMKT模型
│   ├── DKT/                 # DKT模型
│   ├── DKVMN/               # DKVMN模型
│   ├── DTransformer/        # DTransformer模型
│   ├── GKT/                 # GKT模型
│   ├── GRKT/                # GRKT模型
│   ├── LPKT/                # LPKT模型
│   ├── MIKT/                # MIKT模型
│   ├── PKT/                 # PKT模型
│   ├── PSKT/                # PSKT模型
│   ├── QIKT/                # QIKT模型
│   ├── REKT/                # REKT模型
│   ├── SAINT/               # SAINT模型
│   ├── SAKT/                # SAKT模型
│   └── simpleKT/            # simpleKT模型
├── pre_process_data/        # 预处理后的数据目录（运行preprocess_data.py后生成）
│   └── {dataset}/           # 数据集名称，如assist09
│       └── {fold}/          # 交叉验证折数，0-4
│           ├── train_test/
│           │   ├── train_question.txt
│           │   └── test_question.txt
│           └── graph/
│               ├── ques_skill.csv
│               └── train_graphs.npz
└── output/                  # 输出目录（训练时自动创建）
    ├── best_model_fold_{fold}.pt  # 保存的最佳模型
    └── training_curve_fold_{fold}.png  # 训练曲线图
```   

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
本项目支持多个数据集，默认使用 ASSISTments 2009 数据集。可在 `config.py` 中修改 `DATASET` 参数切换数据集。

| 数据集 | 技能数 | 题目数 | 批次大小 |
|--------|--------|--------|----------|
| assist09 | 150 | 26,688 | 64 |
| assist12_1 | 265 | 53,091 | 50 |
| assist17 | 102 | 3,162 | 64 |
| junyi | 39 | 690 | 128 |
| eedi | 309 | 27,566 | 64 |
| ednet | 188 | 12,192 | 64 |
| xes3g5m | 865 | 7,652 | 15 |

**注意**：使用不同数据集时，需要在 `config.py` 中修改对应的 `NUM_SKILL` 和 `NUM_PROBLEM` 参数。