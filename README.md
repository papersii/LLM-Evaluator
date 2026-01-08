# LLM-Evaluator

<div align="center">

**[English](#english) | [中文](#chinese)**

A lightweight, flexible LLM evaluation framework for academic research and benchmarking.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

<a name="english"></a>
## 🇬🇧 English

### 📖 Overview

LLM-Evaluator is a modular evaluation pipeline designed for testing Large Language Models on custom datasets. It supports multiple answer formats (LaTeX, GSM8K, natural language) and provides an extensible framework for academic evaluation tasks.

### ✨ Features

- 🎯 **Multi-format Answer Extraction**: Supports `\boxed{}`, GSM8K format, and natural language patterns
- 🔄 **Automatic Retry Mechanism**: Handles API rate limits and network issues gracefully
- 📊 **JSONL Dataset Format**: Easy to create and maintain test cases
- 🌐 **Hugging Face Router Support**: Compatible with various LLM providers
- ⚙️ **Environment-based Configuration**: Secure API key management via `.env`
- 🧪 **Reproducible Results**: Zero temperature for deterministic evaluation
- 📈 **Interactive Dashboard**: Streamlit interface for result visualization and analysis

### 🚀 Quick Start

#### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd LLM-Evaluator-1

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Configuration

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
HF_API_BASE=https://router.huggingface.co/v1
DEFAULT_MODEL=deepseek-ai/DeepSeek-V3.2:novita
```

#### 3. Prepare Test Data

Create your test cases in `data/test_cases.jsonl`:

```json
{"id": 1, "question": "If x + 5 = 12, what is x?", "answer": "7"}
{"id": 2, "question": "What is 15 multiplied by 3?", "answer": "45"}
```

#### 4. Run Evaluation

```bash
# Using the shell script (recommended)
./run_eval.sh

# Or directly with Python
python main.py --data_path data/test_cases.jsonl
```

#### 5. Data Visualization (Optional)

Generate visual charts to analyze results:

```bash
# Display charts interactively
python main.py --visualize

# Save charts to file
python main.py --save-viz evaluation_results.png
```

**Example Output:**

![Visualization Example](/Users/papersiii/.gemini/antigravity/brain/c979daa4-a75f-470d-8e3f-9993c807669d/evaluation_results.png)

- **Pie Chart**: Overall accuracy distribution (correct vs incorrect)
- **Bar Chart**: Per-question results (green = correct, red = incorrect)

#### 6. Interactive Dashboard

Launch the web interface for an easier evaluation experience:

```bash
streamlit run app.py
```

**Features:**
- 📁 **File Upload**: Drag and drop your JSONL datasets
- ⚡ **Async Evaluation**: Concurrent processing with rate limiting
- 🕸️ **Radar Chart**: Visual analysis of model performance across categories
- 📊 **Detailed Metrics**: View accuracy trends and specific failure cases

### 📁 Project Structure

```
LLM-Evaluator-1/
├── main.py                 # Main evaluation script
├── run_eval.sh            # Execution wrapper with auto-detection
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration (create this)
├── data/
│   └── test_cases.jsonl  # Test dataset
└── src/
    ├── model_client.py   # LLM API client
    └── scorer.py         # Answer extraction and scoring logic
```

### 🔧 Advanced Usage

#### Custom Data Path

```bash
./run_eval.sh path/to/custom_data.jsonl
```

#### Programmatic Usage

```python
from src.model_client import LLMClient
from src.scorer import exact_match_scorer

client = LLMClient()
response = client.get_response("What is 2 + 2?")
is_correct = exact_match_scorer(response, "4")
```

### 📊 Supported Answer Formats

The scorer automatically extracts answers from:

- **LaTeX**: `\boxed{42}`
- **GSM8K**: `#### 42`
- **Natural Language**: `"The answer is: 42"`
- **Chinese**: `"答案是：42"`
- **Plain Numbers**: Last number in the response

### 🛠️ Customization

#### Adding New Scoring Patterns

Edit `src/scorer.py` and add patterns to the `patterns` list:

```python
patterns = [
    r'\\boxed\{([^}]+)\}',
    r'your_custom_pattern_here',
    # ...
]
```

#### Using Different Models

Update `DEFAULT_MODEL` in your `.env` file:

```env
DEFAULT_MODEL=your-model-name
```

### 📝 Example Output

```
Using Python: .venv/bin/python
Python version: Python 3.14.0

Starting evaluation with data: data/test_cases.jsonl
----------------------------------------
Loading data from data/test_cases.jsonl...
Evaluating ID 1...
Evaluating ID 2...
...
------------------------------
Evaluation Finished!
Final Accuracy: 93.33%
```

### 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests

### 📄 License

This project is licensed under the MIT License.

---

<a name="chinese"></a>
## 🇨🇳 中文

### 📖 项目简介

LLM-Evaluator 是一个轻量级、模块化的大语言模型评测框架，专为学术研究和基准测试设计。支持多种答案格式（LaTeX、GSM8K、自然语言），提供可扩展的评测任务框架。

### ✨ 核心特性

- 🎯 **多格式答案提取**：支持 `\boxed{}`、GSM8K 格式和自然语言模式
- 🔄 **自动重试机制**：优雅处理 API 限流和网络问题
- 📊 **JSONL 数据集格式**：易于创建和维护测试用例
- 🌐 **Hugging Face Router 支持**：兼容多种 LLM 提供商
- ⚙️ **环境变量配置**：通过 `.env` 安全管理 API 密钥
- 🧪 **可复现结果**：零温度参数确保评测结果一致
- 📈 **交互式仪表板**：基于 Streamlit 的可视化评测与分析界面

### 🚀 快速开始

#### 1. 安装

```bash
# 克隆仓库
git clone <your-repo-url>
cd LLM-Evaluator-1

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

#### 2. 配置

在项目根目录创建 `.env` 文件：

```env
HF_TOKEN=your_huggingface_token_here
HF_API_BASE=https://router.huggingface.co/v1
DEFAULT_MODEL=deepseek-ai/DeepSeek-V3.2:novita
```

#### 3. 准备测试数据

在 `data/test_cases.jsonl` 中创建测试用例：

```json
{"id": 1, "question": "If x + 5 = 12, what is x?", "answer": "7"}
{"id": 2, "question": "What is 15 multiplied by 3?", "answer": "45"}
```

#### 4. 运行评测

```bash
# 使用 Shell 脚本（推荐）
./run_eval.sh

# 或直接使用 Python
python main.py --data_path data/test_cases.jsonl
```

#### 5. 启动交互式仪表板

使用 Web 界面进行更直观的评测：

```bash
streamlit run app.py
```

**功能特性：**
- 📁 **文件上传**：直接拖拽 JSONL 数据集
- ⚡ **异步评测**：支持并发处理与自动限流
- 🕸️ **雷达图分析**：多维度展示模型能力
- 📊 **详细指标**：实时查看准确率与具体错误用例


### 📁 项目结构

```
LLM-Evaluator-1/
├── main.py                 # 主评测脚本
├── run_eval.sh            # 执行包装器（自动检测环境）
├── requirements.txt       # Python 依赖
├── .env                   # 环境配置（需自行创建）
├── data/
│   └── test_cases.jsonl  # 测试数据集
└── src/
    ├── model_client.py   # LLM API 客户端
    └── scorer.py         # 答案提取与评分逻辑
```

### 🔧 高级用法

#### 自定义数据路径

```bash
./run_eval.sh path/to/custom_data.jsonl
```

#### 编程式调用

```python
from src.model_client import LLMClient
from src.scorer import exact_match_scorer

client = LLMClient()
response = client.get_response("What is 2 + 2?")
is_correct = exact_match_scorer(response, "4")
```

### 📊 支持的答案格式

评分器自动从以下格式中提取答案：

- **LaTeX 格式**：`\boxed{42}`
- **GSM8K 格式**：`#### 42`
- **自然语言**：`"The answer is: 42"`
- **中文格式**：`"答案是：42"`
- **纯数字**：回复中的最后一个数字

### 🛠️ 自定义扩展

#### 添加新的评分模式

编辑 `src/scorer.py`，在 `patterns` 列表中添加自定义模式：

```python
patterns = [
    r'\\boxed\{([^}]+)\}',
    r'你的自定义模式',
    # ...
]
```

#### 使用不同的模型

更新 `.env` 文件中的 `DEFAULT_MODEL`：

```env
DEFAULT_MODEL=你的模型名称
```

### 📝 示例输出

```
Using Python: .venv/bin/python
Python version: Python 3.14.0

Starting evaluation with data: data/test_cases.jsonl
----------------------------------------
Loading data from data/test_cases.jsonl...
Evaluating ID 1...
Evaluating ID 2...
...
------------------------------
Evaluation Finished!
Final Accuracy: 93.33%
```

### 🤝 贡献指南

欢迎贡献！您可以：

- 报告 Bug
- 提出新功能建议
- 提交 Pull Request

### 📄 开源协议

本项目采用 MIT 协议开源。

---

<div align="center">
Made with ❤️ for LLM Research
</div>