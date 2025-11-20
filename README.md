# Transformers 情感分析项目

这是一个使用 Hugging Face Transformers 库进行情感分析的 Python 项目示例。

## 功能特性

- 使用预训练模型进行情感分析
- 支持单个文本和批量文本分析
- 支持英文和中文文本分析
- 提供清晰的输出结果

## 环境要求

- Python 3.8 或更高版本
- pip 包管理器

## 安装步骤

### 方法1：使用自动设置脚本（推荐）

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

如果遇到执行策略错误，请先运行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows (CMD):**
```cmd
setup.bat
```

脚本会自动完成：创建虚拟环境、激活环境、安装依赖等所有步骤。

### 方法2：手动安装

#### 1. 创建虚拟环境

在项目根目录下，运行以下命令创建虚拟环境：

**Windows (PowerShell):**
```powershell
python -m venv venv
```

**Windows (CMD):**
```cmd
python -m venv venv
```

**Linux/Mac:**
```bash
python3 -m venv venv
```

#### 2. 激活虚拟环境

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

如果遇到执行策略错误，请先运行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

#### 3. 安装依赖

激活虚拟环境后，运行：

```bash
pip install -r requirements.txt
```

#### 4. 运行示例

```bash
python sentiment_analysis.py
```

## 项目结构

```
python-for-models/
├── venv/                  # 虚拟环境目录（运行后生成）
├── requirements.txt       # 项目依赖
├── sentiment_analysis.py  # 主程序文件
└── README.md             # 项目说明文档
```

## 使用说明

### 基本用法

```python
from transformers import pipeline

# 创建情感分析管道
classifier = pipeline("sentiment-analysis")

# 分析文本
result = classifier("I love this product!")
print(result)
```

### 自定义模型

项目中使用的是 `cardiffnlp/twitter-roberta-base-sentiment-latest` 模型，你也可以替换为其他模型：

- `distilbert-base-uncased-finetuned-sst-2-english` - 轻量级英文模型
- `nlptown/bert-base-multilingual-uncased-sentiment` - 多语言模型
- `uer/roberta-base-finetuned-dianping-chinese` - 中文模型

## 注意事项

1. **首次运行**: 首次运行时会自动下载预训练模型，可能需要一些时间，请确保网络连接正常。

2. **模型大小**: 预训练模型可能较大（几百MB到几GB），请确保有足够的磁盘空间。

3. **GPU支持**: 如果有NVIDIA GPU并安装了CUDA，可以加速推理。需要安装对应的PyTorch版本。

4. **内存要求**: 建议至少4GB可用内存。

## 常见问题

### Q: setup.ps1 脚本执行失败怎么办？

**问题1: 执行策略限制**
```
错误: 无法加载文件，因为在此系统上禁止运行脚本
```
**解决方案:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**问题2: Python 未找到**
```
错误: 未找到 Python
```
**解决方案:**
- 确保已安装 Python 3.8 或更高版本
- 检查 Python 是否添加到系统 PATH 环境变量
- 尝试使用 `py` 命令代替 `python`：`py -m venv venv`

**问题3: 虚拟环境创建失败**
**解决方案:**
- 手动创建虚拟环境：`python -m venv venv`
- 如果使用 `python` 命令失败，尝试 `py -m venv venv`
- 确保有足够的磁盘空间和写入权限

**问题4: 依赖安装失败**
**解决方案:**
- 检查网络连接
- 尝试使用国内镜像源：
  ```powershell
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- 手动激活虚拟环境后安装：
  ```powershell
  .\venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

### Q: 模型下载很慢怎么办？
A: 可以设置镜像源或使用代理。也可以手动下载模型到本地。

### Q: 如何离线使用？
A: 首次运行后，模型会缓存在 `~/.cache/huggingface/` 目录，之后可以离线使用。

### Q: 支持哪些语言？
A: 不同的模型支持不同的语言。项目中的英文模型主要支持英文，中文模型支持中文。

## 参考资料

- [Hugging Face Transformers 文档](https://huggingface.co/docs/transformers)
- [Pipeline API 文档](https://huggingface.co/docs/transformers/main_classes/pipelines)

## 许可证

本项目仅用于学习和演示目的。

