# Transformers 快速参考指南

## 三个核心组件对比

| 组件 | 作用 | 使用场景 | 优点 | 缺点 |
|------|------|----------|------|------|
| **Tokenizer** | 文本 ↔ 数字转换 | 需要自定义预处理 | 完全控制 | 需要手动处理 |
| **Model** | 神经网络推理 | 需要获取中间结果 | 灵活强大 | 代码较复杂 |
| **Pipeline** | 端到端解决方案 | 快速原型开发 | 简单易用 | 灵活性较低 |

## 快速开始

### 1. 使用 Pipeline（最简单）

```python
from transformers import pipeline

# 一行代码完成情感分析
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

### 2. 使用 Tokenizer + Model（更灵活）

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载组件
tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModelForSequenceClassification.from_pretrained("model-name")

# 编码
inputs = tokenizer("I love this!", return_tensors="pt")

# 推理
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

# 获取结果
predicted_id = probabilities.argmax().item()
label = model.config.id2label[predicted_id]
print(f"预测: {label}")
```

### 3. 只使用 Tokenizer（文本处理）

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 编码
encoded = tokenizer("Hello world!", return_tensors="pt")
print(encoded['input_ids'])

# 解码
decoded = tokenizer.decode(encoded['input_ids'][0])
print(decoded)
```

## 常用任务 Pipeline

```python
from transformers import pipeline

# 情感分析
sentiment = pipeline("sentiment-analysis")

# 文本生成
generator = pipeline("text-generation", model="gpt2")

# 命名实体识别
ner = pipeline("ner")

# 问答
qa = pipeline("question-answering")

# 文本摘要
summarizer = pipeline("summarization")

# 翻译
translator = pipeline("translation_en_to_fr")

# 零样本分类
classifier = pipeline("zero-shot-classification")
```

## Tokenizer 常用参数

```python
tokenizer(
    text,                      # 输入文本
    max_length=512,            # 最大长度
    truncation=True,           # 是否截断
    padding=True,              # 是否填充
    return_tensors="pt",       # 返回 PyTorch tensor
    add_special_tokens=True,   # 添加特殊标记
)
```

## 模型类型

- `AutoModel` - 基础模型，输出隐藏状态
- `AutoModelForSequenceClassification` - 序列分类
- `AutoModelForQuestionAnswering` - 问答
- `AutoModelForTokenClassification` - 标记分类（NER）
- `AutoModelForCausalLM` - 因果语言模型（文本生成）

## 工作流程

```
文本输入
   ↓
Tokenizer (编码)
   ↓
Token IDs + Attention Mask
   ↓
Model (推理)
   ↓
Logits / Hidden States
   ↓
后处理 (Softmax, Argmax)
   ↓
最终结果
```

## 性能优化技巧

1. **使用 `torch.no_grad()`** - 推理时禁用梯度计算
   ```python
   with torch.no_grad():
       outputs = model(**inputs)
   ```

2. **批量处理** - 一次处理多个样本
   ```python
   texts = ["text1", "text2", "text3"]
   results = pipeline(texts)  # 批量处理
   ```

3. **模型评估模式** - 使用 `model.eval()`
   ```python
   model.eval()  # 关闭 dropout 等训练特性
   ```

4. **GPU 加速** - 如果有 GPU
   ```python
   model = model.to("cuda")
   inputs = inputs.to("cuda")
   ```

## 常见问题

### Q: 如何选择使用 Pipeline 还是手动组合？
A: 
- **Pipeline**: 快速原型、简单任务、学习阶段
- **手动组合**: 生产环境、自定义需求、性能优化

### Q: Tokenizer 和 Model 必须匹配吗？
A: 是的，必须使用同一个模型系列的 Tokenizer 和 Model。

### Q: 如何查看模型支持的标签？
A: 
```python
model.config.id2label  # ID 到标签的映射
model.config.label2id  # 标签到 ID 的映射
```

### Q: 如何保存和加载模型？
A: 
```python
# 保存
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# 加载
model = AutoModel.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")
```

## 学习路径建议

1. **入门**: 先学习 `learn_transformers.py`，理解三个核心组件
2. **实践**: 运行 `transformers_examples.py`，查看实际应用
3. **深入**: 阅读官方文档，了解高级特性
4. **应用**: 在自己的项目中使用和实验

## 有用的资源

- [Hugging Face 官方文档](https://huggingface.co/docs/transformers)
- [模型库](https://huggingface.co/models)
- [Pipeline 文档](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [课程](https://huggingface.co/course)

