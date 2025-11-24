"""
Transformers 核心组件学习指南
=====================================
本文件详细讲解 Hugging Face Transformers 的三个核心组件：
1. Tokenizer（分词器）
2. Model（模型）
3. Pipeline（管道）

每个组件都有详细的说明和实际示例。
"""

from transformers import (
    AutoTokenizer,      # 自动加载分词器
    AutoModel,          # 自动加载模型
    AutoModelForSequenceClassification,  # 自动加载分类模型
    pipeline             # 高级API管道
)
import torch

print("=" * 60)
print("Transformers 核心组件学习指南")
print("=" * 60)
print()


# ============================================================================
# 第一部分：Tokenizer（分词器）
# ============================================================================
print("\n" + "=" * 60)
print("第一部分：Tokenizer（分词器）")
print("=" * 60)
print("""
Tokenizer 的作用：
1. 将文本转换为模型可以理解的数字（token IDs）
2. 将数字转换回文本
3. 处理特殊标记（如 [CLS], [SEP], [PAD] 等）
4. 处理注意力掩码（attention mask）
""")

# 示例1：加载和使用 Tokenizer
print("\n【示例1】加载 Tokenizer")
print("-" * 60)
try:
    # 尝试使用已下载的模型，如果没有则下载
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"✓ Tokenizer 加载成功: {type(tokenizer).__name__}")
except Exception as e:
    print(f"⚠ 模型下载失败（网络问题）: {str(e)[:100]}")
    print("提示: 如果之前已经下载过模型，可以尝试使用本地路径")
    print("或者检查网络连接后重试")
    # 尝试使用已下载的情感分析模型的 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        print(f"✓ 使用已下载的 Tokenizer: {type(tokenizer).__name__}")
    except:
        print("✗ 无法加载 Tokenizer，请检查网络连接或使用已下载的模型")
        tokenizer = None

# 示例2：基本编码
if tokenizer is not None:
    print("\n【示例2】文本编码（Text → Token IDs）")
    print("-" * 60)
    text = "Hello, I love transformers!"
    encoded = tokenizer(text, return_tensors="pt")  # pt = PyTorch tensor
    print(f"原始文本: {text}")
    print(f"Token IDs: {encoded['input_ids']}")
    print(f"Token IDs 形状: {encoded['input_ids'].shape}")

    # 解码查看实际的 tokens
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    print(f"Tokens: {tokens}")
else:
    print("\n【示例2】跳过（Tokenizer 未加载）")

# 示例3：批量编码
if tokenizer is not None:
    print("\n【示例3】批量编码")
    print("-" * 60)
    texts = [
        "I love this product!",
        "This is terrible.",
        "It's okay, nothing special."
    ]
    encoded_batch = tokenizer(
        texts,
        padding=True,           # 自动填充到相同长度
        truncation=True,         # 自动截断到最大长度
        max_length=128,          # 最大长度
        return_tensors="pt"
    )
    print(f"批量文本数量: {len(texts)}")
    print(f"编码后形状: {encoded_batch['input_ids'].shape}")
    print(f"注意力掩码形状: {encoded_batch['attention_mask'].shape}")

    # 示例4：解码
    print("\n【示例4】Token IDs → 文本（解码）")
    print("-" * 60)
    decoded_text = tokenizer.decode(encoded['input_ids'][0])
    print(f"解码后的文本: {decoded_text}")

    # 示例5：特殊标记
    print("\n【示例5】特殊标记说明")
    print("-" * 60)
    print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print(f"词汇表大小: {len(tokenizer)}")
else:
    print("\n【示例3-5】跳过（Tokenizer 未加载）")


# ============================================================================
# 第二部分：Model（模型）
# ============================================================================
print("\n\n" + "=" * 60)
print("第二部分：Model（模型）")
print("=" * 60)
print("""
Model 的作用：
1. 加载预训练的神经网络模型
2. 对编码后的输入进行前向传播
3. 输出隐藏状态或预测结果

模型类型：
- AutoModel: 基础模型，输出隐藏状态
- AutoModelForSequenceClassification: 分类模型，输出类别概率
- AutoModelForQuestionAnswering: 问答模型
- AutoModelForTokenClassification: 标记分类模型（如NER）
""")

# 示例1：加载基础模型
print("\n【示例1】加载基础模型（输出隐藏状态）")
print("-" * 60)
print("正在加载模型（首次运行会下载，可能需要几分钟）...")
try:
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()  # 设置为评估模式
    print(f"✓ 模型加载成功: {type(model).__name__}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    model_loaded = True
except Exception as e:
    print(f"⚠ 模型下载失败（网络问题）: {str(e)[:100]}")
    print("提示: 可以稍后重试，或使用已下载的模型")
    model = None
    model_loaded = False

# 示例2：使用模型进行推理
if model_loaded and tokenizer is not None:
    print("\n【示例2】模型推理（获取隐藏状态）")
    print("-" * 60)
    with torch.no_grad():  # 不计算梯度，节省内存
        outputs = model(**encoded)
        print(f"输出类型: {type(outputs)}")
        print(f"隐藏状态形状: {outputs.last_hidden_state.shape}")
        print(f"说明: [批次大小, 序列长度, 隐藏层维度]")
else:
    print("\n【示例2】跳过（模型或 Tokenizer 未加载）")

# 示例3：加载分类模型
print("\n【示例3】加载分类模型（输出类别概率）")
print("-" * 60)
print("正在加载情感分析分类模型...")
classifier_model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
classifier_model.eval()
print(f"✓ 分类模型加载成功")
print(f"分类标签数量: {classifier_model.config.num_labels}")

# 示例4：使用分类模型进行预测
print("\n【示例4】使用分类模型进行情感分析")
print("-" * 60)
# 使用之前加载的 tokenizer（需要匹配模型）
classifier_tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

test_text = "I love this product! It's amazing!"
encoded_input = classifier_tokenizer(test_text, return_tensors="pt")

with torch.no_grad():
    outputs = classifier_model(**encoded_input)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
print(f"输入文本: {test_text}")
print(f"Logits (原始分数): {logits}")
print(f"概率分布: {probabilities}")
print(f"预测类别: {probabilities.argmax().item()}")
print(f"置信度: {probabilities.max().item():.4f}")


# ============================================================================
# 第三部分：Pipeline（管道）
# ============================================================================
print("\n\n" + "=" * 60)
print("第三部分：Pipeline（管道）")
print("=" * 60)
print("""
Pipeline 的作用：
1. 封装了 Tokenizer 和 Model 的完整流程
2. 提供简单易用的高级API
3. 自动处理预处理和后处理
4. 支持多种任务：情感分析、文本生成、问答等

Pipeline 的优势：
- 简单易用，一行代码完成推理
- 自动处理所有细节
- 适合快速原型开发

Pipeline 的劣势：
- 灵活性较低
- 无法自定义中间步骤
- 性能可能不如手动组合
""")

# 示例1：创建情感分析 Pipeline
print("\n【示例1】创建情感分析 Pipeline")
print("-" * 60)
print("正在创建情感分析管道...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)
print("✓ Pipeline 创建成功")

# 示例2：使用 Pipeline 进行单文本分析
print("\n【示例2】使用 Pipeline 分析单个文本")
print("-" * 60)
text = "I'm so happy today!"
result = sentiment_pipeline(text)
print(f"输入: {text}")
print(f"结果: {result}")

# 示例3：批量处理
print("\n【示例3】使用 Pipeline 批量处理")
print("-" * 60)
texts = [
    "This is the best day ever!",
    "I'm feeling really sad.",
    "The weather is okay."
]
results = sentiment_pipeline(texts)
for text, result in zip(texts, results):
    print(f"文本: {text}")
    print(f"  情感: {result['label']}, 置信度: {result['score']:.4f}")

# 示例4：其他类型的 Pipeline
print("\n【示例4】其他类型的 Pipeline")
print("-" * 60)
print("""
Transformers 支持多种 Pipeline 任务：

1. 情感分析 (sentiment-analysis)
2. 文本生成 (text-generation)
3. 命名实体识别 (ner)
4. 问答 (question-answering)
5. 文本摘要 (summarization)
6. 翻译 (translation)
7. 零样本分类 (zero-shot-classification)

示例代码：
    # 文本生成
    generator = pipeline("text-generation", model="gpt2")
    result = generator("The future of AI is")
    
    # 命名实体识别
    ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    result = ner("My name is John and I live in New York")
    
    # 零样本分类
    classifier = pipeline("zero-shot-classification")
    result = classifier("This is a course about Python", 
                       candidate_labels=["education", "politics", "business"])
""")


# ============================================================================
# 第四部分：完整工作流程对比
# ============================================================================
print("\n\n" + "=" * 60)
print("第四部分：完整工作流程对比")
print("=" * 60)
print("""
下面展示两种方式完成相同任务：
1. 使用 Pipeline（简单）
2. 手动组合 Tokenizer + Model（灵活）
""")

test_text = "I love transformers library!"

# 方式1：使用 Pipeline（简单）
print("\n【方式1】使用 Pipeline（简单）")
print("-" * 60)
result_pipeline = sentiment_pipeline(test_text)
print(f"输入: {test_text}")
print(f"Pipeline 结果: {result_pipeline}")

# 方式2：手动组合（灵活）
print("\n【方式2】手动组合 Tokenizer + Model（灵活）")
print("-" * 60)
# 编码
encoded = classifier_tokenizer(test_text, return_tensors="pt")

# 模型推理
with torch.no_grad():
    outputs = classifier_model(**encoded)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

# 获取结果
predicted_class = probabilities.argmax().item()
confidence = probabilities.max().item()

# 获取标签名称（需要从模型配置中获取）
id2label = classifier_model.config.id2label
label = id2label[predicted_class]

print(f"输入: {test_text}")
print(f"手动组合结果: {{'label': '{label}', 'score': {confidence:.4f}}}")

print("\n" + "=" * 60)
print("学习总结")
print("=" * 60)
print("""
1. Tokenizer:
   - 负责文本的编码和解码
   - 处理特殊标记和填充
   - 是模型输入的第一步

2. Model:
   - 核心的神经网络模型
   - 执行实际的计算和推理
   - 输出隐藏状态或预测结果

3. Pipeline:
   - 封装了 Tokenizer + Model 的完整流程
   - 提供简单易用的高级API
   - 适合快速开发和原型验证

使用建议：
- 快速原型：使用 Pipeline
- 生产环境：手动组合 Tokenizer + Model（更好的控制和性能）
- 自定义需求：必须使用 Tokenizer + Model
""")
print("=" * 60)
print("学习完成！")
print("=" * 60)

