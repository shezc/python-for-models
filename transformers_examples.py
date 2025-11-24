"""
Transformers 实战示例集合
=====================================
包含多个实际应用场景的完整示例代码
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    pipeline
)
import torch

# ============================================================================
# 示例1：情感分析（完整流程）
# ============================================================================
def example_sentiment_analysis():
    """情感分析完整示例"""
    print("\n" + "=" * 60)
    print("示例1：情感分析（完整流程）")
    print("=" * 60)
    
    # 方式1：使用 Pipeline（最简单）
    print("\n方式1：使用 Pipeline")
    print("-" * 60)
    classifier = pipeline("sentiment-analysis")
    text = "I love this product!"
    result = classifier(text)
    print(f"文本: {text}")
    print(f"结果: {result}")
    
    # 方式2：手动组合（更灵活）
    print("\n方式2：手动组合 Tokenizer + Model")
    print("-" * 60)
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    # 编码
    inputs = tokenizer(text, return_tensors="pt")
    
    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # 获取结果
    predicted_id = probabilities.argmax().item()
    confidence = probabilities.max().item()
    label = model.config.id2label[predicted_id]
    
    print(f"文本: {text}")
    print(f"预测: {label}, 置信度: {confidence:.4f}")


# ============================================================================
# 示例2：批量处理文本
# ============================================================================
def example_batch_processing():
    """批量处理文本示例"""
    print("\n" + "=" * 60)
    print("示例2：批量处理文本")
    print("=" * 60)
    
    texts = [
        "This is amazing!",
        "I hate this.",
        "It's okay, nothing special.",
        "Best product ever!",
        "Terrible experience."
    ]
    
    # 使用 Pipeline 批量处理
    classifier = pipeline("sentiment-analysis")
    results = classifier(texts)
    
    print("\n批量分析结果:")
    print("-" * 60)
    for text, result in zip(texts, results):
        print(f"{text:30} -> {result['label']:15} ({result['score']:.4f})")


# ============================================================================
# 示例3：自定义 Tokenizer 参数
# ============================================================================
def example_custom_tokenizer():
    """自定义 Tokenizer 参数示例"""
    print("\n" + "=" * 60)
    print("示例3：自定义 Tokenizer 参数")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    text = "This is a very long text that needs to be truncated because it exceeds the maximum length allowed by the model."
    
    # 默认编码
    encoded_default = tokenizer(text, return_tensors="pt")
    print(f"原始文本长度: {len(text.split())} 词")
    print(f"默认编码长度: {encoded_default['input_ids'].shape[1]} tokens")
    
    # 自定义参数编码
    encoded_custom = tokenizer(
        text,
        max_length=32,          # 最大长度
        truncation=True,         # 截断
        padding="max_length",    # 填充到最大长度
        return_tensors="pt"
    )
    print(f"自定义编码长度: {encoded_custom['input_ids'].shape[1]} tokens")
    print(f"填充后形状: {encoded_custom['input_ids'].shape}")


# ============================================================================
# 示例4：获取模型的隐藏状态
# ============================================================================
def example_hidden_states():
    """获取模型隐藏状态示例"""
    print("\n" + "=" * 60)
    print("示例4：获取模型的隐藏状态")
    print("=" * 60)
    
    from transformers import AutoModel
    
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    text = "Hello, transformers!"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取最后一层的隐藏状态
    last_hidden_state = outputs.last_hidden_state
    print(f"隐藏状态形状: {last_hidden_state.shape}")
    print(f"说明: [批次大小=1, 序列长度={last_hidden_state.shape[1]}, 隐藏维度={last_hidden_state.shape[2]}]")
    
    # 获取 [CLS] token 的表示（通常用于句子级别的任务）
    cls_embedding = last_hidden_state[0, 0, :]  # 第一个样本，第一个token，所有维度
    print(f"[CLS] token 嵌入维度: {cls_embedding.shape}")


# ============================================================================
# 示例5：问答任务
# ============================================================================
def example_question_answering():
    """问答任务示例"""
    print("\n" + "=" * 60)
    print("示例5：问答任务")
    print("=" * 60)
    
    # 使用 Pipeline
    qa_pipeline = pipeline("question-answering")
    
    context = """
    Transformers is a library by Hugging Face that provides thousands of 
    pre-trained models for Natural Language Processing tasks. It supports 
    both PyTorch and TensorFlow frameworks.
    """
    
    question = "What is Transformers?"
    
    result = qa_pipeline(question=question, context=context)
    print(f"问题: {question}")
    print(f"上下文: {context.strip()}")
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['score']:.4f}")


# ============================================================================
# 示例6：零样本分类
# ============================================================================
def example_zero_shot_classification():
    """零样本分类示例"""
    print("\n" + "=" * 60)
    print("示例6：零样本分类")
    print("=" * 60)
    
    classifier = pipeline("zero-shot-classification")
    
    text = "I love programming in Python!"
    candidate_labels = ["programming", "cooking", "travel", "sports"]
    
    result = classifier(text, candidate_labels)
    
    print(f"文本: {text}")
    print(f"候选标签: {candidate_labels}")
    print(f"\n分类结果:")
    for label, score in zip(result['labels'], result['scores']):
        print(f"  {label:15} : {score:.4f}")


# ============================================================================
# 示例7：模型配置信息
# ============================================================================
def example_model_config():
    """查看模型配置信息"""
    print("\n" + "=" * 60)
    print("示例7：模型配置信息")
    print("=" * 60)
    
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    config = model.config
    print(f"模型名称: {model_name}")
    print(f"模型类型: {config.model_type}")
    print(f"隐藏层大小: {config.hidden_size}")
    print(f"注意力头数: {config.num_attention_heads}")
    print(f"层数: {config.num_hidden_layers}")
    print(f"标签数量: {config.num_labels}")
    print(f"标签映射: {config.id2label}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    """运行所有示例"""
    print("=" * 60)
    print("Transformers 实战示例集合")
    print("=" * 60)
    print("\n注意: 首次运行会下载模型，可能需要一些时间...")
    
    # 运行示例（可以注释掉不需要的示例）
    example_sentiment_analysis()
    example_batch_processing()
    example_custom_tokenizer()
    example_hidden_states()
    # example_question_answering()  # 需要下载问答模型，较慢
    # example_zero_shot_classification()  # 需要下载零样本分类模型
    example_model_config()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

