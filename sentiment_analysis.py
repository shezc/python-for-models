"""
使用 Transformers 进行情感分析的示例脚本
"""
from transformers import pipeline
import pandas as pd

def analyze_sentiment(text):
    """
    分析单个文本的情感
    
    Args:
        text (str): 要分析的文本
        
    Returns:
        dict: 包含情感标签和置信度的字典
    """
    # 使用预训练的情感分析模型
    # 这个模型支持中文和英文
    classifier = pipeline("sentiment-analysis", 
                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    result = classifier(text)
    return result[0]

def analyze_batch(texts):
    """
    批量分析多个文本的情感
    
    Args:
        texts (list): 要分析的文本列表
        
    Returns:
        list: 包含每个文本情感分析结果的列表
    """
    classifier = pipeline("sentiment-analysis",
                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    results = classifier(texts)
    return results

def main():
    """
    主函数：演示情感分析的使用
    """
    print("=" * 50)
    print("Transformers 情感分析示例")
    print("=" * 50)
    print()
    
    # 示例1: 单个文本分析
    print("示例1: 单个文本情感分析")
    print("-" * 50)
    test_text = "I love this product! It's amazing and works perfectly."
    result = analyze_sentiment(test_text)
    print(f"文本: {test_text}")
    print(f"情感: {result['label']}")
    print(f"置信度: {result['score']:.4f}")
    print()
    
    # 示例2: 批量分析
    print("示例2: 批量文本情感分析")
    print("-" * 50)
    test_texts = [
        "This is the best day ever!",
        "I'm feeling really sad today.",
        "The weather is okay, nothing special.",
        "I'm extremely happy with the service!",
        "This is terrible, I hate it."
    ]
    
    results = analyze_batch(test_texts)
    
    # 创建DataFrame以便更好地展示结果
    df = pd.DataFrame({
        '文本': test_texts,
        '情感标签': [r['label'] for r in results],
        '置信度': [r['score'] for r in results]
    })
    
    print(df.to_string(index=False))
    print()
    
    # 示例3: 中文文本分析（使用中文模型）
    print("示例3: 中文文本情感分析")
    print("-" * 50)
    try:
        # 使用支持中文的情感分析模型
        chinese_classifier = pipeline("sentiment-analysis",
                                     model="uer/roberta-base-finetuned-dianping-chinese")
        
        chinese_texts = [
            "这个产品太棒了！我非常喜欢。",
            "今天心情很糟糕，什么都不顺利。",
            "一般般吧，没什么特别的。"
        ]
        
        chinese_results = chinese_classifier(chinese_texts)
        
        chinese_df = pd.DataFrame({
            '文本': chinese_texts,
            '情感标签': [r['label'] for r in chinese_results],
            '置信度': [r['score'] for r in chinese_results]
        })
        
        print(chinese_df.to_string(index=False))
    except Exception as e:
        print(f"中文模型加载失败: {e}")
        print("提示: 如果网络较慢，可能需要等待模型下载完成")
    
    print()
    print("=" * 50)
    print("分析完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()

