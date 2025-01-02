from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 1. 加载模型和分词器
model_path = "bert_sentiment_model"  # 替换为你的模型路径
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 2. 定义预测函数
def predict_sentiment(text):
    # 分词并生成输入张量
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # 模型预测
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # 获取预测标签
    predicted_label = torch.argmax(logits, dim=1).item()
    
    # 将标签转换为对应情感
    label_map = {0: "Negative", 1: "Positive"}  # 根据训练时的标签映射调整
    return label_map[predicted_label]

# 3. 交互式输入
if __name__ == "__main__":
    print("Sentiment Prediction Script")
    print("Please enter a sentence to analyze:")
    input_text = input(">> ")  # 提示用户输入文本

    # 预测情感
    result = predict_sentiment(input_text)
    print(f"Text: {input_text}\nPredicted Sentiment: {result}")