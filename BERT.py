import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# 加载数据
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", header=None)

# 随机选10%的数据
data_0 = df[df[0] == 0].sample(frac=0.1, random_state=42)
data_4 = df[df[0] == 4].sample(frac=0.1, random_state=42)
df = pd.concat([data_0, data_4])

# 删除无关列并清洗数据
df = df.drop(df.columns[1:5], axis=1)
df.columns = ['label', 'text']
df['label'] = df['label'].replace({0: 0, 4: 1})  # 转换为二分类



# 数据划分
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# 将数据转换为 Hugging Face Dataset 格式
train_data = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_data = Dataset.from_dict({"text": test_texts, "label": test_labels})
dataset = DatasetDict({"train": train_data, "test": test_data})

# 加载分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 数据分词
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

encoded_dataset = dataset.map(tokenize_function, batched=True)

# 加载模型
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


# 定义评估函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,              # 每 500 步评估一次
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,           # 每 100 步记录日志
                     
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # 添加准确率评估
)

# 模型训练
trainer.train()

# 模型评估
predictions = trainer.predict(encoded_dataset["test"])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = encoded_dataset["test"]["label"]

print(classification_report(y_true, y_pred))

# 保存模型和分词器
model.save_pretrained("bert_sentiment_model")
tokenizer.save_pretrained("bert_sentiment_model")