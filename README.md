# Sentiment Analysis with BERT

This project uses a pre-trained BERT transformer model to perform binary sentiment analysis on a dataset. The model classifies text into two categories: **positive (1)** and **negative (0)**.

## Features

- Pre-processes and tokenizes data using Hugging Face's `transformers` library.
- Fine-tunes a `bert-base-uncased` model for sentiment classification.
- Evaluates the model's performance using metrics like accuracy and classification reports.
- Saves the fine-tuned model and tokenizer for future use.

## Dataset

The project uses the Sentiment140 dataset `training.1600000.processed.noemoticon.csv`, which contains 1.6 million labeled tweets. A subset (10%) of the data is randomly sampled for training and testing. However, the dataset is not included in the repository so you need to download it from elsewhere. 

#### Data Preprocessing

- The dataset is reduced to two columns: `label` (0 for negative, 4 for positive) and `text`.
- Labels are converted to a binary format: **0 (negative)** and **1 (positive)**.
- Data is split into training (80%) and testing (20%) sets.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/error-surface/Inspired.git
   cd Inspired
   ```

2. Install dependencies:

   ```bash
   pip install pandas scikit-learn transformers datasets torch
   ```

3. Download the Sentiment140 dataset from elsewhere and place it in the root directory as `training.1600000.processed.noemoticon.csv`.

## Usage

1. Run the script to train and evaluate the model:

   ```python
   python BERT.py
   ```

2. Fine-tuned model and tokenizer will be saved in the`bert_sentiment_model`directory. 

## Results

#### Model Training Parameters

- Batch Size: 16
- Learning Rate: 2e-5
- Epochs: 2
- Evaluation Strategy: Evaluate every 500 steps
- Save Strategy: Save at the end
- Logging Steps: Logs training metrics every 100 steps

#### Saved Outputs

- Model: `bert_sentiment_model/model.safetensors`
- Tokenizer: `bert_sentiment_model/tokenizer_config.json`
- Configuration: `bert_sentiment_model/config.json`
- Special Tokenizer: `bert_sentiment_model/special_tokens_map.json`
- Vocabulary: `bert_sentiment_model/vocab.txt`

## License

This project is licensed under the [MIT License](LICENSE)

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Sentiment140 dataset for providing the labeld data for sentiment analysis.

##  After the model is trained

#### Model Files

The model and tokenizer are stored in the`bert_sentiment_model `directory, which includes the following files:

​	•	`model.safetensor`: The fine-tuned model’s weights.

​	•	`config.json`: The model configuration.

​	•	`vocab.txt`: The tokenizer vocabulary.

​	•	`special_tokens_map.json`: The mapping for special tokens.

​	•    `tokenizer_config.json`: The tokenizer configuration.

#### How to use the model

Run the script to use the model:

```bash
python test BERT.py
```

You will see the following prompt:

```bash
Sentiment Prediction Script
Please enter a sentence to analyze:
>> 
```

For example, if you enter:

```bash
What a happy day!
```

The expected output will be: 

```bash
Text: What a happy day!
Predicted Sentiment: Positive
```
