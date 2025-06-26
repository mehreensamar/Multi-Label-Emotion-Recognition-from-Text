# Multi-Label-Emotion-Recognition-from-Text
# 🎭 Multi-Label Emotion Recognition from Text

This project detects multiple emotions from textual data using fine-tuned BERT on the GoEmotions dataset.

## 📌 Objective
To develop a transformer-based system that identifies multiple emotional tones like joy, anger, sadness, etc., from customer feedback, social media, or any text data.

## 📂 Dataset
- **Name:** GoEmotions
- **Source:** Google Research
- **Classes:** 27 emotion labels + neutral

## 🚀 Model
- **Base:** BERT (`bert-base-uncased`)
- **Method:** Fine-tuned for multi-label classification
- **Metrics:** Hamming Loss, F1 Score

## 💻 Usage
```bash
# Install requirements
pip install -r requirements.txt
