import pandas as pd
import torch
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss
from data_loader import load_emotion_data
from model import get_model, get_tokenizer
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# Load data
df, label_names = load_emotion_data("20370d11-694c-4e3d-9d2c-7e724d9e1f04.csv")
tokenizer = get_tokenizer()
encodings = tokenizer(df['text'].tolist(), truncation=True, padding=True, max_length=128)
labels = df['labels'].tolist()

# Split and prepare
X_train, X_test, y_train, y_test = train_test_split(encodings, labels, test_size=0.2, random_state=42)
train_dataset = EmotionDataset(X_train, y_train)
test_dataset = EmotionDataset(X_test, y_test)

model = get_model(len(label_names))

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
