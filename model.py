from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn

def get_model(num_labels):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    return model

def get_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")
