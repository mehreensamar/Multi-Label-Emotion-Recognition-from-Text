import pandas as pd

def load_emotion_data(path):
    df = pd.read_csv(path)
    df.dropna(subset=['text'], inplace=True)
    labels = df.columns[9:]  # Based on your shared columns
    df['labels'] = df[labels].values.tolist()
    return df[['text', 'labels']], labels.tolist()
