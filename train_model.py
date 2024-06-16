import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

# Load dataset
df = pd.read_csv('data/mental_health_data.csv')

# Preprocess data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(df['text'].tolist(), return_tensors='pt', max_length=128, padding=True, truncation=True)
labels = torch.tensor([1 if label == 'positive' else 0 for label in df['label']])

# Create DataLoader
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=2)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
def train_model(model, dataloader):
    model.train()
    for batch in dataloader:
        b_input_ids, b_input_mask, b_labels = batch
        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Train the model
train_model(model, train_dataloader)

# Save the model
model.save_pretrained('saved_model')
tokenizer.save_pretrained('saved_model')
