import json
import time
import warnings
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import LabelBinarizer
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
BATCH_SIZE = 10
LEARNING_RATE = 1e-5
EPOCH = 500
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(MLPClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.dropout(x)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


class CombinedModel(nn.Module):
    def __init__(self, bert_model, classifier):
        super(CombinedModel, self).__init__()
        self.bert = bert_model
        self.classifier = classifier

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)


class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.news_content = df["news_content"].to_numpy()
        self.impact_type = LabelBinarizer().fit_transform(df.impact_type).astype(np.float32)

    def __len__(self):
        return len(self.news_content)

    def __getitem__(self, idx):
        return {'text': self.news_content[idx], 'label': self.impact_type[idx]}


def get_model(model_name='bert-base-chinese', hidden_dim=1024, output_dim=5, dropout_rate=0.2):
    model_name = model_name

    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)

    input_dim = bert_model.config.hidden_size
    hidden_dim = hidden_dim
    output_dim = output_dim
    dropout_rate = dropout_rate

    classifier = MLPClassifier(input_dim, hidden_dim, output_dim, dropout_rate)
    classifier = CombinedModel(bert_model, classifier)
    return tokenizer, classifier.to(DEVICE)


def get_data(batch_size=64):
    train_df = pd.read_csv("static/Train.csv")[["news_content", "impact_type"]]
    dev_df = pd.read_csv("static/Dev.csv")[["news_content", "impact_type"]]

    train_dataset = CustomDataset(train_df)
    val_dataset = CustomDataset(dev_df)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
        DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


def train_model(tokenizer, classifier, criterion, optimizer, train_loader):
    train_loss = 0
    train_acc = 0

    classifier.train()
    for batch in train_loader:
        tokenizer_res = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
        input_ids = tokenizer_res['input_ids'].to(DEVICE)
        attention_mask = tokenizer_res['attention_mask'].to(DEVICE)
        labels = torch.tensor(batch['label']).to(DEVICE)

        optimizer.zero_grad()
        output = classifier(input_ids, attention_mask)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        output = torch.argmax(output, dim=1)
        labels = torch.argmax(labels, dim=1)
        train_acc += (labels == output).sum().item()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    return train_loss, train_acc


def valid_model(tokenizer, classifier, criterion, val_loader):
    valid_loss = 0
    valid_acc = 0

    classifier.eval()
    with torch.no_grad():
        for batch in val_loader:
            tokenizer_res = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
            input_ids = tokenizer_res['input_ids'].to(DEVICE)
            attention_mask = tokenizer_res['attention_mask'].to(DEVICE)
            labels = torch.tensor(batch['label']).to(DEVICE)

            output = classifier(input_ids, attention_mask)

            loss = criterion(output, labels)

            valid_loss += loss.item()
            output = torch.argmax(output, dim=1)
            labels = torch.argmax(labels, dim=1)
            valid_acc += (labels == output).sum().item()

    valid_loss /= len(val_loader)
    valid_acc /= len(val_loader)

    return valid_loss, valid_acc


if __name__ == "__main__":
    train_loader, val_loader = get_data(batch_size=BATCH_SIZE)

    tokenizer, classifier = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH):
        t1 = time.time()
        train_loss, train_acc = train_model(tokenizer, classifier, criterion, optimizer, train_loader)
        valid_loss, valid_acc = valid_model(tokenizer, classifier, criterion, val_loader)
        print(
            f"epoch:{epoch: 03d} train cost: {time.time() - t1: 0.2f}s, train_loss: {train_loss: 0.4f}, train_acc: {train_acc / BATCH_SIZE: 0.4f}, valid_loss: {valid_loss: 0.4f}, valid_acc: {valid_acc/BATCH_SIZE: 0.4f}")
        torch.save(classifier, f"Bert Model/epoch {epoch}.pt")
