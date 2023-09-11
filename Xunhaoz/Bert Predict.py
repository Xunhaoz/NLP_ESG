import json
import warnings
import pandas as pd
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

warnings.filterwarnings("ignore")
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
EPOCH = 100
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


def get_data():
    train_df = pd.read_csv("static/Train.csv")[["news_content", "impact_type"]]
    dev_df = pd.read_csv("static/Dev.csv")[["news_content", "impact_type"]]

    all_df = pd.concat([train_df, dev_df]).reset_index(drop=True)
    return all_df


if __name__ == "__main__":

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = torch.load("Bert Model/epoch 6.pt").to('cpu')

    output_list = []
    all_df = get_data()
    all_df_numpy = all_df['news_content'].to_numpy()

    for data in tqdm(all_df_numpy, total=len(all_df_numpy)):
        tokenizer_res = tokenizer(data, return_tensors='pt', padding=True, truncation=True)
        input_ids = tokenizer_res['input_ids']
        attention_mask = tokenizer_res['attention_mask']

        output = model(input_ids, attention_mask)
        output = torch.argmax(output, dim=1)

        output_list.append(output.item())

    all_df['predict'] = pd.Series(output_list)
    all_df.to_csv("check.csv", index=False)
