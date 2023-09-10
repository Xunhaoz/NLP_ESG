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
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

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
    with open("../static/ML-ESG-2_English_Train.json", "r") as f:
        all_dataset = json.load(f)

    all_dataset_df = pd.DataFrame(all_dataset)[['news_content', 'impact_type']]
    all_dataset_df['impact_type'] = all_dataset_df['impact_type'].apply(lambda x: 1 if x == "Opportunity" else 0)

    return all_dataset_df


if __name__ == "__main__":

    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    model = torch.load("../epoch 12.pt").to('cpu')

    output_list = []
    all_dataset_df = get_data()
    all_dataset_numpy = all_dataset_df['news_content'].to_numpy()

    for data in tqdm(all_dataset_numpy, total=len(all_dataset_df)):
        input_ids = tokenizer(data, return_tensors='pt', padding=True, truncation=True)['input_ids']
        attention_mask = tokenizer(data, return_tensors='pt', padding=True, truncation=True)['attention_mask']
        output = model(input_ids, attention_mask)
        _, output = torch.max(output, 1)
        output_list.append(output.item())

    all_dataset_df['predict'] = pd.Series(output_list)
    all_dataset_df.to_csv("check.csv", index=False)
