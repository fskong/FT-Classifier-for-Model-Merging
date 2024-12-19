import torch
import torch.nn as nn
from transformers import BertModel

class BaseModel(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.n_class = n_class
        self.bert = BertModel.from_pretrained('../checkpoints/bert-base-uncased', local_files_only=True,  output_hidden_states=False,)
        self.classifier = nn.Sequential(nn.Linear(768, n_class))
        # self.classifier = nn.Linear(768, n_class)

    def forward(self, x):
        x = self.bert(x)
        x = x.last_hidden_state
        x = torch.mean(x, 1)
        logits = self.classifier(x)
        return logits

    def get_feature(self, x):
        x = self.bert(x)
        x = x.last_hidden_state
        x = torch.mean(x, 1)
        return x
