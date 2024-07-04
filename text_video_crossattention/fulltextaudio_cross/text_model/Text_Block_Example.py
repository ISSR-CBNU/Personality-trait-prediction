import torch
import torch.nn as nn
from transformers import AutoConfig, AlbertTokenizer, AutoModel


class TextBlock(nn.Module):
    """
    TextBlock : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, path, hidden_size=768, dropout=0.1):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.backbone = AutoModel.from_pretrained(path).to(self.device)
        self.linear = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)

    def forward(self, x):
        tokenized_x = self.tokenizer(x,
                                     return_tensors="pt",
                                     max_length=512,
                                     padding="max_length",
                                     truncation=True)['input_ids'].to(self.device)
        x = self.backbone(tokenized_x)
        x = x[1]  # get [CLS] token
        x = self.dropout(x)
        # x = self.linear(x)
        
        return x

############################################### Example ##########################################

# Model_Path = '/home/user/...'

# textblockforcrossattention = TextBlock(Model_Path)

# text = "I'm ChungBuk National University Student."

# textblockforcrossattention(text)