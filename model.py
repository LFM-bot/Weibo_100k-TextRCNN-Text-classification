import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1234
np.random.seed(SEED)
torch.random.manual_seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class TextRCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.voc_size,
                                      config.embed_size,
                                      padding_idx=config.voc_size - 1)
        self.lstm = nn.LSTM(input_size=config.embed_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            dropout=config.dropout,
                            bidirectional=True,
                            batch_first=True)
        self.max_pool = nn.MaxPool1d(config.max_len)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed_size, config.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_seq):
        """
        :param input_seq: Torch.LongTensor, [batch_size, max_len]
        """
        word_seq_emb = self.embedding(input_seq)  # [batch_size, max_len, dims]
        out, _ = self.lstm(word_seq_emb)  # [batch_size, max_len, dims]
        out = torch.cat([out, word_seq_emb], dim=2)  # [batch_size, max_len, 2*dims]
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.max_pool(out).squeeze()  # [batch_size, pool_len, 2*dims]
        out = self.fc(out)
        logits = self.softmax(out)

        return logits
