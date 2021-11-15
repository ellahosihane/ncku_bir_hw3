import torch.nn as nn
import torch.nn.functional as F
import torch

class CBOW_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, 1)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        out = self.embeddings(x)
        out = out.mean(axis=1)
        out = self.linear(out)
        return out

class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, 1)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        out = self.embeddings(x)
        out = self.linear(out)
        return out