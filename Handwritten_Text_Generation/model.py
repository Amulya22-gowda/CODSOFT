import torch
import torch.nn as nn

# ----------------------------
# Model 1: For handwriting images
# ----------------------------
class HandwritingRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, num_layers=2):
        super(HandwritingRNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out

# ----------------------------
# Model 2: For text generation
# ----------------------------
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers=2):
        super(CharRNN, self).__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
