import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_shape, hidden_size, output_size, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        N, T, V, C = input_shape
        input_size = V * C
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        N, T, V, C = inputs.shape
        x = inputs.view(N, T, V * C)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  
        out = out[:, -1, :]
        out = self.classifier(out)
        return out