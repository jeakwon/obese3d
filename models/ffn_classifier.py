import torch
import torch.nn as nn

class FFNClassifier(nn.Module):
    def __init__(self, input_shape, hidden_size, output_size, num_layers=1):
        super(FFNClassifier, self).__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        N, T, V, C = input_shape
        input_size = T * V * C
        self.layers = nn.Sequential()
        for _ in range(num_layers):
            self.layers.append( nn.Linear(input_size, hidden_size) )
            self.layers.append( nn.ReLU() )
            input_size = hidden_size
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        N, T, V, C = inputs.shape
        x = inputs.view(N, T * V * C)
        x = self.layers(x)
        out = self.classifier(x)
        return out
