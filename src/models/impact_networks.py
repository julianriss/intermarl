import torch.nn as nn
import torch.nn.functional as F


class SimpleFF(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, hidden_layer_size: int = 32
    ):
        super(SimpleFF, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
