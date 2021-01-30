import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LRModel(torch.nn.Module):

    def __init__(self, conf):
        super(LRModel, self).__init__()
        self.vocab_size = conf.get('vocab_size', None)
        self.team_size = conf['team_size']
        self.inp_dim = conf['num_features']
        self.out_dim = conf['num_classes']
        self.device = conf.get('device', 'cpu')
        self.fc = nn.Linear(2 * self.team_size * self.inp_dim, self.out_dim)

    def forward(self, x):
        x1, x2 = x  # (N, T, D)
        # x1, x1_mask, x2, x2_mask = inputs  # (T, N, D), (T, N)
        batch_size, team_size = x1.size(0), x1.size(1)
        x = torch.cat([x1, x2], dim=1)  # (N, 2T, H)
        return torch.sigmoid(self.fc(x.view(batch_size, -1)))  # (N, 1)

    def predict(self, x):
        x1, x2 = x
        x1 = torch.FloatTensor(x1).to(self.device)  # (N, T, D)
        x2 = torch.FloatTensor(x2).to(self.device)  # (N, T, D)
        o = self((x1, x2))  # (N, 1), o.squeeze(1) -> # (N,)
        o = o.detach().cpu().numpy()  # (N, 1)
        return o  # (N, 1)
