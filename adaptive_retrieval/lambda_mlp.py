import torch
import torch.nn as nn


class LambdaMLP(nn.Module):
    def __init__(self, hidden_units=128, nlayers=4, dropout=0.6, ctxt_dim=1024, activation='relu'):
        super().__init__()


        models = [nn.Linear(ctxt_dim, hidden_units), nn.Dropout(p=dropout)]
        if activation == 'relu':
            models.append(nn.ReLU())

        for _ in range(nlayers-1):
            models.extend([nn.Linear(hidden_units, hidden_units), nn.Dropout(p=dropout)])
            if activation == 'relu':
                models.append(nn.ReLU())

        models.append(nn.Linear(hidden_units, 2))

        models.append(nn.LogSoftmax(dim=-1))

        self.model = nn.Sequential(*models)


    def forward(self, features):
  
        return self.model(features)