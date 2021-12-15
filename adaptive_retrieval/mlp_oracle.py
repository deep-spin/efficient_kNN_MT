import torch
import torch.nn as nn


class LeakyReLUNet(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.LeakyReLU(),
            nn.Linear(out_feat, out_feat),
        )

    def forward(self, features):
        return self.model(features)


class MLPOracle(nn.Module):
    def __init__(self, hidden_units=128, nlayers=4, dropout=0.5, ctxt_dim=1024, activation='relu', use_conf_ent=False, compute_loss=False):
        super().__init__()

        self.use_conf_ent = use_conf_ent 
        self.compute_loss = compute_loss

        if use_conf_ent:    
            input_dim=int(ctxt_dim*2)
        else:
            input_dim=ctxt_dim

        models = [nn.Linear(input_dim, hidden_units), nn.Dropout(p=dropout)]
        if activation == 'relu':
            models.append(nn.ReLU())

        for _ in range(nlayers-1):
            models.extend([nn.Linear(hidden_units, hidden_units), nn.Dropout(p=dropout)])
            if activation == 'relu':
                models.append(nn.ReLU())

        models.append(nn.Linear(hidden_units, 1))

        models.append(nn.Sigmoid(dim=-1))

        self.model = nn.Sequential(*models)

        if use_conf_ent:
            input_layer = {}
            ndim = int(ctxt_dim / 2)
            for k in ['conf','ent']:
                input_layer[k] = LeakyReLUNet(1, ndim)

            self.input_layer = nn.ModuleDict(input_layer)

        if self.compute_loss:
            self.loss_ = nn.BCELoss()


    def forward(self, features, conf=None, ent=None):
        if self.use_conf_ent:
            features_cat = [features]
            features_cat.append(self.input_layer['conf'](conf.unsqueeze(-1)))
            features_cat.append(self.input_layer['ent'](ent.unsqueeze(-1)))
            features_cat = torch.cat(features_cat,-1)

            scores = self.model(features_cat)
        else:
            scores = self.model(features)

        if self.compute_loss:
            loss = self.loss_(scores, targets)
            return scores, loss
        else:
            return scores