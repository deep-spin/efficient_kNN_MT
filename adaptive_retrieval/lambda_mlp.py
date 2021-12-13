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


class LambdaMLP(nn.Module):
    def __init__(self, hidden_units=128, nlayers=4, dropout=0.6, ctxt_dim=1024, activation='relu', use_conf_ent=False):
        super().__init__()

        self.use_conf_ent = use_conf_ent 

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

        models.append(nn.Linear(hidden_units, 2))

        models.append(nn.LogSoftmax(dim=-1))

        self.model = nn.Sequential(*models)


        if use_conf_ent:
            input_layer = {}
            ndim = int(ctxt_dim / 2)
            for k in ['conf','ent']:
                input_layer[k] = LeakyReLUNet(1, ndim)

            self.input_layer = nn.ModuleDict(input_layer)


    def forward(self, features, conf=None, ent=None):

        if self.use_conf_ent:
            print(conf.shape)
            x=self.input_layer['conf'](conf)
            print(x.shape)
            print(features.shape)
            features_cat = [features]
            features_cat.append(self.input_layer['conf'](conf))
            features_cat.append(self.input_layer['ent'](ent))
            features_cat = torch.cat(features_cat,-1)

            return self.model(features_cat)
  
        return self.model(features)