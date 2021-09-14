import torch
import torch.nn as nn

class LeakyReLUNet(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.LeakyReLU(),
            nn.Linear(out_feat, out_feat),)

    def forward(self, features):
        return self.model(features)


class LambdaMLP(nn.Module):
    def __init__(self, feature_set=None, hidden_units=32, nlayers=3, dropout=0,ctxt_dim=1024, non_ctxt_dim=512, activation='relu'):
        super().__init__()


        non_ctxt_dim=len(feature_set)-1*non_ctxt_dim

        models = [nn.Linear(ctxt_dim + non_ctxt_dim, hidden_units), nn.Dropout(p=dropout)]
        if activation == 'relu':
            models.append(nn.ReLU())
        elif activation == 'linear':
            pass
        else:
            raise ValueError(f'activation {activation} not supported')

        for _ in range(nlayers-1):
            models.extend([nn.Linear(hidden_units, hidden_units), nn.Dropout(p=dropout)])
            if activation == 'relu':
                models.append(nn.ReLU())
            elif activation == 'linear':
                pass
            else:
                raise ValueError(f'activation {activation} not supported')

        models.append(nn.Linear(hidden_units, 2))


        self.model = nn.Sequential(*models)

        input_layer = {}
        for k in feature_set:
            if k != 'ctxt':
                input_layer[k] = LeakyReLUNet(1, non_ctxt_dim)

        self.input_layer = nn.ModuleDict(input_layer)

        self.feature_set = feature_set

    def forward(self, features):

        features_cat = [features['ctxt']] if 'ctxt' in self.feature_set else []

        for k in self.feature_set:
            if k != 'ctxt':
                print(k)
                print(features[k])
                print(features[k].shape)
                features_cat.append(self.input_layer[k](features[k]))

        return torch.softmax(self.model(torch.cat(features_cat, -1)), dim=-1)

