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
    def __init__(self, hidden_units=128, nlayers=4, dropout=0.5, ctxt_dim=1024, activation='relu', use_conf_ent=False, use_freq_fert=False, use_faiss_centroids=False):
        super().__init__()

        self.use_conf_ent = use_conf_ent 
        self.use_freq_fert = use_freq_fert
        self.use_faiss_centroids = use_faiss_centroids

        if use_conf_ent and not use_freq_fert:    
            input_dim=int(ctxt_dim)*2
        elif use_conf_ent and use_freq_fert:    
            input_dim=2044
        elif use_conf_ent and use_faiss_centroids:
            input_dim=int(ctxt_dim)*2
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


        if use_conf_ent and not use_freq_fert and not use_faiss_centroids:
            input_layer = {}
            ndim = int(ctxt_dim / 2)
            for k in ['conf','ent']:
                input_layer[k] = LeakyReLUNet(1, ndim)

            self.input_layer = nn.ModuleDict(input_layer)
        elif use_conf_ent and use_freq_fert:
            input_layer = {}
            ndim = int(ctxt_dim / 10)
            for k in ['conf','ent','freq_1','freq_2','freq_3','freq_4','fert_1','fert_2','fert_3','fert_4']:
                input_layer[k] = LeakyReLUNet(1, ndim)

            self.input_layer = nn.ModuleDict(input_layer)
        elif use_conf_ent and use_faiss_centroids:
            input_layer = {}
            ndim = int(ctxt_dim / 4)
            for k in ['conf','ent','min_dist', 'min_top32_dist']:
                input_layer[k] = LeakyReLUNet(1, ndim)

            self.input_layer = nn.ModuleDict(input_layer)


    def forward(self, features, targets=None, conf=None, ent=None, freq_1=None, freq_2=None, freq_3=None, freq_4=None, fert_1=None, fert_2=None, fert_3=None, fert_4=None, min_dist=None, min_top32_dist=None):
        if self.use_conf_ent and not self.use_freq_fert  and not self.use_faiss_centroids:
            features_cat = [features]
            features_cat.append(self.input_layer['conf'](conf))
            features_cat.append(self.input_layer['ent'](ent))
            features_cat = torch.cat(features_cat,-1)

            scores = self.model(features_cat)
        elif self.use_conf_ent and self.use_freq_fert:
            features_cat = [features]
            features_cat.append(self.input_layer['conf'](conf))
            features_cat.append(self.input_layer['ent'](ent))
            features_cat.append(self.input_layer['freq_1'](freq_1))
            features_cat.append(self.input_layer['freq_2'](freq_2))
            features_cat.append(self.input_layer['freq_3'](freq_3))
            features_cat.append(self.input_layer['freq_4'](freq_4))
            features_cat.append(self.input_layer['fert_1'](fert_1))
            features_cat.append(self.input_layer['fert_2'](fert_2))
            features_cat.append(self.input_layer['fert_3'](fert_3))
            features_cat.append(self.input_layer['fert_4'](fert_4))
            features_cat = torch.cat(features_cat,-1)

            scores = self.model(features_cat)
        elif self.use_conf_ent and self.use_faiss_centroids:
            features_cat = [features]
            features_cat.append(self.input_layer['conf'](conf))
            features_cat.append(self.input_layer['ent'](ent))
            features_cat.append(self.input_layer['min_dist'](min_dist))
            features_cat.append(self.input_layer['min_top32_dist'](min_top32_dist))
            features_cat = torch.cat(features_cat,-1)            
  
            scores = self.model(features_cat)

        else:
            scores = self.model(features)

        return scores