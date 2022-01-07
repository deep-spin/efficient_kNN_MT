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
    def __init__(self, hidden_units=128, nlayers=4, dropout=0.5, ctxt_dim=1024, activation='relu', use_conf_ent=False, use_freq_fert=False, compute_loss=False, use_faiss_centroids=False, loss='cross_entropy', use_context=False, use_leaky_relu=False):
        super().__init__()

        self.use_conf_ent = use_conf_ent 
        self.use_freq_fert = use_freq_fert
        self.use_faiss_centroids = use_faiss_centroids
        self.use_context = use_context
        self.use_leaky_relu = use_leaky_relu
        self.compute_loss = compute_loss

        if use_conf_ent and not use_freq_fert and use_context:
            if use_leaky_relu:
                input_dim=int(ctxt_dim)*2
            else:
                input_dim=int(ctxt_dim)+2
        elif use_conf_ent and use_freq_fert and use_context and not use_faiss_centroids:    
            if use_leaky_relu:
                input_dim=2044
            else:
                input_dim=int(ctxt_dim)+10
        elif use_conf_ent and use_faiss_centroids and use_context and not use_freq_fert:
            if use_leaky_relu:
                input_dim=int(ctxt_dim)*2
            else:
                input_dim=int(ctxt_dim)+4
        elif use_conf_ent and use_faiss_centroids and use_context and use_freq_fert:
            if use_leaky_relu:
                input_dim=2044
            else:
                input_dim=int(ctxt_dim)+12
        elif use_conf_ent and use_freq_fert and not use_faiss_centroids:
            if use_leaky_relu:
                input_dim=1020
            else:
                input_dim=10
        elif use_conf_ent and use_faiss_centroids and not use_freq_fert:
            if use_leaky_relu:
                input_dim=int(ctxt_dim)
            else:
                input_dim=4
        elif use_conf_ent and use_faiss_centroids and use_freq_fert:
            if use_leaky_relu:
                input_dim=1020
            else:
                input_dim=12
        else:
            input_dim=ctxt_dim

        
        if nlayers>0:
            models = [nn.Linear(input_dim, hidden_units), nn.Dropout(p=dropout)]
            if activation == 'relu':
                models.append(nn.ReLU())
        
            for _ in range(nlayers-1):
                models.extend([nn.Linear(hidden_units, hidden_units), nn.Dropout(p=dropout)])
                if activation == 'relu':
                    models.append(nn.ReLU())

            models.append(nn.Linear(hidden_units, 1))

        else:
            hidden_units=input_dim
            models=[nn.Linear(hidden_units, 1)]

        models.append(nn.Sigmoid())

        self.model = nn.Sequential(*models)

        if use_leaky_relu:
            if use_conf_ent and not use_freq_fert and not use_faiss_centroids:
                input_layer = {}
                ndim = int(ctxt_dim / 2)
                for k in ['conf','ent']:
                    input_layer[k] = LeakyReLUNet(1, ndim)

                self.input_layer = nn.ModuleDict(input_layer)
            elif use_conf_ent and use_freq_fert and not use_faiss_centroids:
                input_layer = {}
                ndim = int(ctxt_dim / 10)
                for k in ['conf','ent','freq_1','freq_2','freq_3','freq_4','fert_1','fert_2','fert_3','fert_4']:
                    input_layer[k] = LeakyReLUNet(1, ndim)
                self.input_layer = nn.ModuleDict(input_layer)
            elif use_conf_ent and use_faiss_centroids and not use_freq_fert:
                input_layer = {}
                ndim = int(ctxt_dim / 4)
                for k in ['conf','ent','min_dist', 'min_top32_dist']:
                    input_layer[k] = LeakyReLUNet(1, ndim)

                self.input_layer = nn.ModuleDict(input_layer)

            elif use_conf_ent and use_faiss_centroids and use_freq_fert:
                input_layer = {}
                ndim = int(ctxt_dim / 12)
                for k in ['conf','ent','freq_1','freq_2','freq_3','freq_4','fert_1','fert_2','fert_3','fert_4','min_dist', 'min_top32_dist']:
                    input_layer[k] = LeakyReLUNet(1, ndim)
                
                self.input_layer = nn.ModuleDict(input_layer)

        self.loss_function=loss
        if self.compute_loss:
            if loss=='cross_entropy':
                self.loss_ = nn.BCELoss()
            elif loss=='mse':
                self.loss_ = nn.MSELoss()
                


    def forward(self, features=None, targets=None, conf=None, ent=None, freq_1=None, freq_2=None, freq_3=None, freq_4=None, fert_1=None, fert_2=None, fert_3=None, fert_4=None, min_dist=None, min_top32_dist=None):
        if self.use_conf_ent and not self.use_freq_fert  and not self.use_faiss_centroids and self.use_context:
            features_cat = [features]
            if self.use_leaky_relu:
                features_cat.append(self.input_layer['conf'](conf))
                features_cat.append(self.input_layer['ent'](ent))
            else:
                features_cat.append(conf)
                features_cat.append(ent)
            features_cat = torch.cat(features_cat,-1)

            scores = self.model(features_cat)
        elif self.use_conf_ent and self.use_freq_fert and self.use_context and not self.use_faiss_centroids:
            features_cat = [features]
            if self.use_leaky_relu:
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
            else:
                features_cat.append(conf)
                features_cat.append(ent)
                features_cat.append(freq_1)
                features_cat.append(freq_2)
                features_cat.append(freq_3)
                features_cat.append(freq_4)
                features_cat.append(fert_1)
                features_cat.append(fert_2)
                features_cat.append(fert_3)
                features_cat.append(fert_4)

            features_cat = torch.cat(features_cat,-1)

            scores = self.model(features_cat)
        elif self.use_conf_ent and self.use_faiss_centroids and not self.use_freq_fert and self.use_context:
            features_cat = [features]
            if self.use_leaky_relu:
                features_cat.append(self.input_layer['conf'](conf))
                features_cat.append(self.input_layer['ent'](ent))
                features_cat.append(self.input_layer['min_dist'](min_dist))
                features_cat.append(self.input_layer['min_top32_dist'](min_top32_dist))
            else:
                features_cat.append(conf)
                features_cat.append(ent)
                features_cat.append(min_dist)
                features_cat.append(min_top32_dist)
            features_cat = torch.cat(features_cat,-1)            

            scores = self.model(features_cat)

        elif self.use_conf_ent and self.use_faiss_centroids and  self.use_freq_fert and self.use_context:
            features_cat = [features]
            if self.use_leaky_relu:
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
                features_cat.append(self.input_layer['min_dist'](min_dist))
                features_cat.append(self.input_layer['min_top32_dist'](min_top32_dist))
            else:
                features_cat.append(conf)
                features_cat.append(ent)
                features_cat.append(freq_1)
                features_cat.append(freq_2)
                features_cat.append(freq_3)
                features_cat.append(freq_4)
                features_cat.append(fert_1)
                features_cat.append(fert_2)
                features_cat.append(fert_3)
                features_cat.append(fert_4)
                features_cat.append(min_dist)
                features_cat.append(min_top32_dist)
            features_cat = torch.cat(features_cat,-1)

            scores = self.model(features_cat)

        elif self.use_conf_ent and not self.use_freq_fert and not self.use_faiss_centroids:
            features_cat = []
            if self.use_leaky_relu:
                features_cat.append(self.input_layer['conf'](conf))
                features_cat.append(self.input_layer['ent'](ent))
            else:
                features_cat.append(conf)
                features_cat.append(ent)
            features_cat = torch.cat(features_cat,-1)

            scores = self.model(features_cat)
        elif self.use_conf_ent and self.use_freq_fert and not self.use_faiss_centroids:
            features_cat = []
            if self.use_leaky_relu:
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
            else:
                features_cat.append(conf)
                features_cat.append(ent)
                features_cat.append(freq_1)
                features_cat.append(freq_2)
                features_cat.append(freq_3)
                features_cat.append(freq_4)
                features_cat.append(fert_1)
                features_cat.append(fert_2)
                features_cat.append(fert_3)
                features_cat.append(fert_4)
            features_cat = torch.cat(features_cat,-1)

            scores = self.model(features_cat)

        elif self.use_conf_ent and self.use_faiss_centroids and not self.use_freq_fert:
            features_cat = []
            if self.use_leaky_relu:
                features_cat.append(self.input_layer['conf'](conf))
                features_cat.append(self.input_layer['ent'](ent))
                features_cat.append(self.input_layer['min_dist'](min_dist))
                features_cat.append(self.input_layer['min_top32_dist'](min_top32_dist))
            else:
                features_cat.append(conf)
                features_cat.append(ent)
                features_cat.append(min_dist)
                features_cat.append(min_top32_dist)
            features_cat = torch.cat(features_cat,-1)            

            scores = self.model(features_cat)

        elif self.use_conf_ent and self.use_faiss_centroids and  self.use_freq_fert:
            features_cat = []
            if self.use_leaky_relu:
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
                features_cat.append(self.input_layer['min_dist'](min_dist))
                features_cat.append(self.input_layer['min_top32_dist'](min_top32_dist))
            else:
                features_cat.append(conf)
                features_cat.append(ent)
                features_cat.append(freq_1)
                features_cat.append(freq_2)
                features_cat.append(freq_3)
                features_cat.append(freq_4)
                features_cat.append(fert_1)
                features_cat.append(fert_2)
                features_cat.append(fert_3)
                features_cat.append(fert_4)
                features_cat.append(min_dist)
                features_cat.append(min_top32_dist)
            features_cat = torch.cat(features_cat,-1)

            scores = self.model(features_cat)

        else:
            scores = self.model(features)

        if self.compute_loss:
            if self.loss_function=='weighted_cross_entropy':
                weights = torch.ones(targets.size(0)).cuda()
                for i in range(len(targets)):
                    if targets[i]==0:
                        weights[i]=.25
                loss_ = loss_ = nn.BCELoss(weight=weights)
                loss = loss_(scores.squeeze(-1), targets.squeeze(-1))
            else:
                loss = self.loss_(scores.squeeze(-1), targets.squeeze(-1))
            return scores, loss
        else:
            return scores