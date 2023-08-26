import time

import torch
from transformers import AutoModel, AutoModelForMaskedLM
import torch.nn as nn
from transformer import Transformer, get_sinusoid_encoding_table, trunc_normal_
import torch.nn.functional as F


class MFB(nn.Module):
    def __init__(
            self,
            input_dims,
            output_dim,
            mm_dim=128,
            factor=2,
            activ_input="relu",
            activ_output="relu",
            normalize=False,
            dropout_input=0.0,
            dropout_pre_norm=0.0,
            dropout_output=0.0,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.mm_dim = mm_dim
        self.factor = factor
        self.output_dim = output_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_norm = dropout_pre_norm
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim * factor)
        self.linear1 = nn.Linear(input_dims[1], mm_dim * factor)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z = x0 * x1

        if self.dropout_pre_norm > 0:
            z = F.dropout(z, p=self.dropout_pre_norm, training=self.training)

        z = z.view(z.size(0), self.mm_dim, self.factor)
        z = z.sum(2)

        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)

        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class Model(nn.Module):
    def __init__(self, embed_dim=128):
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        checkpoint = 'hfl/chinese-bert-wwm'
        self.bert = AutoModel.from_pretrained(checkpoint)
        self.classification_head = nn.Linear(16, 3)

        self.cls_token_new = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.cls_token_hot = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.project = nn.Sequential(nn.Linear(768, self.embed_dim), nn.ReLU())

        self.transformer = Transformer(embed_dim=self.embed_dim)

        self.mfb = MFB([self.embed_dim, self.embed_dim], 16, factor=16, dropout_input=0.1, dropout_output=0.1)

        trunc_normal_(self.cls_token_new, std=.02)
        trunc_normal_(self.cls_token_hot, std=.02)

    def forward(self, input):
        probs = []
        for item in input:
            with torch.no_grad():
                start_time=time.time()
                new_result = self.bert(**item['content_new_token']).last_hidden_state
                end_time=time.time()
                print("feature encoder time {}".format(end_time-start_time))
                new_feature = new_result[:, 0, :]
                new_feature = self.project(new_feature)

                hot_result = self.bert(**item['content_hot_token']).last_hidden_state
                hot_feature = hot_result[:, 0, :]
                hot_feature = self.project(hot_feature)
            self.pos_embed_new = get_sinusoid_encoding_table(len(new_feature) + 1, self.embed_dim)
            self.pos_embed_hot = get_sinusoid_encoding_table(len(hot_feature) + 1, self.embed_dim)
            trunc_normal_(self.pos_embed_new, std=.02)
            trunc_normal_(self.pos_embed_hot, std=.02)

            cls_tokens_new = self.cls_token_new.expand(1, -1,
                                                       -1)  # stole cls_tokens impl from Phil Wang, thanks
            new_feature = torch.cat((cls_tokens_new, new_feature.reshape(1, -1, self.embed_dim)), dim=1)
            # pos_1 = self.pos_embed.type_as(x).to(x.device).clone().detach()
            # pos_2 = get_sinusoid_encoding_table(T + 1, self.embed_dim).type_as(x).to(x.device).clone().detach()

            # new_feature = new_feature + self.pos_embed_new.type_as(new_feature).to(new_feature.device).clone().detach()
            start_time = time.time()
            new_feature = self.transformer(new_feature)
            end_time = time.time()
            print("transformer encoder time {}".format(end_time - start_time))
            cls_tokens_hot = self.cls_token_hot.expand(1, -1,
                                                       -1)  # stole cls_tokens impl from Phil Wang, thanks
            hot_feature = torch.cat((cls_tokens_hot, hot_feature.reshape(1, -1, self.embed_dim)), dim=1)
            # pos_1 = self.pos_embed.type_as(x).to(x.device).clone().detach()
            # pos_2 = get_sinusoid_encoding_table(T + 1, self.embed_dim).type_as(x).to(x.device).clone().detach()

            # hot_feature = hot_feature + self.pos_embed_hot.type_as(hot_feature).to(hot_feature.device).clone().detach()

            hot_feature = self.transformer(hot_feature)
            start_time = time.time()
            gate = self.mfb(torch.cat(
                (new_feature.reshape(-1, self.embed_dim)[0, :].view(1, 1, -1),
                 hot_feature.reshape(-1, self.embed_dim)[0, :].view(1, 1, -1)), dim=0))
            end_time = time.time()
            print("FHBP encoder time {}".format(end_time - start_time))
            gate = gate.view(-1)
            # gate = new_feature.reshape(-1, self.embed_dim)[0, :] + hot_feature.reshape(-1, self.embed_dim)[0, :]
            start_time = time.time()
            prob = torch.softmax(self.classification_head(gate), dim=0)
            end_time=time.time()
            print("classification head time {}".format(end_time - start_time))
            # prob = self.classification_head(gate)
            probs.append(prob)
        return torch.stack(probs)
