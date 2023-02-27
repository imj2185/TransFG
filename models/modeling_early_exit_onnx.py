import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from torch.nn import CrossEntropyLoss
import models.configs as configs

from .modeling import Part_Attention, Block, LayerNorm, LabelSmoothing, Embeddings, np2th

logger = logging.getLogger(__name__)

def entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)    # sum of exp(x_i)
    B = torch.sum(x*exp_x, dim=1)  # sum of x_i * exp(x_i)
    return torch.log(A) - B/A


def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss


class ViT_early_exit(nn.Module):
    r"""A module to provide a shortcut
    from
    the output of one non-final ViTLayer in ViTEncoder
    to
    loss computation
    """
    def __init__(self, config, num_classes, zero_heads=True):
        super(ViT_early_exit, self).__init__()
        #self.dropout = nn.Dropout(config.transformer['dropout_rate'])
        self.head = nn.Linear(config.hidden_size, num_classes)
        self.zero_head = zero_heads
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)

    def forward(self, encoder_output):
        logits = self.head(encoder_output[:, 0])
        return logits


class Encoder(nn.Module):
    def __init__(self, config, num_classes, vis, early_exit_th):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.early_exit_layer = nn.ModuleList()
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))
        for _ in range(config.transformer["num_layers"]):
            early_exit_layer = ViT_early_exit(config, num_classes)
            self.early_exit_layer.append(copy.deepcopy(early_exit_layer))

        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

        self.num_layers = config.transformer["num_layers"]
        self.hidden_size = config.hidden_size
        self.early_exit_entropy = [-1 for _ in range(config.hidden_size)]
        self.num_classes = num_classes
        self.use_lte = True
        self.early_exit_th = early_exit_th
        self.init_lte()

    def init_lte(self):
        self.lte_th = [self.early_exit_th] * self.num_layers
        #self.lte_classifier = nn.Linear(self.hidden_size, 1)
        self.lte_classifier = nn.Linear(self.num_classes, 1)
        self.lte_activation = nn.Sigmoid()

    def forward(self, hidden_states, exit_layer):
        #new_hidden_states = []
        #lte_outputs = []

        #choose by single sample
        for j, layer in enumerate(self.layer):
            if j < exit_layer:
                hidden_states, weights = layer(hidden_states)
                early_exit = self.early_exit_layer[j](self.encoder_norm(hidden_states))
        
        return early_exit


class Transformer(nn.Module):
    def __init__(self, config, img_size, num_classes, vis, early_exit_th):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, num_classes, vis, early_exit_th)

    def forward(self, input_ids, early_exit):
        embedding_output = self.embeddings(input_ids)
        all_early_exit = self.encoder(embedding_output, early_exit)
        return all_early_exit


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, early_exit_th=0.0, train_strategy='continuous-lte'):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, num_classes, vis, early_exit_th)
        #self.head = nn.Linear(config.hidden_size, num_classes)

        self.num_layers = config.transformer["num_layers"]
        self.train_strategy = train_strategy

    def set_early_exit_th(self, early_exit_th):
        self.transformer.encoder.lte_th = [early_exit_th] * self.num_layers

    def forward(self, x, early_exit=11):
        logits = self.transformer(x, early_exit)
        return logits

    def load_from(self, weights):
        with torch.no_grad():
            # if self.zero_head:
            #     nn.init.zeros_(self.head.weight)
            #     nn.init.zeros_(self.head.bias)
            # else:
            #     self.head.weight.copy_(np2th(weights["head/kernel"]).t())
            #     self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                if bname != 'early_exit_layer':
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}