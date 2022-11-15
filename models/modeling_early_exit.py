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

    def forward(self, hidden_states):
        attn_weights = []
        all_early_exit = []
        all_lte_output = []
        last_classifier = []
        exit_layer = []
        #new_hidden_states = []
        #lte_outputs = []

        #choose by single sample
        for i, single_hidden_state in enumerate(hidden_states): # loop every sample in a batch
            single_hidden_state = single_hidden_state.unsqueeze(0) # add the lost dim
            attn_weights_layers = []
            early_exit_layers = []
            lte_output_layers = []
            for j, layer in enumerate(self.layer):
                single_hidden_state, weights = layer(single_hidden_state)
                if self.vis:
                    attn_weights_layers.append(weights.squeeze())
                early_exit = self.early_exit_layer[j](self.encoder_norm(single_hidden_state))
                early_exit_entropy = entropy(early_exit)
                early_exit_layers.append(early_exit.squeeze())
                # lte block
                if self.use_lte:
                    lte_input = early_exit
                    lte_output = self.lte_activation(self.lte_classifier(lte_input)).squeeze()
                    # lte_outputs.append(lte_output)

                if not self.training:
                    if j >= 6 - 1:
                        if (
                                (j + 1 < self.num_layers)
                                and (
                                (self.use_lte and lte_output < self.lte_th[j])
                                or (not self.use_lte and early_exit_entropy < self.early_exit_entropy[j])
                        )
                        ): # early exit
                            break
                else:
                    #early_exit_layers.append(early_exit.squeeze())
                    lte_output_layers.append(lte_output)

            if self.vis:
                attn_weights.append(torch.stack(attn_weights_layers))
            if self.training:
                #all_early_exit.append(torch.stack(early_exit_layers))
                all_lte_output.append(torch.stack(lte_output_layers))
                all_early_exit.append(torch.stack(early_exit_layers))
            else:
                exit_layer.append(j + 1)
                last_classifier.append(early_exit.squeeze()) # torch.Size([16, 200])
            #new_hidden_states.append(single_hidden_state.squeeze())

        if self.vis:
            attn_weights = torch.stack(attn_weights).permute(1, 0, 2, 3, 4) # torch.Size([11, 16, 12, 785, 785])
        if self.training:
            #all_early_exit = torch.stack(all_early_exit).permute(1, 0, 2) # torch.Size([11, 16, 200])
            all_lte_output = torch.stack(all_lte_output).permute(1, 0) # torch.Size([11, 16])
            all_early_exit = torch.stack(all_early_exit).permute(1, 0, 2)  # torch.Size([11, 16, 200])
            return attn_weights, all_early_exit, all_lte_output
        else:
            return attn_weights, torch.stack(last_classifier), exit_layer
        #hidden_states = torch.stack(new_hidden_states) # torch.Size([16, 785, 768]), grad_fn=<StackBackward0>

        # # choose by batch
        # for i, layer in enumerate(self.layer):
        #     hidden_states, weights = layer(hidden_states)
        #     if self.vis:
        #         attn_weights.append(weights)
        #
        #     early_exit = self.early_exit_layer[i](self.encoder_norm(hidden_states))
        #     early_exit_entropy = entropy(early_exit)
        #     # lte block
        #     if self.use_lte:
        #         lte_input = early_exit
        #         lte_output = self.lte_activation(self.lte_classifier(lte_input)).squeeze()
        #         #lte_outputs.append(lte_output)
        #
        #     if not self.training:
        #         if (
        #                 (i + 1 < self.num_layers)
        #                 and (
        #                 (self.use_lte and lte_output.max() < self.lte_th[i])
        #                 or (not self.use_lte and early_exit_entropy.max() < self.early_exit_entropy[i])
        #         )
        #         ):
        #             break
        #     else:
        #         all_early_exit.append(early_exit)
        #         all_lte_output.append(lte_output)
        #
        # # print(torch.stack(all_early_exit).shape) # torch.Size([11, 16, 200])
        # # print(torch.stack(all_lte_output).shape) # torch.Size([11, 16])
        # # print(hidden_states.shape) # torch.Size([16, 785, 768]), device='cuda:0', grad_fn=<AddBackward0>
        # # print(torch.stack(attn_weights).shape) # torch.Size([11, 16, 12, 785, 785])

        #encoded = self.encoder_norm(hidden_states)
        return attn_weights, all_early_exit, all_lte_output


class Transformer(nn.Module):
    def __init__(self, config, img_size, num_classes, vis, early_exit_th):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, num_classes, vis, early_exit_th)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        attn_weights, all_early_exit, all_lte_output = self.encoder(embedding_output)
        return attn_weights, all_early_exit, all_lte_output


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

    def forward(self, x, labels=None, exit_layer=12, classifier_backward=False):
        if labels is not None:
            all_early_exit_loss = []
            attn_weights, all_early_exit, all_lte_output = self.transformer(x)
            logits = all_early_exit[exit_layer - 1]

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

            for i, early_exit in enumerate(all_early_exit):
                if i <= exit_layer - 1:
                    early_exit_logits = early_exit
                    early_exit_loss = loss_fct(early_exit_logits.view(-1, self.num_classes), labels.view(-1))
                    all_early_exit_loss.append(early_exit_loss)

            # alternating not working
            if self.train_strategy == 'continuous-lte':
                lte_loss_fct = nn.MSELoss()
                layer_acc = []
                exit_pred = []
                for i in range(exit_layer):
                    exit_pred.append(all_lte_output[i])
                    # label
                    layer_early_exit_output = all_early_exit[i]
                    if self.num_classes == 1:
                        correctness_loss = torch.tanh(layer_early_exit_output.squeeze() - labels).abs()
                    else:
                        lte_gold = torch.eq(
                            torch.argmax(layer_early_exit_output, dim=1),
                            labels
                        )  # 0 for wrong/continue, 1 for right/exit
                        correctness_loss = ~(lte_gold).half()  # 1 for continue, match exit_pred
                    layer_acc.append(correctness_loss)

                exit_pred = torch.stack(exit_pred)
                exit_label = torch.stack(layer_acc).detach()
                if classifier_backward:
                    #loss = loss + sum(all_early_exit_loss[:-1]) + lte_loss_fct(exit_pred, exit_label)  # classifier + lte
                    #loss = loss + sum(all_early_exit_loss[:-1]) + lte_loss_fct(exit_pred, exit_label)
                    loss = loss + lte_loss_fct(exit_pred[exit_layer - 1], exit_label[exit_layer - 1])
                else:
                    loss = loss

            return loss, logits
        else:
            attn_weights, logits, exit_layer = self.transformer(x)
            return logits, sum(exit_layer)/len(exit_layer)

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