import argparse
import logging
import os
import random
import csv

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models.modeling_early_exit import VisionTransformer, CONFIGS
from utils.data_utils import get_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)

    # model.load_from(np.load(args.pretrained_dir))
    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model, map_location='cuda:0')['model']
        model.load_state_dict(pretrained_model)
    model.to(args.device)
    num_params = count_parameters(model)
    return args, model


def valid(args, model, test_loader):
    # Validation!
    eval_losses = AverageMeter()
    exit_layers = []

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            y = y.type(torch.LongTensor).cuda()
            logits, exit_layer = model(x)
            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        exit_layers.append(exit_layer)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
        epoch_iterator.set_postfix(exit_layer=exit_layer)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy = torch.tensor(accuracy).to(args.device)
    val_accuracy = accuracy

    val_accuracy = val_accuracy.detach().cpu().numpy()

    return val_accuracy, sum(exit_layers)/len(exit_layers)

def th_search(th, old_exit_layer, new_exit_layer):
    accuracy_percent_diff = (old_exit_layer - new_exit_layer)/12
    if accuracy_percent_diff <= 0.005:
        th = th * 2
    elif accuracy_percent_diff > 0.005:
        th = th * 0.75

    return th

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=False, default='vit_distil_v3_2',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--pretrained_model", type=str,
                        default='D:/Research Projects/Early_Exit/TransFG-baseline/output/vit_distil_v3_2_checkpoint.bin',
                        help="load pretrained model")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017"],
                        default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='D:/Datasets/CUB_200_2011')
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    args = parser.parse_args()
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_loader, test_loader = get_loader(args)
    model_names = ['vit_distil_v3_2', 'vit_distil_v3']
    model_names = ['vit_early_exit_final_10000_6']
    for model_name in model_names:
        args.name = model_name
        args.pretrained_model = os.path.join('D:/Research Projects/Early_Exit/TransFG-baseline/new_output/',
                                             args.name) + '_checkpoint.bin'
        args, model = setup(args)
        accuracy = []
        exit_layers = []
        early_exit_threshold = []
        early_exit_th = 1e-5
        old_exit_layer = 12
        #store_point_exit_layer = 12
        count = 0
        while True:
            if count == 100 or (old_exit_layer - 6) / 12 <= 0.005:
                break

            with torch.no_grad():
                val_accuracy, exit_layer = valid(args, model, test_loader)
            new_exit_layer = exit_layer
            accuracy_percent_diff = (old_exit_layer - new_exit_layer) / 12

            if count == 0 or 0.0025 <= accuracy_percent_diff <= 0.0075:
                print('early_exit_th = ' + str(early_exit_th))
                print('val_accuracy = ' + str(val_accuracy))
                print('exit_layer = ' + str(exit_layer))
                print('saved_data_count = ' + str(count))
                accuracy.append(val_accuracy)
                exit_layers.append(exit_layer)
                early_exit_threshold.append(early_exit_th)
                store_point_exit_layer = new_exit_layer
                count += 1

            early_exit_th = th_search(early_exit_th, old_exit_layer, new_exit_layer)
            print('old_exit_layer = ' + str(old_exit_layer))
            print('new_exit_layer = ' + str(new_exit_layer))
            print('next_threshold = ' + str(early_exit_th))
            old_exit_layer = store_point_exit_layer
            model.set_early_exit_th(early_exit_th)

        with open(os.path.join("finetune_logs", args.name)+'.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(accuracy)
            writer.writerow(exit_layers)
            writer.writerow(early_exit_threshold)


if __name__ == "__main__":
    main()
