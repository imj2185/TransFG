import argparse
import logging
import os
import random
import csv

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm import tqdm
import numpy as np

from models.modeling_early_exit import VisionTransformer, CONFIGS
from apex.parallel import DistributedDataParallel as DDP
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

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def setup(args, early_exit_th):
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

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,
                              early_exit_th=early_exit_th)

    # model.load_from(np.load(args.pretrained_dir))
    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)
    model.to(args.device)
    num_params = count_parameters(model)
    return args, model


def valid(args, model, test_loader):
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
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
    dist.barrier()
    val_accuracy = reduce_mean(accuracy, args.nprocs)

    val_accuracy = val_accuracy.detach().cpu().numpy()

    return val_accuracy, sum(exit_layers)/len(exit_layers)


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
    parser.add_argument("--eval_batch_size", default=8, type=int,
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

    accuracy = []
    exit_layers = []
    early_exit_threshold = []
    for i in range(101):
        early_exit_th = i/100000
        args, model = setup(args, early_exit_th)
        with torch.no_grad():
            val_accuracy, exit_layer = valid(args, model, test_loader)
            print('early_exit_th = ' + str(early_exit_th))
            print('val_accuracy = ' + str(val_accuracy))
            print('exit_layer = ' + str(exit_layer))
            accuracy.append(val_accuracy)
            exit_layers.append(exit_layer)
            early_exit_threshold.append(early_exit_th)

    with open(os.path.join("finetune_logs", args.name)+'.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(accuracy)
        writer.writerow(exit_layers)
        writer.writerow(early_exit_threshold)


if __name__ == "__main__":
    main()
