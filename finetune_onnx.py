import argparse
import logging
import os
import random
import csv

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import timeit

from models.modeling_early_exit import VisionTransformer, CONFIGS
#from models.modeling_early_exit_onnx import VisionTransformer, CONFIGS
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
    elif args.dataset == 'air':
        num_classes = 100

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

def latency(args, model, test_loader):
    # Validation!
    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    start_time = timeit.default_timer()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)
    evalTime = timeit.default_timer() - start_time
        
    return evalTime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=False, default='vit_distil_v3_2',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--pretrained_model", type=str,
                        default='./vit_car_80000_checkpoint.bin',
                        help="load pretrained model")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017", "air"],
                        default="car",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='/home/bassett/Documents/archive/datasets/fine_grained_classification')
    parser.add_argument("--eval_batch_size", default=16, type=int,
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
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Total batch size for training.")

    parser.add_argument("--do_lat_mem_measure", action="store_true", 
                        help="do evo search")
    parser.add_argument("--do_onnx", action="store_true", 
                        help="do onnx conversion")
    parser.add_argument("--do_onnx_runtime", action="store_true", 
                        help="do onnx conversion")
    parser.add_argument("--threshold_value", default=0.0, type=float,
                        help="Total batch size for training.")

    args = parser.parse_args()
    import torch
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_loader, test_loader = get_loader(args)
    # early_exit_th = 0.000124979
    args, model = setup(args)
    if args.do_lat_mem_measure:
        import torchprofile
        model.set_early_exit_th(args.threshold_value)

        with torch.no_grad():
            val_accuracy, exit_layer = valid(args, model, test_loader)

        print(val_accuracy, exit_layer)

        evalTime = latency(args, model, test_loader)
        print('Evaluation done in total {:.3f} secs ({:.3f} sec per example)'.format(evalTime, evalTime / len(test_loader)))

        # size = (16, 3, args.img_size, args.img_size)
        # dummy_inputs = (
        #     torch.ones(size, dtype=torch.float).to(args.device)
        # )

        # model.eval()
        # #model.set_early_exit_th(early_exit_th)
        # macs = torchprofile.profile_macs(model, args=dummy_inputs)
        # print("MAC: ", macs)


    if args.do_onnx:
        model.eval()
        import torch.onnx
        size = (8, 3, args.img_size, args.img_size)
        dummy_input_1 = torch.randn(size, dtype=torch.float).to(args.device)
        # threshold = [0.002]
        # dummy_input_2 = torch.Tensor(threshold).to(args.device)

        # val_accuracy, exit_layer = valid(args, model, test_loader, dummy_input_2)
        # print(val_accuracy, exit_layer)

        torch.onnx.export(model,
            args=dummy_input_1,
            f="vitex_pytorch.onnx",
            export_params=True,
            input_names=["x"],
            output_names=["output"],
            verbose=True)


    if args.do_onnx_runtime:
        import onnxruntime

        ort_session = onnxruntime.InferenceSession("vitex_pytorch.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        input11 = np.load('onnx_input.npy')
        input22 = np.array([0.002]).astype(np.float32)
        model.eval()

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        s = model(torch.from_numpy(input11[:8]).to(args.device), 6)
        print(s)
        ort_outs = ort_session.run(None, {"x": input11[:8]})

        print(ort_outs)
        



        #torch.onnx.export(model, dummy_input, "vitex_pytorch.onnx", verbose=True)

    # from pytorch_memlab import MemReporter
    # size = (8, 3, args.img_size, args.img_size)
    # dummy_inputs = (
    #     torch.ones(size, dtype=torch.float).to(args.device)
    # )

    # reporter = MemReporter(model)
    # output = model(dummy_inputs)
    # reporter.report()

    # import torchprofile
    # macs = torchprofile.profile_macs(model, args=dummy_inputs)

    # print('MACs: ', macs)


if __name__ == "__main__":
    main()
