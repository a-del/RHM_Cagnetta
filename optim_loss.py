from functools import partial

import torch
from torch import nn
import torch.optim as optim

import numpy.random as rd


seed = 42
rd.seed(seed)
# TODO better than this!!


def loss_func(args, o, y):
    """
    Compute the loss.
    :param args: parser args
    :param o: network prediction
    :param y: true labels
    :return: value of the loss
    """
    if args.loss == "cross_entropy":
        loss = nn.functional.cross_entropy(args.alpha * o, y, reduction="mean")
    elif args.loss == "hinge":
        loss = ((1 - args.alpha * o * y).relu()).mean()
    else:
        raise NameError("Specify a valid loss function in [cross_entropy, hinge]")

    return loss


class CLAPPUnsupervisedHalfMasking(nn.Module):
    """
    Computes the CLAPP loss using no labels.
    To avoid the use of labels, the half-masked encodings are used to predict the complementary-masked encodings.
    """
    def __init__(self, c_in, leng, k_predictions=1, prop_hidden=0.5, either_pos_or_neg=False):
        super().__init__()
        input_size = c_in*leng
        self.z_size = int(input_size * (1-prop_hidden))
        self.c_size = input_size - self.z_size
        self.Wpred = nn.ModuleList(nn.Linear(self.c_size, self.z_size, bias=False) for _ in range(k_predictions))
        self.k_predictions = k_predictions
        self.masks = [torch.tensor(rd.choice([True for _ in range(self.z_size)] + [False for _ in range(self.c_size)],
                                             size=(input_size,), replace=False)) for _ in range(self.k_predictions)]

    def forward(self, reprs: torch.Tensor, y):
        # reprs: b, chans, len
        tot_loss = 0

        device = reprs.get_device()
        b = reprs.size(0)
        reprs = reprs.reshape(b, -1)
        for k in range(self.k_predictions):
            mask = self.masks[k]
            if device >= 0:
                mask = mask.to(device)
            batch_mask = torch.vmap(partial(torch.masked_select, mask=mask))
            batch_anti_mask = torch.vmap(partial(torch.masked_select, mask=~mask))
            z = batch_mask(reprs)
            c = batch_anti_mask(reprs)
            zhat = self.Wpred[0](c.reshape(b, self.c_size))

            # positive samples:
            u_pos = torch.einsum("bij,bjk->b", z.reshape(b, 1, self.z_size), zhat.unsqueeze(2))   # b,
            loss_pos = ((1 - u_pos).relu()).mean()

            # negative samples: shuffle zhat along batch dimension such that predictions are across 2 different words
            idx = torch.randperm(b)
            zhat_shuf = zhat[idx]
            u_neg = torch.einsum("bij,bjk->b", z.reshape(b, 1, self.z_size), zhat_shuf.unsqueeze(2))   # b,
            loss_neg = ((1 + u_neg).relu()).mean()

            tot_loss = tot_loss + (loss_pos + loss_neg) / 2
        return tot_loss / self.k_predictions   # Todo is avg ok, or want just sum?


def regularize(loss, f, l, reg_type):
    """
    add L1/L2 regularization to the loss.
    :param loss: current loss
    :param f: network function
    :param args: parser arguments
    """
    for p in f.parameters():
        if reg_type == 'l1':
            loss += l * p.abs().mean()
        elif reg_type == 'l2':
            loss += l * p.pow(2).mean()

def measure_accuracy(args, out, targets, correct, total):
    """
    Compute out accuracy on targets. Returns running number of correct and total predictions.
    """
    if args.loss != "hinge":
        _, predicted = out.max(1)
        correct += predicted.eq(targets).sum().item()
    else:
        correct += (out * targets > 0).sum().item()

    total += targets.size(0)

    return correct, total


def opt_algo(args, net):
    """
    Define training optimizer and scheduler.
    :param args: parser args
    :param net: network function
    :return: torch scheduler and optimizer.
    """

    # rescale loss by alpha or alpha**2 if doing feature-lazy
    args.lr = args.lr / args.alpha

    if args.optim == "sgd":
        # if args.net == 'hlcn':
        #     optimizer = optim.SGD([
        #         {
        #             "params": (p for n, p in net.named_parameters() if 'b' not in n and 'weight' not in n),
        #             "lr": args.lr * args.width ** .5,
        #         },
        #         {
        #             "params": (p for n, p in net.named_parameters() if 'b' in n or 'weight' in n),
        #             "lr": args.lr * args.width,
        #         },
        #     ], lr=args.lr * args.width, momentum=0.9
        # )
        # else:
        optimizer = optim.SGD(
            net.parameters(), lr=args.lr * 100, momentum=args.momentum   # 100 is a heuristic, used to be width   # TODO improve that!
        )  ## 5e-4
    elif args.optim == "adam":
        optimizer = optim.Adam(
            net.parameters(), lr=args.lr * 100   # 100 is a heuristic, used to be width   # TODO improve that!
        )  ## 1e-5
    else:
        raise NameError("Specify a valid optimizer [Adam, (S)GD]")

    if args.scheduler == "cosineannealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * 0.8
        )
    elif args.scheduler == "none":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epochs // 3, gamma=1.0
        )
    elif args.scheduler == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)
    else:
        raise NameError(
            "Specify a valid scheduler [cosineannealing, exponential, none]"
        )

    return optimizer, scheduler
