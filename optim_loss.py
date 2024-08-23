from functools import partial

import torch
from torch import nn
import torch.optim as optim

import numpy as np
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
    def __init__(self, c_in, leng, k_predictions=1, prop_hidden=0.5, detach_c=False, random_masking=False,
                 masking_axis=None, either_pos_or_neg=False):
        super().__init__()
        self.k_predictions = k_predictions
        self.detach_c = detach_c
        self.random_masking = random_masking
        self.masking_axis = masking_axis
        if self.masking_axis not in ["none", "space"]: raise ValueError

        input_size = c_in*leng
        self.c_in = c_in
        self.leng = leng
        self.input_size = input_size
        if self.masking_axis == "space":
            self.z_size = int(self.leng * (1 - prop_hidden))
            self.c_size = self.leng - self.z_size
        elif self.masking_axis == "none":
            if self.random_masking:
                self.z_size = self.input_size
                self.c_size = self.input_size
            else:
                self.z_size = int(input_size * (1 - prop_hidden))
                self.c_size = input_size - self.z_size
        if not self.random_masking:
            self._build_mask()

        if self.random_masking:
            self.Wpred = nn.ModuleList(nn.Linear(self.input_size, self.input_size, bias=False) for _ in range(k_predictions))
        else:
            fac = self.c_in if self.masking_axis == "space" else 1
            self.Wpred = nn.ModuleList(nn.Linear(self.c_size * fac, self.z_size * fac, bias=False) for _ in range(k_predictions))

    def forward(self, reprs: torch.Tensor, y):
        # reprs: b, chans, len
        tot_loss = 0

        device = reprs.get_device()
        b = reprs.size(0)
        # if self.masking_axis == "none":
        #     reprs = reprs.reshape(b, -1)
        for k in range(self.k_predictions):
            if self.random_masking:
                mask = self._build_mask(b, device)
                c = reprs
                z = mask * reprs

            else:
                mask = self.masks[k]
                if device >= 0:
                    mask = mask.to(device)
                batch_mask = torch.vmap(partial(torch.masked_select, mask=mask))
                batch_anti_mask = torch.vmap(partial(torch.masked_select, mask=~mask))
                z = batch_mask(reprs)
                c = batch_anti_mask(reprs)
            if self.detach_c:
                c = c.detach()
            zhat = self.Wpred[0](c.reshape(b, -1))

            # positive samples:
            u_pos = torch.einsum("bij,bjk->b", z.reshape(b, 1, -1), zhat.unsqueeze(2))   # b,
            loss_pos = ((1 - u_pos).relu()).mean()

            # negative samples: shuffle zhat along batch dimension such that predictions are across 2 different words
            idx = torch.randperm(b)
            zhat_shuf = zhat[idx]
            u_neg = torch.einsum("bij,bjk->b", z.reshape(b, 1, -1), zhat_shuf.unsqueeze(2))   # b,
            loss_neg = ((1 + u_neg).relu()).mean()

            tot_loss = tot_loss + (loss_pos + loss_neg) / 2
        return tot_loss / self.k_predictions   # Todo is avg ok, or want just sum?

    def _build_mask(self, batch_size=None, device=None):
        if self.random_masking:
            if self.masking_axis == "none":
                mask = torch.tensor(np.stack(
                    [rd.choice([True for _ in range(self.z_size)] + [False for _ in range(self.c_size)],
                              size=(self.c_in, self.leng), replace=False)
                     for _ in range(batch_size)],   # b, input_size
                    axis=0
                ), device=device if device >= 0 else "cpu"
                )   # b, c_in, leng

            elif self.masking_axis == "space":
                mask = torch.tensor(np.stack(
                    [rd.choice([True for _ in range(self.z_size)] + [False for _ in range(self.c_size)],
                               size=(self.leng,), replace=False)
                     for _ in range(batch_size)],   # b, leng
                    axis=0
                ), device=device if device >= 0 else "cpu"
                ).unsqueeze(1)   # b, 1, leng

            return mask
        else:
            self.masks = [
                torch.tensor(rd.choice([True for _ in range(self.z_size)] + [False for _ in range(self.c_size)],
                                       size=(self.z_size+self.c_size,), replace=False))
                for _ in range(self.k_predictions)]
            # each: input_size, or leng,
            self.masks = [mask.reshape(-1, self.leng) for mask in self.masks]
            # each: 1, leng or c_in, leng


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

def measure_accuracy(args, out, targets):
    """
    Compute out accuracy on targets. Returns running number of correct and total predictions.
    """
    total = targets.size(0)

    if args.loss == "clapp_unsup":
        return 0, total
    if args.loss != "hinge":
        _, predicted = out.max(1)
        correct = predicted.eq(targets).sum().item()
    else:
        correct = (out * targets > 0).sum().item()

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
