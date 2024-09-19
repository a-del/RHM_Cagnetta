from functools import partial
import itertools

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
            self.n_shown = max(1, int(self.leng * (1 - prop_hidden)))
            self.n_masked = self.leng - self.n_shown
            if self.n_masked <= 0:
                print("WARNING: Not enough length to mask along space; masking along channels.")
                self.masking_axis = "none"
        if self.masking_axis == "none":
            self.n_shown = max(1, int(self.input_size * (1 - prop_hidden)))
            self.n_masked = self.input_size - self.n_shown
        if self.random_masking:
            z_size = self.input_size
            c_size = self.input_size
        else:
            if self.masking_axis == "space":
                z_size = self.n_shown * self.c_in
            else:
                z_size = self.n_shown
            c_size = input_size - z_size
        if not self.random_masking:
            self._build_mask()

        self.Wpred = nn.ModuleList(nn.Linear(c_size, z_size, bias=False) for _ in range(k_predictions))

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
                batch_mask = torch.vmap(partial(torch.masked_select, mask=mask))   # TODO checkout what this does!! does it selected only unmasked features, and since they change from mask to mask, impose a form of order in them?
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
                    [rd.choice([True for _ in range(self.n_shown)] + [False for _ in range(self.n_masked)],
                               size=(self.c_in, self.leng), replace=False)
                     for _ in range(batch_size)],   # b, input_size
                    axis=0
                ), device=device if device >= 0 else "cpu"
                )   # b, c_in, leng

            elif self.masking_axis == "space":
                mask = torch.tensor(np.stack(
                    [rd.choice([True for _ in range(self.n_shown)] + [False for _ in range(self.n_masked)],
                               size=(self.leng,), replace=False)
                     for _ in range(batch_size)],   # b, leng
                    axis=0
                ), device=device if device >= 0 else "cpu"
                ).unsqueeze(1)   # b, 1, leng

            return mask
        else:
            self.masks = [
                torch.tensor(rd.choice([True for _ in range(self.n_shown)] + [False for _ in range(self.n_masked)],
                                       size=(self.n_shown+self.n_masked,), replace=False))
                for _ in range(self.k_predictions)]
            # each: input_size, or leng,
            self.masks = [mask.reshape(-1, self.leng) for mask in self.masks]
            # each: 1, leng or c_in, leng


class CLAPPUnsupervisedNoMasking(nn.Module):
    """
    Computes the CLAPP loss using no labels.
    Supposes that input is of size bs*(k+1) instead of bs, where for each element of the original batch there is one
    input that has prop_hidden masked indices in its spatial dimension, and the following k elements are the same
    with different 1-prop_hidden masked indices.
    """
    def __init__(self, c_in, leng, k_predictions=1, detach_c=False, either_pos_or_neg=False, **kwargs):
        super().__init__()
        self.k_predictions = k_predictions
        self.detach_c = detach_c

        input_size = c_in*leng
        self.c_in = c_in
        self.leng = leng
        self.input_size = input_size
        z_size = self.input_size
        c_size = self.input_size

        self.Wpred = nn.ModuleList(nn.Linear(c_size, z_size, bias=False) for _ in range(k_predictions))

    def forward(self, reprs: torch.Tensor, y):
        # reprs: b, chans, len
        b, chans, leng = reprs.size()
        b = b // (self.k_predictions+1)

        b_tot = b * self.k_predictions
        c = reprs.reshape(b, self.k_predictions+1, chans, leng)[:, 1:].reshape(b_tot, chans, leng)
        z = reprs[::self.k_predictions+1].reshape(b, 1, chans, leng).repeat(1, self.k_predictions, 1, 1)   # b, k , c, l

        if self.detach_c:
            c = c.detach()
        zhat = self.Wpred[0](c.reshape(b_tot, chans * leng)).reshape(b, self.k_predictions, chans * leng)
        # b, k, c*l

        # positive samples:
        u_pos = torch.einsum("bkij,bkjl->bk", z.reshape(b, self.k_predictions, 1, -1), zhat.unsqueeze(3))   # b, k
        loss_pos = ((1 - u_pos).relu()).mean()

        # negative samples: shuffle zhat along batch dimension such that predictions are across 2 different words
        idx = torch.randperm(b_tot)
        zhat_shuf = zhat.reshape(b_tot, chans * leng)[idx]
        u_neg = torch.einsum("bkij,bkjl->bk", z.reshape(b_tot, self.k_predictions, 1, -1), zhat_shuf.unsqueeze(3))   # b, k
        loss_neg = ((1 + u_neg).relu()).mean()

        loss = (loss_pos + loss_neg) / 2
        return loss


class MaskInputs(nn.Module):
    def __init__(self, leng, k_predictions=1, prop_hidden=0.5, random_masking=False,
                 either_pos_or_neg=False):
        if either_pos_or_neg or not random_masking:
            raise NotImplementedError
        # TODO prop_hidden=0.5 might be way too hard, but if we always do the symmetric (do we?), then smaller is worse in the sym case
        super().__init__()
        self.leng = leng
        self.k_predictions = k_predictions
        self.random_masking = random_masking
        self.n_shown = max(1, int(self.leng * (1 - prop_hidden)))
        self.n_masked = self.leng - self.n_shown

    def forward(self, x):
        # x: bs, chans, leng
        bs, chans, leng = x.size()
        device = x.get_device()
        x = x.repeat(self.k_predictions+1, 1, 1, 1   # k_preds+1, bs, chans, leng
                     ).permute(1, 0, 2, 3).reshape(-1, chans, leng)   # bs*(k_preds+1), chans, leng
        if not self.random_masking:
            mask = self._build_mask(batch_size=bs, device=device)
        return x*mask

    def _build_mask(self, batch_size=None, device=None):
        if self.random_masking:
            mask = torch.tensor(np.stack(itertools.chain(
            [rd.choice([True for _ in range(self.n_shown)] + [False for _ in range(self.n_masked)],
                      size=(self.leng,), replace=False)
            ] + [rd.choice([True for _ in range(self.n_masked)] + [False for _ in range(self.n_shown)],
                          size=(self.leng,), replace=False)
                 for _ in range(self.k_predictions)
            ]
            for _ in range(batch_size))),
            device=device if device >= 0 else "cpu").unsqueeze(1)   # ((k+1)*b, 1, leng)

            return mask
        else:
            pass
            # self.masks = [
            #     torch.tensor(rd.choice([True for _ in range(self.n_shown)] + [False for _ in range(self.n_masked)],
            #                            size=(self.n_shown+self.n_masked,), replace=False))
            #     for _ in range(self.k_predictions)]
            # # each: input_size, or leng,
            # self.masks = [mask.reshape(-1, self.leng) for mask in self.masks]
            # # each: 1, leng or c_in, leng

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
