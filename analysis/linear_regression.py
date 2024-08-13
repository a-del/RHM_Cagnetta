import os
import argparse
import pickle

import numpy as np
import torch
from sklearn import linear_model

from main import set_up
from utils import args2train_test_sizes, reload_model


def get_encodings_data(context_model, data_loader):
    reps = []
    targets = []

    for step, (img, target) in enumerate(data_loader):

        with torch.no_grad():
            y, outs = context_model(img)

        reps.append(y.numpy())
        targets.append(target.numpy())

    return reps, targets


def prepare_regression_input(encodings):
    # # flatten encodings if need be
    # inputs_f = [np.reshape(z, (z.shape[0], -1)) for z in encodings]
    inputs = np.vstack(encodings)
    return inputs


def prepare_regression_targets(targets):
    # targets: list of batches, each batch is 1-dim.
    # Converting target indices to one-hot:
    tgt_idx = np.hstack(targets)  # shape (tot_nb_targets,)
    n_tot = tgt_idx.shape[0]
    tgt = np.zeros((n_tot, np.amax(tgt_idx) + 1))
    tgt[np.arange(n_tot), tgt_idx] = 1

    return tgt_idx, tgt


def perform_linear_regression(encs, tgts):
    clf = linear_model.Ridge()  # Simple linear ridge regression of the 1-hot labels over the network outputs
    clf.fit(encs, tgts)
    # print(f"Fitting time: {time.time() - tic:.3g}")
    return clf


def test_linear_regression(clf, test_encs, test_targets_idx):
    t = clf.predict(test_encs)
    tm = np.argmax(t, axis=1)
    tacc = np.mean(test_targets_idx==tm)
    print(f"Test linear regression accuracy:     {100*tacc:.3g}%")
    return tacc


def linear_regression_from_encodings(encs, tgts, test_encs, test_targets_idx):
    # perform linear regression
    clf = perform_linear_regression(encs, tgts)
    # Test the model
    test_acc = test_linear_regression(clf, test_encs, test_targets_idx)
    return test_acc


def test_by_lin_reg(model, train_loader, test_loader):
    encodings, targets = get_encodings_data(model, train_loader)
    test_encodings, test_targets = get_encodings_data(model, test_loader)
    encs = prepare_regression_input(encodings)
    test_encs = prepare_regression_input(test_encodings)

    _, tgts = prepare_regression_targets(targets)
    test_targets_idx, _ = prepare_regression_targets(test_targets)

    test_acc = linear_regression_from_encodings(encs, tgts, test_encs, test_targets_idx)
    return test_acc


def main_linreg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--output", type=str, required=False, default="None")
    parser.add_argument("--output_sfx", type=str, required=False, default=None)

    args = parser.parse_args()

    args, data = reload_model(args)
    args.ptr, args.pte = args2train_test_sizes(args)
    args.output = os.path.join(os.path.dirname(args.output), os.path.basename(args.output)+"_linreg"+(
        f"_{args.output_sfx}" if args.output_sfx else ''))
    args.device = "cpu"
    trainloader, testloader, net0, criterion = set_up(args, crit=False)
    if "best" in data:
        state_dict = data["best"]["net"]
        args.epoch_loaded = data["best"]["epoch"]
    else:
        state_dict = data["last"]
        args.epoch_loaded = data["epochs_lst"][-1]
    net0.load_state_dict(state_dict)
    net0.eval_mode()
    net0.beta = None
    test_acc = test_by_lin_reg(net0, trainloader, testloader)
    args.linreg_test_acc = test_acc
    with open(args.output + ".txt", "w") as handle:
        handle.write(f"test_acc: {test_acc:.3g}")


if __name__ == '__main__':
    main_linreg()

