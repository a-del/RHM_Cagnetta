"""
    Train networks on 1d hierarchical models of data.
"""

import os
import argparse
import time
import math
import pickle

from main import set_up, run
from models import *
import copy
from functools import partial

from init import init_fun
from optim_loss import loss_func, regularize, opt_algo, measure_accuracy
from utils import cpu_state_dict, args2train_test_sizes
# from observables import locality_measure, state2permutation_stability, state2clustering_error   # will be needed if stability, locality or clustering_error




def main():

    parser = argparse.ArgumentParser()

    ### Tensors type ###
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")

    ### Seeds ###
    parser.add_argument("--seed_init", type=int, default=0)  # seed random-hierarchy-model
    parser.add_argument("--seed_net", type=int, default=-1)  # network initalisation
    parser.add_argument("--seed_trainset", type=int, default=-1)  # training sample

    ### DATASET ARGS ###
    parser.add_argument("--dataset", type=str, default="hier1")    # hier1 for hierarchical
    parser.add_argument("--ptr", type=float, default=0.8,
        help="Number of training point. If in [0, 1], fraction of training points w.r.t. total. If negative argument, P = |arg|*P_star",
    )
    parser.add_argument("--pte", type=float, default=.2)   # proportion of data to keep for test set?
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--scale_batch_size", type=int, default=0)


    ### ARCHITECTURES ARGS ###
    parser.add_argument("--net", type=str, default="cnn")
    parser.add_argument("--random_features", type=int, default=0)   # for no training


    # ## Auto-regression with Transformers ##
    # parser.add_argument("--pmask", type=float, default=.2)    # not for now


    ### ALGORITHM ARGS ###
    parser.add_argument("--loss", type=str, default="cross_entropy")
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--scheduler", type=str, default="cosineannealing")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--reg_type", default='l2', type=str)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--zero_loss_epochs", type=int, default=0)
    parser.add_argument("--zero_loss_threshold", type=float, default=0.01)
    parser.add_argument("--rescale_epochs", type=int, default=0)
    parser.add_argument("--layerwise", type=int, default=-1)

    parser.add_argument(
        "--alpha", default=1.0, type=float, help="alpha-trick parameter"
    )

    ### Observables ###
    # how to use: 1 to compute stability every checkpoint; 2 at end of training. Default 0.
    parser.add_argument("--stability", type=int, default=0)
    parser.add_argument("--clustering_error", type=int, default=0)
    parser.add_argument("--locality", type=int, default=0)

    ### SAVING ARGS ###
    parser.add_argument("--save_init_net", type=int, default=1)
    parser.add_argument("--save_best_net", type=int, default=1)
    parser.add_argument("--save_last_net", type=int, default=1)
    parser.add_argument("--save_dynamics", type=int, default=0)

    ## saving path ##
    parser.add_argument("--pickle", type=str, required=False, default="None")
    parser.add_argument("--output", type=str, required=False, default="None")
    args = parser.parse_args()

    if args.pickle == "None":
        assert (
            args.output != "None"
        ), "either `pickle` or `output` must be given to the parser!!"
        args.pickle = args.output

    # special value -1 to set some equal arguments
    if args.seed_trainset == -1:
        args.seed_trainset = args.seed_init
    if args.seed_net == -1:
        args.seed_net = args.seed_init

    if args.layerwise == -1:
        args.layerwise = 1 if args.loss == "clapp_unsup" else 0


    with open(args.output + ".pk", "rb") as handle:
        args_saved = pickle.load(handle)
        data = pickle.load(handle)
    args_saved = vars(args_saved)
    args_saved.update(vars(args))
    args = argparse.Namespace(**args_saved)   # or could define a special function to update() directly namespaces
    args.ptr, args.pte = args2train_test_sizes(args)
    args.loss = "cross_entropy"
    args.last_lin_layer = 0
    trainloader, testloader, net0, criterion = set_up(args)
    args.output = os.path.join(os.path.dirname(args.output), os.path.basename(args.output)+"_clfe")
    if "best" in data:
        state_dict = data["best"]["net"]
    else:
        state_dict = data["last"]
    tbdel = []
    for k in state_dict.keys():
        if k.startswith("losses"):
            tbdel.append(k)   # cannot delete keys during iteration
    for k in tbdel:
        del state_dict[k]
    net0.load_state_dict(state_dict)
    net0.evaluating = True
    net0.layerwise = False   # Todo should regroup at least these 2 into a method of net??
    # TODO need to change anything else??
    for data in run(args, trainloader, testloader, net0, criterion):
        with open(args.output + ".pk", "wb") as handle:
            pickle.dump(args, handle)  # a bit useless as args is also in data
            pickle.dump(data[0], handle)


if __name__ == "__main__":
    main()
