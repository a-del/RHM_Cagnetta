"""
    Train networks on 1d hierarchical models of data.
"""

import os
import argparse
import time
import math
import pickle
from models import *
import copy
from functools import partial

from init import init_fun
from optim_loss import loss_func, regularize, opt_algo, measure_accuracy
from utils import cpu_state_dict, args2train_test_sizes
# from observables import locality_measure, state2permutation_stability, state2clustering_error   # will be needed if stability, locality or clustering_error


def run(args, trainloader, testloader, net0, criterion):

    # if args.dtype == 'float64':
    #     torch.set_default_dtype(torch.float64)
    # if args.dtype == 'float32':
    #     torch.set_default_dtype(torch.float32)
    # if args.dtype == 'float16':
    #     torch.set_default_dtype(torch.float16)

    best_acc = 0  # best test accuracy
    best_loss = 1e6

    # scale batch size when larger than train-set size
    if (args.batch_size >= args.ptr) and args.scale_batch_size:
        args.batch_size = args.ptr // 2

    if args.save_dynamics:
        dynamics = [{"acc": 0.0, "epoch": 0., "net": cpu_state_dict(net0)}]
    else:
        dynamics = None

    loss_tr = []
    loss_te = []
    err_tr = []
    err_te = []
    locality = []
    stability = []
    clustering_error = []
    epochs_list = []

    best = dict()
    trloss_flag = 0

    for net, epoch, losstr, acc_tr, avg_epoch_time in train(args, trainloader, net0, criterion):

        assert str(losstr) != "nan", "Loss is nan value!!"
        loss_tr.append(losstr)
        err_tr.append(100 - acc_tr)
        epochs_list.append(epoch)

        # measuring locality for fcn nets
        if args.locality == 1:
            assert args.net == 'fcn', "Locality can only be computed for fcns !!"
            state = net.state_dict()
            hidden_layers = [state[k] for k in state if 'w' in k][:-2]
            with torch.no_grad():
                locality.append(locality_measure(hidden_layers, args)[0])

        # measure stability to semantically equivalent data realizations
        if args.stability == 1:
            state = net.state_dict()
            stability.append(state2permutation_stability(state, args))
        if args.clustering_error == 1:
            state = net.state_dict()
            clustering_error.append(state2clustering_error(state, args))

        if epoch % 10 != 0 and not args.save_dynamics: continue

        acc, lo = test(args, testloader, net, criterion, print_flag=epoch % 5 == 0)
        err_te.append(100 - acc)
        loss_te.append(lo)

        if args.save_dynamics:
        #     and (
        #     epoch
        #     in (10 ** torch.linspace(-1, math.log10(args.epochs), 30)).int().unique()
        # ):
            # save dynamics at 30 log-spaced points in time
            dynamics.append(
                {"acc": acc, "epoch": epoch, "net": cpu_state_dict(net)}
            )
        if args.loss == "clapp_unsup" and losstr < best_loss:
            best["loss"] = losstr
            best["epoch"] = epoch
            if args.save_best_net:
                best["net"] = cpu_state_dict(net)
            # if args.save_dynamics:
            #     dynamics.append(best)
            best_loss = losstr
            print(f"BEST LOSS ({losstr:.02g}) at epoch {epoch:.02f} !!", flush=True)

        elif args.loss != "clapp_unsup" and acc > best_acc:
            best["acc"] = acc
            best["epoch"] = epoch
            if args.save_best_net:
                best["net"] = cpu_state_dict(net)
            # if args.save_dynamics:
            #     dynamics.append(best)
            best_acc = acc
            print(f"BEST ACCURACY ({acc:.02f}) at epoch {epoch:.02f} !!", flush=True)

        out = {
            "args": args,
            "epoch": epochs_list,
            "train loss": loss_tr,
            "train err": err_tr,
            "test loss": loss_te,
            "test err": err_te,
            "locality": locality,
            "stability": stability,
            "clustering_error": clustering_error,
            "dynamics": dynamics,
            "best": best,
        }

        yield out, net   # this is just so it is saved to pickle

        if (losstr == 0 and args.loss == 'hinge') or (losstr < args.zero_loss_threshold and args.loss in ['cross_entropy', 'clapp_unsup']):
            trloss_flag += 1
            if trloss_flag >= args.zero_loss_epochs:
                break

    try:
        wo = weights_evolution(net0, net)
    except:
        print("Weights evolution failed!")
        wo = None

    if args.locality == 2:
        assert args.net == 'fcn', "Locality can only be computed for fcns !!"
        state = net.state_dict()
        hidden_layers = [state[k] for k in state if 'w' in k][:-2]
        with torch.no_grad():
            locality.append(locality_measure(hidden_layers, args)[0])

    if args.stability == 2:
        state = net.state_dict()
        stability.append(state2permutation_stability(state, args))

    if args.clustering_error == 2:
        state = net.state_dict()
        clustering_error.append(state2clustering_error(state, args))

    out = {
        "args": args,
        "epoch": epochs_list,
        "train loss": loss_tr,
        "train err": err_tr,
        "test loss": loss_te,
        "test err": err_te,
        "locality": locality,
        "stability": stability,
        "clustering_error": clustering_error,
        "dynamics": dynamics,
        "init": cpu_state_dict(net0) if args.save_init_net else None,
        "best": best,
        "last": cpu_state_dict(net) if args.save_last_net else None,
        "weight_evo": wo,
        'avg_epoch_time': avg_epoch_time,
    }
    yield out, net


def set_up(args, net=True, crit=True):
    if crit:
        if args.loss == 'clapp_unsup':
            criterion = None
        else:
            criterion = partial(loss_func, args)
    else:
        criterion = None

    if net:
        trainloader, testloader, net0 = init_fun(args)
        # if args.loss == "clapp_unsup":
        #     testloader = None
    else:
        trainloader, testloader, net0 = None, None, None

    return trainloader, testloader, net0, criterion


def train(args, trainloader, net0, criterion):

    net = copy.deepcopy(net0)

    optimizer, scheduler = opt_algo(args, net)
    print(f"Training for {args.epochs} epochs...")

    start_time = time.time()

    num_batches = math.ceil(args.ptr / args.batch_size)
    checkpoint_batches = torch.linspace(0, num_batches, 10, dtype=int)

    for epoch in range(args.epochs):

        # layerwise training
        if epoch % (args.epochs // args.net_layers + 1) == 0:
            if 'layerwise' in args.net:
                l = epoch // (args.epochs // args.net_layers + 1)
                net.init_layerwise_(l)
                print(f'Layer-wise training up to layer {l}.', flush=True)

        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            if args.net == "cnn":
                loss = net.compute_loss(outputs, targets, criterion)
            else:
                loss = criterion(outputs, targets)
            train_loss += loss.detach().item()
            regularize(loss, net, args.weight_decay, reg_type=args.reg_type)
            loss.backward()
            optimizer.step()

            b_correct, b_total = measure_accuracy(args, outputs[0], targets)   # WARNING, even for layerwise only computes acc of last layer
            correct += b_correct
            total += b_total

            # during first epoch, save some sgd steps instead of after whole epoch
            if epoch < 10 and batch_idx in checkpoint_batches and batch_idx != (num_batches - 1):
                yield net, epoch + (batch_idx + 1) / num_batches, loss / (batch_idx + 1), 100*b_correct / b_total, None

        avg_epoch_time = (time.time() - start_time) / (epoch + 1)

        if epoch % 5 == 0:
            print(
                f"[Train epoch {epoch+1} / {args.epochs}, {print_time(avg_epoch_time)}/epoch, ETA: {print_time(avg_epoch_time * (args.epochs - epoch - 1))}]"
                f"[tr.Loss: {train_loss * args.alpha / (batch_idx + 1):.03f}]"
                f"[tr.Acc: {100.*correct/total:.03f}, {correct} / {total}]",
                flush=True
            )

        scheduler.step()

        yield net, epoch + 1, train_loss / (batch_idx + 1), 100*correct / total, avg_epoch_time


def test(args, testloader, net, criterion, print_flag=True):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            if args.net == "cnn":
                loss = net.compute_loss(outputs, targets, criterion)
            else:
                loss = criterion(outputs, targets)

            test_loss += loss.item()

            b_correct, b_total = measure_accuracy(args, outputs[0], targets)   # WARNING, even for layerwise only computes acc of last layer
            correct += b_correct
            total += b_total

        if print_flag:
            print(
                f"[TEST][te.Loss: {test_loss * args.alpha / (batch_idx + 1):.03f}]"
                f"[te.Acc: {100. * correct / total:.03f}, {correct} / {total}]",
                flush=True
            )

    return 100.0 * correct / total, test_loss / (batch_idx + 1)


# timing function
def print_time(elapsed_time):

    # if less than a second, print milliseconds
    if elapsed_time < 1:
        return f"{elapsed_time * 1000:.00f}ms"

    elapsed_seconds = round(elapsed_time)

    m, s = divmod(elapsed_seconds, 60)
    h, m = divmod(m, 60)

    elapsed_time = []
    if h > 0:
        elapsed_time.append(f"{h}h")
    if not (h == 0 and m == 0):
        elapsed_time.append(f"{m:02}m")
    elapsed_time.append(f"{s:02}s")

    return "".join(elapsed_time)


def weights_evolution(f0, f):
    s0 = f0.state_dict()
    s = f.state_dict()
    nd = 0
    for k in s:
        nd += (s0[k] - s[k]).norm() / s0[k].norm()
    nd /= len(s)
    return nd.item()


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
    parser.add_argument("--dataset", type=str, required=True)    # hier1 for hierarchical
    parser.add_argument("--ptr", type=float, default=0.8,
        help="Number of training point. If in [0, 1], fraction of training points w.r.t. total. If negative argument, P = |arg|*P_star",
    )
    parser.add_argument("--pte", type=float, default=.2)   # proportion of data to keep for test set?
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--scale_batch_size", type=int, default=0)

    parser.add_argument("--background_noise", type=float, default=0)

    # Hierarchical dataset #
    parser.add_argument("--num_features", type=int, default=8)   # v
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=-1)
    parser.add_argument("--input_format", type=str, default="onehot")
    parser.add_argument("--whitening", type=int, default=0)
    parser.add_argument("--auto_regression", type=int, default=0)    # not for now

    ### ARCHITECTURES ARGS ###
    parser.add_argument("--net", type=str, required=True)    # fcn or cnn
    parser.add_argument("--random_features", type=int, default=0)   # for no training

    ## Nets params ##
    parser.add_argument("--width", type=str, default="64")
    parser.add_argument("--net_layers", type=int, default=2)   # TODO why default is wrong with num_layers default??
    parser.add_argument("--filter_size", type=int, default=2)
    # parser.add_argument("--stride", type=int, default=2)   # not used
    parser.add_argument("--batch_norm", type=int, default=0)
    parser.add_argument("--bias", type=int, default=1, help="for some archs, controls bias presence")
    parser.add_argument("--last_lin_layer", type=int, default=1,
                        help="for cnn, whether to use self.beta, which is a linear layer from width to num_features")


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
    parser.add_argument("--k_predictions", type=int, default=1, help="for clapp only, how many predictions (masks)")
    parser.add_argument("--prop_fixed_pred", type=float, default=0.5, help="proportion of neurons hidden for pred")
    parser.add_argument("--detach_c", type=int, default=0, help="detach context for 1-direction-only gradients")
    parser.add_argument("--sequential", type=int, default=0, help="train layers one after the other")
    parser.add_argument("--random_pred", type=int, default=0, help="do not use fixed masking,"
                                                                   "but different random masking every time")

    parser.add_argument(
        "--alpha", default=1.0, type=float, help="alpha-trick parameter"
    )
    
    ### ALGORITHM ARGS FOR CLASSIFIER IF SEPARATE ###
    parser.add_argument("--eval_optim", type=str, default="sgd")
    parser.add_argument("--eval_scheduler", type=str, default="cosineannealing")
    parser.add_argument("--eval_lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--eval_momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--eval_weight_decay", default=5e-4, type=float)
    parser.add_argument("--eval_reg_type", default='l2', type=str)
    parser.add_argument("--eval_epochs", type=int, default=250)
    parser.add_argument("--eval_zero_loss_epochs", type=int, default=5)
    parser.add_argument("--eval_zero_loss_threshold", type=float, default=0.01)
    parser.add_argument("--eval_rescale_epochs", type=int, default=0)

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
    if args.num_classes == -1:
        args.num_classes = args.num_features
    if args.net_layers == -1:
        args.net_layers = args.num_layers
    if args.m == -1:
        args.m = args.num_features

    if args.layerwise == -1:
        args.layerwise = 1 if args.loss == "clapp_unsup" else 0

    # define train and test sets sizes

    args.ptr, args.pte = args2train_test_sizes(args)
    args.width = eval(args.width)   # either int or list of int

    if args.sequential:
        layers_to_train = range(args.num_layers)
    else:
        layers_to_train = ["all"]

    # with open(args.output+".pk", "wb") as handle:
    #     pickle.dump(args, handle)
    try:
        trainloader, testloader, net0, criterion = set_up(args)
        for layer in layers_to_train:
            net0.train_only_layer(layer)
            for data in run(args, trainloader, testloader, net0, criterion):
                with open(args.output+("" if layer == 'all' else f"_l{layer}")+".pk", "wb") as handle:
                    pickle.dump(args, handle)   # a bit useless as args is also in data
                    pickle.dump(data[0], handle)
        if args.loss == "clapp_unsup":
            print("\n\nDone training encoder; now training classifier!!\n\n")
            args.output = os.path.join(os.path.dirname(args.output), os.path.basename(args.output)+"_clf")
            args.loss = "cross_entropy"
            net = data[1]
            net.eval_mode()
            args_to_change = ["eval_optim", "eval_scheduler", "eval_lr", "eval_momentum", "eval_weight_decay",
                              "eval_reg_type", "eval_epochs", "eval_zero_loss_epochs", "eval_zero_loss_threshold", "eval_rescale_epochs"]
            for a in args_to_change:
                setattr(args, a[5:], getattr(args, a))
            _, _, _, criterion = set_up(args, net=False)
            # TODO need to change anything else??
            # Todo should have separate lr and early stopping criteria?
            for data in run(args, trainloader, testloader, net, criterion):
                with open(args.output + ".pk", "wb") as handle:
                    pickle.dump(args, handle)  # a bit useless as args is also in data
                    pickle.dump(data[0], handle)
        print("Done!!\n\n\n")
    except:
        # os.remove(args.output + ".pk")
        print("Fail!!\n\n\n")
        raise


if __name__ == "__main__":
    main()
