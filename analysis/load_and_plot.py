import os
import pickle as pk
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_all_data(path):
    encoder_runs = []
    eval_runs = []
    for file in os.listdir(path):
        if not file.endswith(".pk"):
            continue
        try:
            with open(os.path.join(path, file), "rb") as f:
                args = pk.load(f)
                data = pk.load(f)
        except:
            print("Passing file", file)
            continue
        if "clf" not in file and args.loss=="clapp_unsup":
            encoder_runs.append({"name": file[:-3], "epochs_lst": data["epoch"], "train_loss": data["train loss"], **vars(args)})
        else:
            eval_runs.append({"name": file[:-3], "encoder_run": os.path.basename(args.output).split("_clf")[0],
                              "epochs_lst": data["epoch"], "train_loss": data["train loss"], "test_err": data["terr"],
                              "best_acc": data["best"]["acc"] if "best" in data and "acc" in data["best"] else np.nan,
                              "best_ep": data["best"]["epoch"] if "best" in data and "epoch" in data["best"] else np.nan,
                              **vars(args)},
                             )
    return pd.DataFrame(encoder_runs), pd.DataFrame(eval_runs)


def load_data(path):
    orig_folder = os.path.basename(path.split("logs/")[1])
    save_file = os.path.join("analysis", orig_folder)
    encoder_runs = pd.read_csv(os.path.join(save_file, "encoder_runs.csv"))
    eval_runs = pd.read_csv(os.path.join(save_file, "eval_runs.csv"))
    encoder_runs["epochs_lst"] = encoder_runs["epochs_lst"].apply(eval)
    encoder_runs["train_loss"] = encoder_runs["train_loss"].apply(eval)
    eval_runs["epochs_lst"] = eval_runs["epochs_lst"].apply(eval)
    eval_runs["train_loss"] = eval_runs["train_loss"].apply(eval)
    eval_runs["test_err"] = eval_runs["test_err"].apply(eval)
    return encoder_runs, eval_runs


def save_data(encoder_runs:pd.DataFrame, eval_runs, path):
    orig_folder = os.path.basename(path.split("logs/")[1])
    save_file = os.path.join("analysis", orig_folder)
    os.makedirs(save_file, exist_ok=True)
    encoder_runs.to_csv(os.path.join(save_file, "encoder_runs.csv"))
    eval_runs.to_csv(os.path.join(save_file, "eval_runs.csv"))


def load_and_plot_all(path="logs/", load=False, save=False):
    if load:
        encoder_runs, eval_runs = load_data(path)
    else:
        encoder_runs, eval_runs = get_all_data(path)
    if save:
        save_data(encoder_runs, eval_runs, path)
    # mask = encoder_runs.epochs_lst.apply(lambda x: x[-1] < 700)
    # joint = pd.merge(encoder_runs, eval_runs, left_on="name", right_on="encoder_run", suffixes=("_enc", "_dec")).drop("name_enc", axis=1)
    # df = joint.epochs_lst_enc.apply(lambda x: x[-1] < 700)
    # encoder_runs = eval_runs
    of_interest = encoder_runs# [(encoder_runs.epochs_lst.apply(lambda x: 1000 < x[-1] < 2000)) & (encoder_runs.m == 4) & (encoder_runs.num_layers == 3)]
    col_fun = setup_colors_dec(of_interest)
    plot_all_train_losses(of_interest, title="encoder", col_fun=col_fun)
    # eval_of_interest = eval_runs[eval_runs.encoder_run == "fig_18"]
    plot_all_train_losses(eval_runs, title="decoder", col_fun=col_fun)
    plot_all_test_errors(eval_runs, title="decoder", col_fun=col_fun)
    plt.show()


def smoothen(vals, width=20):
    l = len(vals)
    x = np.zeros((l + 2*width-2,))
    sm = np.zeros((l,))
    k = np.ones((width,))
    x[:width-1] = vals[0]
    x[width-1:width-1+l] = vals
    x[width-1+l:] = vals[-1]
    for i in range(l):
        sm[i] = (x[i:i+width] * k).mean()
    return sm


def plot_all_train_losses(df:pd.DataFrame, title=None, col_fun=None, smooth=20):
    # smooth=1 for no smoothing; smooth is width of averaging window
    fig, ax = plt.subplots()
    i = 0
    for _, row in df.iterrows():
        col, a, ls = col_fun(row) if col_fun is not None else None, None, None
        ax.plot(row.epochs_lst, smoothen(row.train_loss, width=smooth), label=row["name"], color=col, linewidth=1, alpha=a, linestyle=ls)   # [x+0.002*i for x in row.train_loss]
        i += 1
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Train loss")
    ax.legend()
    ax.set_yscale('log')
    ax.set_title(title)


def plot_all_test_errors(df:pd.DataFrame, title=None, col_fun=None):
    fig, ax = plt.subplots()
    for _, row in df.iterrows():
        col, a, ls = col_fun(row) if col_fun is not None else None, None, None
        test_epochs = [ep for ep in row.epochs_lst if not ep%10]
        ax.plot(test_epochs, row.test_err, label=row["name"], color=col, alpha=a, linestyle=ls)
        ax.scatter([row.best_ep], [100 - row.best_acc], marker="*", color=col, alpha=a)
    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Test error (%)")
    ax.set_title(title)


def setup_colors(df):
    colormap = mpl.colormaps["tab20"]
    col_dict = {row["name"]: colormap(i%20) for i, row in df.iterrows()}
    def col_fun(row):
        try:
            c = col_dict[row["name"].split("_clf")[0]]
        except KeyError:
            c = "k"
        return c, 1, "-"
    return col_fun


def setup_colors_orig(df):
    col_dict = {3: "C0", 4: "C1", 6: "C3", 8: "C5", 12: "C7"}
    ls_dict = {2: ":", 3: "-.", 4: "--"}
    def col_fun(row):
        c = col_dict[row.m]
        ls = ls_dict[row.num_layers]
        return c, 1, ls
    return col_fun


def setup_colors_dec(df):
    col_dict = {0.1: "C3", 0.01: "C1", 0.001: "C2", 0.0001: "C0"}
    col_dict = {0.2: "C3", 0.1: "C4", 0.05: "C6", 0.01: "C0", 0.004: "C9", 0.001: "C8", 0.0001: "C5"}
    ls_dict = {"sgd": "-", "adam": "--"}
    def col_fun(row):
        c = col_dict[row.lr]
        ls = ls_dict[row.optim]
        return c, 1, ls
    return col_fun


if __name__ == '__main__':
    # load_and_plot_all("/Volumes/lcncluster/delrocq/code/RHM_Cagnetta/logs/test_better", save=True)
    load_and_plot_all("/Volumes/lcncluster/delrocq/code/RHM_Cagnetta/logs/figure", load=True)
