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
        if "clf" not in file:
            encoder_runs.append({"name": file[:-3], "epochs": data["epoch"], "train_loss": data["train loss"]})
        else:
            eval_runs.append({"name": file[:-3], "encoder_run": os.path.basename(args.output).split("_clf")[0],
                              "epochs": data["epoch"], "train_loss": data["train loss"], "test_err": data["terr"],
                              "best_acc": data["best"]["acc"] if "best" in data and "acc" in data["best"] else np.nan,
                              "best_ep": data["best"]["epoch"] if "best" in data and "epoch" in data["best"] else np.nan},
                             )
    return pd.DataFrame(encoder_runs), pd.DataFrame(eval_runs)


def load_and_plot_all(path="logs/"):
    encoder_runs, eval_runs = get_all_data(path)
    col_fun = setup_colors(encoder_runs)
    plot_all_train_losses(encoder_runs, title="encoder", col_fun=col_fun)
    plot_all_train_losses(eval_runs, title="decoder", col_fun=col_fun)
    plot_all_test_errors(eval_runs, col_fun=col_fun)
    plt.show()


def plot_all_train_losses(df:pd.DataFrame, title=None, col_fun=None):
    fig, ax = plt.subplots()
    for _, row in df.iterrows():
        col = col_fun(row["name"]) if col_fun is not None else None
        ax.plot(row.epochs, row.train_loss, label=row["name"], color=col)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Train loss")
    ax.legend()
    ax.set_yscale('log')
    ax.set_title(title)


def plot_all_test_errors(df:pd.DataFrame, title=None, col_fun=None):
    fig, ax = plt.subplots()
    for _, row in df.iterrows():
        col = col_fun(row["name"]) if col_fun is not None else None
        test_epochs = [ep for ep in row.epochs if not ep%10]
        ax.plot(test_epochs, row.test_err, label=row["name"], color=col)
        ax.scatter([row.best_ep], [100 - row.best_acc], marker="*", c=col)
    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Test error (%)")
    ax.set_title(title)

def setup_colors(df):
    colormap = mpl.colormaps["tab20"]
    col_dict = {row["name"]: colormap(i%20) for i, row in df.iterrows()}
    def col_fun(name):
        return col_dict[name.split("_clf")[0]]
    return col_fun



if __name__ == '__main__':
    load_and_plot_all("/Volumes/lcncluster/delrocq/code/RHM_Cagnetta/logs")
