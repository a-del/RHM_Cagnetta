import os
import pickle as pk
import numpy as np
import pandas as pd
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
                              "epochs": data["epoch"], "train_loss": data["train loss"], "test_err": data["terr"], "best_acc": data["best"]["acc"] if "best" in data and "acc" in data["best"] else np.nan})
    return pd.DataFrame(encoder_runs), pd.DataFrame(eval_runs)


def load_and_plot_all(path="logs/"):
    encoder_runs, eval_runs = get_all_data(path)
    plot_all_train_losses(encoder_runs)
    plot_all_train_losses(eval_runs)
    plot_all_test_errors(eval_runs)
    plt.show()


def plot_all_train_losses(df:pd.DataFrame):
    fig, ax = plt.subplots()
    for _, row in df.iterrows():
        ax.plot(row.epochs, row.train_loss, label=row["name"])
    ax.legend()
    ax.set_yscale('log')


def plot_all_test_errors(df:pd.DataFrame):
    fig, ax = plt.subplots()
    for _, row in df.iterrows():
        ax.plot([ep for ep in row.epochs if not ep%10], row.test_err, label=row["name"])
    ax.legend()
    # ax.set_yscale('log')


if __name__ == '__main__':
    load_and_plot_all("/Volumes/lcncluster/delrocq/code/RHM_Cagnetta/logs")
