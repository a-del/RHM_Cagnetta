import os
import pickle as pk
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_all_data(path, rec=False, encoder_df=None, eval_df=None):
    encoder_runs = []
    eval_runs = []
    for file in os.listdir(path):
        if not file.endswith(".pk"):
            if rec:
                try:
                    encoder_df, eval_df = get_all_data(os.path.join(path, file), rec=False, encoder_df=encoder_df, eval_df=eval_df)
                    continue
                except NotADirectoryError:
                    continue
            else:
                continue
        try:
            with open(os.path.join(path, file), "rb") as f:
                args = pk.load(f)
                data = pk.load(f)
        except:
            print("Passing file", file)
            continue
        if "clf" not in file and args.loss=="clapp_unsup":
            this_dct = {"name": file[:-3], "encoder_run": file[:-3], "epochs_lst": data["epoch"],
                        "train_loss": data["train loss"], "test_loss": data.get("test loss"), "layer": -1, **vars(args)}
            for i in range(10):   # 10 will never even be reached
                if this_dct["encoder_run"].endswith(f"_l{i}"):
                    this_dct["encoder_run"] = this_dct["encoder_run"].split(f"_l{i}")[0]
                    this_dct["layer"] = i
            encoder_runs.append(this_dct)
        else:
            eval_runs.append({"name": file[:-3], "encoder_run": os.path.basename(args.output).split("_clf")[0],
                              "train_loss": data["train loss"], "train_err": data.get("train err"),
                              "test_loss": data.get("test loss"),
                              "test_err": (data["test err"] if "test err" in data else data["terr"]),
                              "epochs_lst": data["epoch"],
                              "best_acc": data["best"]["acc"] if "best" in data and "acc" in data["best"] else np.nan,
                              "best_ep": data["best"]["epoch"] if "best" in data and "epoch" in data["best"] else np.nan,
                              **vars(args)},
                             )
    enc = pd.DataFrame(encoder_runs)
    ev = pd.DataFrame(eval_runs)
    if encoder_df is not None:
        enc = pd.concat([encoder_df, enc])
    if eval_df is not None:
        ev = pd.concat([eval_df, ev])

    return enc, ev


def augment_df(encoder_runs, eval_runs):
    encoder_runs["final_loss"] = encoder_runs.train_loss.apply(lambda x: np.mean(x[-10:]))
    encoder_runs["smooth_loss"] = encoder_runs.train_loss.apply(lambda x: smoothen(x, 10))

    # %%
    def when_get_under_4(row):
        below = row.smooth_loss < 3
        return row.epochs_lst[below.argmax()] if below.any() else 5555
    
    def add_ds_data(df):
        df["correlations"] = df.m < df.num_features
        df["Pmax"] = df.m ** ((df.s ** df.num_layers - 1) // (df.s - 1)) * df.num_classes
        df["Pstar"] = df.num_classes * df.m ** df.num_layers
    
    if not encoder_runs.empty:
        encoder_runs["first_under_4"] = encoder_runs.apply(when_get_under_4, axis=1)
        add_ds_data(encoder_runs)

    if not eval_runs.empty:
        add_ds_data(eval_runs)
        eval_runs["random_err"] = 100 - 100 / eval_runs.num_classes


def load_data(path, rec=False):
    orig_folder = os.path.basename(path)
    save_file = os.path.join("analysis", orig_folder)
    encoder_runs = pd.read_csv(os.path.join(save_file, ("rec_" if rec else "") + "encoder_runs.csv"), index_col=0)
    eval_runs = pd.read_csv(os.path.join(save_file, ("rec_" if rec else "") + "eval_runs.csv"), index_col=0)
    def custom_eval(x):
        if not isinstance(x, str):
            return x
        nan = np.nan
        return eval(x)
    encoder_runs["epochs_lst"] = encoder_runs["epochs_lst"].apply(custom_eval)
    encoder_runs["train_loss"] = encoder_runs["train_loss"].apply(custom_eval)
    encoder_runs["test_loss"] = encoder_runs["test_loss"].apply(custom_eval)
    encoder_runs["width"] = encoder_runs["width"].apply(custom_eval)
    encoder_runs["name"] = encoder_runs["name"].apply(str)
    encoder_runs["encoder_run"] = encoder_runs["encoder_run"].apply(str)
    eval_runs["epochs_lst"] = eval_runs["epochs_lst"].apply(custom_eval)
    eval_runs["train_loss"] = eval_runs["train_loss"].apply(custom_eval)
    eval_runs["test_loss"] = eval_runs["test_loss"].apply(custom_eval)
    eval_runs["train_err"] = eval_runs["train_err"].apply(custom_eval)
    eval_runs["test_err"] = eval_runs["test_err"].apply(custom_eval)
    eval_runs["width"] = eval_runs["width"].apply(custom_eval)
    eval_runs["encoder_run"] = eval_runs["encoder_run"].apply(str)
    return encoder_runs, eval_runs


def save_data(encoder_runs:pd.DataFrame, eval_runs, path, rec=False):
    orig_folder = os.path.basename(path)
    save_file = os.path.join("analysis", orig_folder)
    os.makedirs(save_file, exist_ok=True)
    encoder_runs.to_csv(os.path.join(save_file, ("rec_" if rec else "") + "encoder_runs.csv"))
    eval_runs.to_csv(os.path.join(save_file, ("rec_" if rec else "") + "eval_runs.csv"))


def load_and_plot_all(path="logs/", load=False, save=False, rec=False, col_fun=None):
    if load:
        encoder_runs, eval_runs = load_data(path, rec=rec)
    else:
        encoder_runs, eval_runs = get_all_data(path, rec=rec)
        augment_df(encoder_runs, eval_runs)
    if save:
        save_data(encoder_runs, eval_runs, path, rec=rec)
    # mask = encoder_runs.epochs_lst.apply(lambda x: x[-1] < 700)
    # joint = pd.merge(encoder_runs, eval_runs, left_on="name", right_on="encoder_run", suffixes=("_enc", "_dec")).drop("name_enc", axis=1)
    # df = joint.epochs_lst_enc.apply(lambda x: x[-1] < 700)
    # encoder_runs = eval_runs
    of_interest = encoder_runs# [(encoder_runs.epochs_lst.apply(lambda x: 1000 < x[-1] < 2000)) & (encoder_runs.m == 4) & (encoder_runs.num_layers == 3)]
    # of_interest = encoder_runs[(encoder_runs["name"].apply(lambda x: "clapp_loss_3_4" in x))]
    # eval_runs = eval_runs[eval_runs.encoder_run.isin(of_interest["name"])]
    if col_fun is None:
        col_fun = setup_colors(of_interest)
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


def plot_all_train_losses(df:pd.DataFrame, title=None, col_fun=None, smooth=20, legend=True, shift=0, test=False):
    # smooth=1 for no smoothing; smooth is width of averaging window
    fig, ax = plt.subplots(figsize=(15,10))
    i = 0
    for _, row in df.iterrows():
        col, a, ls = col_fun(row) if col_fun is not None else (None, None, None)
        if test:
            eps = [ep for ep in row.epochs_lst if not ep%10]
            data = row.test_loss
        else:
            eps = row.epochs_lst
            data = row.train_loss
        ax.plot(eps, smoothen(data, width=smooth)+shift*i, label=row["name"], color=col, linewidth=1, alpha=a, linestyle=ls)   # [x+0.002*i for x in row.train_loss]
        i += 1
    ax.set_xlabel("Epochs")
    ax.set_ylabel(f"{'Test' if test else 'Train'} loss")
    if legend:
        ax.legend()
    ax.set_yscale('log')
    ax.set_title(title)
    return fig, ax


def plot_all_test_errors(df:pd.DataFrame, title=None, col_fun=None, legend=True, shift=0, rd_level=True, train=False):
    fig, ax = plt.subplots(figsize=(15,10))
    if rd_level and "random_err" in df:
        for rd in df.random_err.unique():
            ax.axhline(rd, color="gray", alpha=0.7)
    j = 0
    for _, row in df.iterrows():
        col, a, ls = col_fun(row) if col_fun is not None else (None, None, None)
        if not train:
            eps = [ep for ep in row.epochs_lst if not ep%10]
            data = row.test_err
        else:
            eps = row.epochs_lst
            data = row.train_err
        ax.plot(eps, [x+shift*j for x in data], label=row["name"], color=col, alpha=a, linestyle=ls)
        ax.scatter([row.best_ep], [100 - row.best_acc], marker="*", color=col, alpha=a)
        j += 1
    if legend:
        ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel(f"{'Train' if train else 'Test'} error (%)")
    ax.set_title(title)
    return fig, ax


def compare_rows(df):
    columns_to_drop = ["epochs_lst", "train_loss", "test_err", "best_ep", "output", "output_sfx"]
    columns_to_keep = []
    for col in df.columns:
        if col in columns_to_drop:
            continue
        try:
            n = df[col].nunique()
        except TypeError:
            print(f"Column {col} is non hashable.")   # Todo could check by hand
            columns_to_keep.append(col)
        else:
            if n > 1:
                columns_to_keep.append(col)
    smaller_df = df[columns_to_keep]
    return smaller_df


def setup_colors(df):
    colormap = mpl.colormaps["tab20"]
    col_dict = {row["name"]: colormap(i%20) for i, row in df.iterrows()}
    def col_fun(row):
        try:
            c = col_dict[row["name"].split("_clf")[0]]
        except KeyError:
            print("Not finding color for", row["name"])
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


def setup_colors_width():
    def col_fun(row):
        if row.m == 8:
            ls = "--"
        else:
            ls = "-"
        if row.width[:3] == [2*row.m**2,2*row.m**2,4*row.m**2]:
            c = "C0"
        elif row.width[:3] == [2*row.m**2, 4*row.m**2, 8*row.m**2]:
            c = "C2"
        elif row.width[:3] == [4*row.m**2 ,8*row.m**2 ,12*row.m**2 ]:
            c = "C1"
        elif row.width[:3] == [4*row.m**2 ,4*row.m**2 ,4*row.m**2]:
            c = "gray"
        else:
            c = "k"
        if row.lr == 0.002:
            a = 0.15
        elif row.lr == 0.005:
            a = 0.35
        elif row.lr == 0.01:
            a = 0.6
        elif row.lr == 0.02:
            a = 0.8
        else:
            a = 1
        return c, a, ls
    return col_fun


def setup_colors_width2():
    def col_fun(row):
        if row.m == 8:
            a=0.8
        else:
            a=0.4
        if row.width[:3] == [2*row.m**2,2*row.m**2,4*row.m**2]:
            c = "C0"
        elif row.width[:3] == [2*row.m**2, 4*row.m**2, 8*row.m**2]:
            c = "C2"
        elif row.width[:3] == [4*row.m**2 ,8*row.m**2 ,12*row.m**2 ]:
            c = "C1"
        elif row.width[:3] == [4*row.m**2 ,4*row.m**2 ,4*row.m**2]:
            c = "gray"
        else:
            c = "k"
        if row.lr == 0.002:
            ls = ':'
        elif row.lr == 0.005:
            ls = '-.'
        elif row.lr == 0.01:
            ls = '--'
        elif row.lr == 0.02:
            ls = '-'
        else:
            ls = '-'
        return c, a, ls
    return col_fun


def setup_colors_preds():
    def col_fun(row):
        print(row.m, row.width)
        if row.m >= 8:
            ls = "--"
        else:
            ls = "-"
        if row.k_predictions == 1:
            c = "C0"
        elif row.k_predictions == 3:
            c = "C2"
        elif row.k_predictions == 8:
            c = "C1"
        else:
            c = "k"
        return c, 1, ls
    return col_fun


if __name__ == '__main__':
    load_and_plot_all("/Volumes/lcncluster/delrocq/code/RHM_Cagnetta/logs/rd", save=True, rec=False)#, col_fun=setup_colors_width2())
    # load_and_plot_all("/Volumes/lcncluster/delrocq/code/RHM_Cagnetta/logs/figure", load=True)
