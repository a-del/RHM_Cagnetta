def write_script_dec():
    p_fac = 2
    m,l = 4, 3
    s = 2
    ols = [("sgd", 0.2), ("sgd", 0.1), ("sgd", 0.05), ("adam", 0.01), ("adam", 0.001), ("adam", 0.004)]

    i = 8
    for optim, lr in ols:
        for _ in [1]:
            for _ in [1]:
                v = m
                n = m

                txt_sh = f"""echo "Start of training job"

conda init bash
source ~/.bashrc

cd /lcncluster/delrocq/code/RHM_Cagnetta
conda activate clappvision5

echo "Really starting"

python -m eval --ptr {-p_fac} --pte -1 --optim {optim} --lr {lr} --zero_loss_epochs 10 --dataset hier1 --epochs 5000 --device cuda --layerwise 0 --zero_loss_threshold {0.001} --output logs/figure/fig_18 --output_sfx '{i}'

"""

                txt_yml = f"""apiVersion: run.ai/v2alpha1
kind: TrainingWorkload
metadata:
  name: tst-dec-{i}
spec:
  image:
    value: nvcr.io/nvidia/pytorch:20.03-py3
  name:
    value: tst-dec-{i}
  runAsUser:
    value: true
  command:  # entrypoint
    value: "bash"
  arguments:  # arguments, space-separated
    value: "/lcncluster/delrocq/runai_files/test_better/dec_{i}.sh"
  cpu:
    value: "2"
  memory:
    value: 40Gi
  memoryLimit:
    value: 100Gi
  largeShm:
    value: true  
  gpu:
    value: "1"
  environment:
    items:
      HOME:
        value: /lcncluster/delrocq/.try_clapp_home # PATH to HOME e.g. /lcncluster/user/.caas_HOME
  pvcs:
    items:
      pvc--0:  # First is "pvc--0", second "pvc--1", etc.
        value:
          claimName: runai-lcn1-delrocq-lcncluster
          existingPvc: true
          path: /lcncluster

  nodePools:
    value: "g9"
    """

                with open(f"/Volumes/lcncluster/delrocq/runai_files/test_better/dec_{i}.sh", "w") as f:
                    f.write(txt_sh)
                with open(f"/Users/lcn1/Desktop/Athese/runai_files/test_better/dec_{i}.yaml", "w") as f:
                    f.write(txt_yml)
                i += 1

    print("Done")


def write_test_script():
    p_fac = 2
    m,l = 6, 4
    s = 2

    i = 100
    for width in [1, 2, 4, 10]:
        for mom in [0, 0.9]:
            for lr in [0.01, 0.004]:
                v = m
                n = m

                txt_sh = f"""echo "Start of training job"

conda init bash
source ~/.bashrc

cd /lcncluster/delrocq/code/RHM_Cagnetta
conda activate clappvision5

echo "Really starting"

python -m main --ptr {-p_fac} --pte -1 --optim sgd --lr {lr} --momentum {mom} --weight_decay 0 --scheduler exponential --num_features {n} --m {m} --num_layers {l} --net_layers {l} --width {width*n**2} --zero_loss_epochs 5 --net cnn --dataset hier1 --epochs 5000 --device cuda --layerwise 1 --last_lin_layer 0 --loss clapp_unsup --zero_loss_threshold {0.005} --output logs/test_better/fig_{i} --eval_optim sgd --eval_lr 0.1 --eval_epochs 5000 --eval_zero_loss_threshold 0.0005

"""

                txt_yml = f"""apiVersion: run.ai/v2alpha1
kind: TrainingWorkload
metadata:
  name: tst-{i}
spec:
  image:
    value: nvcr.io/nvidia/pytorch:20.03-py3
  name:
    value: tst-{i}
  runAsUser:
    value: true
  command:  # entrypoint
    value: "bash"
  arguments:  # arguments, space-separated
    value: "/lcncluster/delrocq/runai_files/test_better/test_{i}.sh"
  cpu:
    value: "2"
  memory:
    value: 40Gi
  memoryLimit:
    value: 100Gi
  largeShm:
    value: true  
  gpu:
    value: "1"
  environment:
    items:
      HOME:
        value: /lcncluster/delrocq/.try_clapp_home # PATH to HOME e.g. /lcncluster/user/.caas_HOME
  pvcs:
    items:
      pvc--0:  # First is "pvc--0", second "pvc--1", etc.
        value:
          claimName: runai-lcn1-delrocq-lcncluster
          existingPvc: true
          path: /lcncluster

  nodePools:
    value: "g9"
    """

                with open(f"/Volumes/lcncluster/delrocq/runai_files/test_better/test_{i}.sh", "w") as f:
                    f.write(txt_sh)
                with open(f"/Users/lcn1/Desktop/Athese/runai_files/test_better/{i}.yaml", "w") as f:
                    f.write(txt_yml)
                i += 1

    print("Done")


def write_figure_script():
    p_facs = [0.8, 1, 2, 4]
    ms = [3, 4, 6, 8, 12]
    ls = [2,3,4]
    s = 2

    i = 0
    for m in ms:
        for l in ls:
            for p_fac in p_facs:
                v = m
                n = m

                txt_sh = f"""echo "Start of training job"

conda init bash
source ~/.bashrc

cd /lcncluster/delrocq/code/RHM_Cagnetta
conda activate clappvision5

echo "Really starting"

python -m main --ptr {-p_fac} --pte -1 --optim sgd --num_features {n} --m {m} --num_layers {l} --net_layers {l} --width {2*n**2} --net cnn --dataset hier1 --epochs 5000 --device cuda --layerwise 0 --last_lin_layer 1 --output logs/fig_orig_{i}

"""

                txt_yml = f"""apiVersion: run.ai/v2alpha1
kind: TrainingWorkload
metadata:
  name: fig-or-{i}
spec:
  image:
    value: nvcr.io/nvidia/pytorch:20.03-py3
  name:
    value: fig-or-{i}
  runAsUser:
    value: true
  command:  # entrypoint
    value: "bash"
  arguments:  # arguments, space-separated
    value: "/lcncluster/delrocq/runai_files/fig_orig/n{n}_l{l}_{i}.sh"
  cpu:
    value: "2"
  memory:
    value: 40Gi
  memoryLimit:
    value: 100Gi
  largeShm:
    value: true  
  gpu:
    value: "1"
  environment:
    items:
      HOME:
        value: /lcncluster/delrocq/.try_clapp_home # PATH to HOME e.g. /lcncluster/user/.caas_HOME
  pvcs:
    items:
      pvc--0:  # First is "pvc--0", second "pvc--1", etc.
        value:
          claimName: runai-lcn1-delrocq-lcncluster
          existingPvc: true
          path: /lcncluster

  nodePools:
    value: "g9"
    """

                if p_fac == 2:
                    with open(f"/Volumes/lcncluster/delrocq/runai_files/fig_orig/n{n}_l{l}_{i}.sh", "w") as f:
                        f.write(txt_sh)
                    with open(f"/Users/lcn1/Desktop/Athese/runai_files/figure_orig/{i}.yaml", "w") as f:
                        f.write(txt_yml)
                i += 1

    print("Done")


if __name__ == '__main__':
    # write_figure_script()
    write_test_script()
    # write_script_dec()
