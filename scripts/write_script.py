def write_script():
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

python -m main --ptr {-p_fac} --pte -1 --optim sgd --lr 0.01 --num_features {n} --m {m} --num_layers {l} --net_layers {l} --width {2*n**2} --zero_loss_epochs 5 --net cnn --dataset hier1 --epochs 5000 --device cuda --layerwise 1 --last_lin_layer 0 --loss clapp_unsup --zero_loss_threshold {0.03} --output logs/fig_{i}

"""

                txt_yml = f"""apiVersion: run.ai/v2alpha1
kind: TrainingWorkload
metadata:
  name: fig-{i}
spec:
  image:
    value: nvcr.io/nvidia/pytorch:20.03-py3
  name:
    value: fig-{i}
  runAsUser:
    value: true
  command:  # entrypoint
    value: "bash"
  arguments:  # arguments, space-separated
    value: "/lcncluster/delrocq/runai_files/fig/n{n}_l{l}_{i}.sh"
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

                with open(f"/Volumes/lcncluster/delrocq/runai_files/fig/n{n}_l{l}_{i}.sh", "w") as f:
                    f.write(txt_sh)
                with open(f"/Users/lcn1/Desktop/Athese/runai_files/figure/{i}.yaml", "w") as f:
                    f.write(txt_yml)
                i += 1

    print("Done")


if __name__ == '__main__':
    write_script()
