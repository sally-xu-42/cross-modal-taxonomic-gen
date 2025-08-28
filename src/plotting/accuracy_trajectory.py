import os
import re
import glob
import argparse
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

PAT = re.compile(r"evaluation_step-(\d+)-epoch-(\d+)-loss=([0-9.]+).pt_summary.csv$")

def load(model_name):
    results_dir = os.path.join("./results/", model_name)
    csvs = glob.glob(os.path.join(results_dir, "evaluation_step-*-epoch-*-loss=*.pt*.csv"))

    dfs = []
    for f in csvs:
        m = PAT.search(os.path.basename(f))
        if not m: 
            continue
        step, epoch, loss = m.groups()
        df = pd.read_csv(f)
        df["step"] = int(step)
        df["epoch"] = int(epoch)
        df["loss"] = float(loss)
        df["file"] = os.path.basename(f)
        dfs.append(df)

    if not dfs:
        print(f"No matching CSVs in {results_dir}")
        return

    out = pd.concat(dfs, ignore_index=True).sort_values(["step"])
    out_path = os.path.join(results_dir, "ppl_trajectory.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows -> {out_path}")

def plot(model_name):
    df = pd.read_csv(os.path.join("./results/", model_name, "ppl_trajectory.csv"))
    df["Accuracy"] = df["Accuracy"].astype(str).str.rstrip("%").astype(float) / 100.0
    df = df.sort_values("step")
    sns.set_theme(style="whitegrid")
    ax = sns.lineplot(data=df, x="step", y="Accuracy", hue="Relation", marker="o")
    ax.set(xlabel="Step", ylabel="Accuracy", title=f"{model_name}: Accuracy trajectory")
    plt.tight_layout()
    out_png = os.path.join("./results/", model_name, "accuracy_trajectory.png")
    plt.savefig(out_png)
    print(f"Saved plot -> {out_png}")


if __name__=="__main__":
    # run from root directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help="Model Name")
    args = parser.parse_args()
    
    # load(args.model_name)
    plot(args.model_name)