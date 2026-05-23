from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import choix
from scipy.stats import kendalltau
warnings.filterwarnings("ignore")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

#Setup
rng = np.random.default_rng(69)

N_MODELS = 8
N_ITEMS = 300
REPEATS = 8
BAD_FRAC = 0.20


#data generation

models = [f"model_{i}" for i in range(N_MODELS)]
true_theta = rng.normal(0, 1, N_MODELS)

bad_items = set(rng.choice(N_ITEMS, int(BAD_FRAC * N_ITEMS), replace=False))
rows = []

for item in range(N_ITEMS):
    item_noise = 0.75 if item in bad_items else 0.0

    for _ in range(REPEATS):
        a, b = rng.choice(N_MODELS, 2, replace=False)

        p = np.exp(true_theta[a]) / (np.exp(true_theta[a]) + np.exp(true_theta[b]))

        # Bad items partially randomize outcomes.
        p = (1 - item_noise) * p + item_noise * 0.5

        winner = a if rng.random() < p else b

        rows.append({
            "item_id": f"item_{item}",
            "model_a": models[a],
            "model_b": models[b],
            "winner": models[winner],
            "is_bad_item": item in bad_items,
        })

df = pd.DataFrame(rows)

#helpers
#fit model
def fit_bt(data):
    comps = []
    for _, r in data.iterrows():
        a = models.index(r["model_a"])
        b = models.index(r["model_b"])
        if r["winner"] == r["model_a"]:
            comps.append((a, b))
        else:
            comps.append((b, a))

    theta = choix.ilsr_pairwise(N_MODELS, comps, alpha=0.01)
    return pd.Series(theta, index=models).sort_values(ascending=False)

#set tops
def topk_agreement(rank_a, rank_b, k=3):
    return len(set(rank_a.index[:k]) & set(rank_b.index[:k])) / k

#pair
def pairwise_consistency(rank_a, rank_b):
    pos_a = {m: i for i, m in enumerate(rank_a.index)}
    pos_b = {m: i for i, m in enumerate(rank_b.index)}
    agree = 0
    total = 0

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            agree += (pos_a[m1] < pos_a[m2]) == (pos_b[m1] < pos_b[m2])
            total += 1

    return agree / total

#kendall
def kendall(rank_ref, rank_sub):
    ref_pos = {m: i for i, m in enumerate(rank_ref.index)}
    sub_pos = {m: i for i, m in enumerate(rank_sub.index)}
    x = [ref_pos[m] for m in models]
    y = [sub_pos[m] for m in models]
    return kendalltau(x, y).statistic

#informative
def item_entropy(data):
    # Higher entropy = more outcome ambiguity / inconsistency.
    def entropy(g):
        p = g["winner"].value_counts(normalize=True)
        return float(-(p * np.log(p + 1e-12)).sum())

    return data.groupby("item_id").apply(entropy).rename("entropy").reset_index()

#sampling
def sample_items(data, method, budget):
    item_ids = data["item_id"].unique()
    n_keep = max(5, int(len(item_ids) * budget))

    if method == "random":
        chosen = rng.choice(item_ids, n_keep, replace=False)

    elif method == "filtered":
        scores = item_entropy(data)
        chosen = scores.sort_values("entropy").head(n_keep)["item_id"].to_numpy()

    elif method == "informative":
        # Keep moderately difficult/ambiguous items, avoid extremely noisy ones.
        scores = item_entropy(data)
        median_entropy = scores["entropy"].median()
        scores["distance"] = (scores["entropy"] - median_entropy).abs()
        chosen = scores.sort_values("distance").head(n_keep)["item_id"].to_numpy()

    else:
        raise ValueError(method)

    return data[data["item_id"].isin(chosen)]

#ground truth
full_rank = fit_bt(df)

results = []

#sampling loops
for method in ["random", "filtered", "informative"]:
    for budget in [0.25, 0.50, 0.75]:
        taus, tops, pcs = [], [], []

        for seed in range(20):
            sub = sample_items(df, method, budget)
            sub_rank = fit_bt(sub)

            taus.append(kendall(full_rank, sub_rank))
            tops.append(topk_agreement(full_rank, sub_rank, k=3))
            pcs.append(pairwise_consistency(full_rank, sub_rank))

        results.append({
            "method": method,
            "budget": budget,
            "kendall_tau_mean": np.mean(taus),
            "top3_agreement_mean": np.mean(tops),
            "pairwise_consistency_mean": np.mean(pcs),
        })

#result
res = pd.DataFrame(results)

print("\nFull ranking:")
print(full_rank)

print("\nResults:")
print(res)

FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def plot_metric(res, metric_mean,  ylabel, title, filename, ylim=None):
    plt.figure(figsize=(7, 5))

    for method in res["method"].unique():
        subset = res[res["method"] == method].sort_values("budget")

        plt.errorbar(
            subset["budget"],
            subset[metric_mean],
            #@yerr=subset[metric_std],
            marker="o",
            capsize=4,
            label=method,
        )

    plt.xlabel("Budget Fraction")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True)
    plt.legend()
    plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


plot_metric(
    res,
    "kendall_tau_mean",
    #"kendall_tau_std",
    "Kendall's Tau",
    "Ranking Stability vs Evaluation Budget",
    "syn_kendall_tau_plot.png",
    ylim=(0.80, 1.02),
)

plot_metric(
    res,
    "top3_agreement_mean",
    #"top10_agreement_std",
    "Top-3 Agreement",
    "Top-3 Agreement vs Evaluation Budget",
    "syn_top10_agreement_plot.png",
    ylim=(0.75, 1.05),
)

plot_metric(
    res,
    "pairwise_consistency_mean",
    #"pairwise_consistency_std",
    "Pairwise Consistency",
    "Pairwise Consistency vs Evaluation Budget",
    "syn_pairwise_consistency_plot.png",
    ylim=(0.90, 1.02),
)

print("\nSaved figures:")
print(FIG_DIR / "syn_kendall_tau_plot.png")
print(FIG_DIR / "syn_top3_agreement_plot.png")
print(FIG_DIR / "syn_pairwise_consistency_plot.png")