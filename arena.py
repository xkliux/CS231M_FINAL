from pathlib import Path
import matplotlib.pyplot as plt
import choix
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import kendalltau

OUT = Path("results")
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(69)


###### Load data


ds = load_dataset("lmarena-ai/arena-human-preference-55k", split="train")
df = ds.to_pandas()

print("Raw columns:")
print(df.columns)
print(df.head())
print(f"unique prompt count is: {len(df['prompt'].unique())}")
print(f"total df row count is: {len(df)}")



############ Preprocess

required_cols = [ "id","model_a","model_b","winner_model_a","winner_model_b","winner_tie"]

df = df[required_cols].dropna().copy()

# Convert winner flags to int in case they load as bool/string
for c in ["winner_model_a", "winner_model_b", "winner_tie"]:
    df[c] = df[c].astype(int)

# Remove ties
df = df[df["winner_tie"] == 0].copy()

# Keep only rows where exactly one model won
df = df[(df["winner_model_a"] + df["winner_model_b"]) == 1].copy()

# Create winner_model column
df["winner_model"] = np.where( df["winner_model_a"] == 1,df["model_a"], df["model_b"])

# Use original row id as item_id
df["item_id"] = df["id"].astype(str)

#inspect
print("\nAfter preprocessing:")
print(df[["item_id", "model_a", "model_b", "winner_model"]].head())
print("Rows:", len(df))


###################### Filter to models with enough battles
min_battles = 200

counts = pd.concat([df["model_a"], df["model_b"]]).value_counts()
valid_models = set(counts[counts >= min_battles].index)

df = df[df["model_a"].isin(valid_models)& df["model_b"].isin(valid_models)].copy()

models = sorted(valid_models)
model_to_idx = {m: i for i, m in enumerate(models)}


#inspect
print("\nAfter model filtering:")
print("Rows:", len(df))
print("Models:", len(models))
print(models[:20])


#############BT model fitting helper


def fit_bt(data: pd.DataFrame) -> pd.Series:
    comps = []

    for _, r in data.iterrows():
        a = model_to_idx[r["model_a"]]
        b = model_to_idx[r["model_b"]]

        if r["winner_model"] == r["model_a"]:
            comps.append((a, b))  #a beats b
        else:
            comps.append((b, a))  #b beats b

    theta = choix.ilsr_pairwise(len(models), comps, alpha=0.01)

    return pd.Series(theta, index=models).sort_values(ascending=False)



######Metric evaluations helper


def kendall_tau(rank_ref: pd.Series, rank_sub: pd.Series) -> float:
    common = [m for m in rank_ref.index if m in rank_sub.index]

    ref_pos = {m: i for i, m in enumerate(rank_ref.index)}
    sub_pos = {m: i for i, m in enumerate(rank_sub.index)}

    x = [ref_pos[m] for m in common]
    y = [sub_pos[m] for m in common]

    return float(kendalltau(x, y).statistic)


def topk_agreement(rank_ref: pd.Series, rank_sub: pd.Series, k: int = 10) -> float:
    k = min(k, len(rank_ref), len(rank_sub))
    return len(set(rank_ref.index[:k]) & set(rank_sub.index[:k])) / k


def pairwise_consistency(rank_ref: pd.Series, rank_sub: pd.Series) -> float:
    common = [m for m in rank_ref.index if m in rank_sub.index]

    ref_pos = {m: i for i, m in enumerate(rank_ref.index)}
    sub_pos = {m: i for i, m in enumerate(rank_sub.index)}

    agree = 0
    total = 0

    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            m1, m2 = common[i], common[j]
            ref_order = ref_pos[m1] < ref_pos[m2]
            sub_order = sub_pos[m1] < sub_pos[m2]

            agree += ref_order == sub_order
            total += 1

    return agree / total if total > 0 else np.nan



######Item / pair noise scores


def add_noise_scores(data: pd.DataFrame) -> pd.DataFrame:
    scored = data.copy()

    scored["pair_key"] = scored.apply(
        lambda r: "__".join(sorted([r["model_a"], r["model_b"]])),
        axis=1,
    )

    def entropy(g: pd.DataFrame) -> float:
        p = g["winner_model"].value_counts(normalize=True)
        return float(-(p * np.log(p + 1e-12)).sum())

    pair_entropy = (scored.groupby("pair_key").apply(entropy).rename("entropy").reset_index())
    scored = scored.merge(pair_entropy, on="pair_key", how="left")

    return scored


df_scored = add_noise_scores(df)


######item Sampling methods


def sample_rows(data: pd.DataFrame, method: str, budget: float, seed: int) -> pd.DataFrame:
    n = max(100, int(len(data) * budget))
    n = min(n, len(data))

    local_rng = np.random.default_rng(seed)

    if method == "random":
        return data.sample(n=n, random_state=seed)

    if method == "filtered":
        # Low entropy = less ambiguous model-pair outcomes
        return data.sort_values("entropy", ascending=True).head(int(n * 1.5)).copy().sample(n=n, random_state=seed)

    if method == "informative":
        # Moderate entropy can be informative:
        # not trivial, but not maximally noisy.
        median_entropy = data["entropy"].median()
        temp = data.copy()
        temp["distance_to_median_entropy"] = (temp["entropy"] - median_entropy).abs()
        return temp.sort_values("distance_to_median_entropy").head(int(n * 1.5)).copy()

    raise ValueError(f"Unknown method: {method}").sample(n=n, random_state=seed)



##execute experiment


print("\nFitting full reference ranking...")
full_rank = fit_bt(df_scored)

print("\nFull ranking top 20:")
print(full_rank.head(20))

results = []

methods = ["random", "filtered", "informative"]
budgets = [0.25, 0.50, 0.75]
n_reps = 10

for method in methods:
    for budget in budgets:
        taus = []
        tops = []
        pcs = []

        for rep in range(n_reps):
            seed = 69 + rep
            sub = sample_rows(df_scored, method, budget, seed)

            try:
                sub_rank = fit_bt(sub)
                taus.append(kendall_tau(full_rank, sub_rank))
                tops.append(topk_agreement(full_rank, sub_rank, k=min(10, len(models))))
                pcs.append(pairwise_consistency(full_rank, sub_rank))
            except Exception as e:
                print(f"Failed: method={method}, budget={budget}, rep={rep}, error={e}")

        results.append(
            {
                "method": method,
                "budget": budget,
                "kendall_tau_mean": np.mean(taus),
                "kendall_tau_std": np.std(taus),
                "top10_agreement_mean": np.mean(tops),
                "top10_agreement_std": np.std(tops),
                "pairwise_consistency_mean": np.mean(pcs),
                "pairwise_consistency_std": np.std(pcs),
            }
        )

res = pd.DataFrame(results)

print("\nResults:")
print(res)

FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def plot_metric(res, metric_mean, metric_std, ylabel, title, filename, ylim=None):
    plt.figure(figsize=(7, 5))

    for method in res["method"].unique():
        subset = res[res["method"] == method].sort_values("budget")

        plt.errorbar(
            subset["budget"],
            subset[metric_mean],
            yerr=subset[metric_std],
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
    "kendall_tau_std",
    "Kendall's Tau",
    "Ranking Stability vs Evaluation Budget",
    "kendall_tau_plot.png",
    ylim=(0.80, 1.02),
)

plot_metric(
    res,
    "top10_agreement_mean",
    "top10_agreement_std",
    "Top-10 Agreement",
    "Top-10 Agreement vs Evaluation Budget",
    "top10_agreement_plot.png",
    ylim=(0.75, 1.05),
)

plot_metric(
    res,
    "pairwise_consistency_mean",
    "pairwise_consistency_std",
    "Pairwise Consistency",
    "Pairwise Consistency vs Evaluation Budget",
    "pairwise_consistency_plot.png",
    ylim=(0.90, 1.02),
)

print("\nSaved figures:")
print(FIG_DIR / "kendall_tau_plot.png")
print(FIG_DIR / "top10_agreement_plot.png")
print(FIG_DIR / "pairwise_consistency_plot.png")