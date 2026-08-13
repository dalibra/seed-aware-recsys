#!/usr/bin/env python
import argparse
import itertools
import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


ROOT = Path(os.environ.get("NOTE_SIG_ROOT", Path(__file__).resolve().parents[1]))
INPUT_ROOT = Path(os.environ.get("SEED_MATRIX_DIR", ROOT / "outputs" / "seed_matrix"))
ANALYSIS_DIR = Path(os.environ.get("ANALYSIS_DIR", ROOT / "outputs" / "analysis"))
FIG_DIR = Path(os.environ.get("FIG_DIR", ROOT / "outputs" / "figures"))
TABLE_DIR = Path(os.environ.get("TABLE_DIR", ROOT / "outputs" / "tables"))

SPLIT = "leave-one-out-no_cold_items"
SEEDS = [11, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73]
DATASETS = ["Movielens-1m", "Beauty", "Movielens-20m"]
CONFIGS = ["C0", "C1", "C2"]
METRIC = "NDCG@20"
EXTRA_NULL_MODELS = [
    {"label": "GRU4Rec", "path": "RNN", "dataset": "Movielens-20m", "config": "C0"},
]
SUBSAMPLE_SIZES = [500, 1000, 5000, 10000]
NULL_PARTITIONS = 1000
SENS_USER_REPS = 20
SENS_RUN_PARTITIONS = 80

DATASET_LABELS = {
    "Movielens-1m": "ML-1M",
    "Movielens-20m": "ML-20M",
    "Beauty": "Beauty",
}


def fmt(x, digits=3):
    if pd.isna(x):
        return "--"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.{digits}f}"


def stars(p):
    if p < 0.001:
        return "$<.001$"
    return f"{p:.3f}"


def label_dataset(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def dataset_order(dataset):
    return DATASETS.index(dataset) if dataset in DATASETS else len(DATASETS)


def per_user_path(dataset, config, seed, model="SASRec"):
    return (
        INPUT_ROOT
        / dataset
        / SPLIT
        / model
        / config
        / f"test_per_user_seed_{seed}.csv"
    )


def load_metric_matrix(dataset, config, metric=METRIC, model="SASRec", seeds=SEEDS, require=True):
    seed_arrays = []
    user_ids = None
    missing = []
    for seed in seeds:
        path = per_user_path(dataset, config, seed, model=model)
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_csv(path, usecols=["user_id", metric]).sort_values("user_id")
        if user_ids is None:
            user_ids = df["user_id"].to_numpy()
        elif not np.array_equal(user_ids, df["user_id"].to_numpy()):
            raise ValueError(f"user mismatch for {model}/{dataset}/{config}/seed{seed}")
        seed_arrays.append(df[metric].to_numpy(dtype=np.float64))
    if missing:
        if require:
            msg = "\n".join(missing[:20])
            extra = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
            raise FileNotFoundError(f"missing per-user metric files:\n{msg}{extra}")
        return None
    return {"users": user_ids, "values": np.vstack(seed_arrays), "seeds": list(seeds)}


def load_arrays(metric=METRIC):
    arrays = {}
    run_rows = []
    missing = []
    for dataset in DATASETS:
        for config in CONFIGS:
            seed_arrays = []
            user_ids = None
            for seed in SEEDS:
                path = per_user_path(dataset, config, seed)
                if not path.exists():
                    missing.append(str(path))
                    continue
                df = pd.read_csv(path, usecols=["user_id", metric, "HR@20", "MRR@20", "NDCG@10"])
                df = df.sort_values("user_id")
                if user_ids is None:
                    user_ids = df["user_id"].to_numpy()
                elif not np.array_equal(user_ids, df["user_id"].to_numpy()):
                    raise ValueError(f"user mismatch for {dataset}/{config}/seed{seed}")
                values = df[metric].to_numpy(dtype=np.float64)
                seed_arrays.append(values)
                row = {
                    "dataset": dataset,
                    "config": config,
                    "seed": seed,
                    metric: values.mean(),
                    "HR@20": df["HR@20"].mean(),
                    "MRR@20": df["MRR@20"].mean(),
                    "NDCG@10": df["NDCG@10"].mean(),
                    "users": len(values),
                }
                run_rows.append(row)
            if len(seed_arrays) == len(SEEDS):
                arrays[(dataset, config)] = {
                    "users": user_ids,
                    "values": np.vstack(seed_arrays),
                }
    if missing:
        msg = "\n".join(missing[:20])
        extra = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(f"missing per-user metric files:\n{msg}{extra}")
    return arrays, pd.DataFrame(run_rows)


def ttest_user(diff):
    t, p = stats.ttest_1samp(diff, popmean=0.0)
    return float(np.nan_to_num(t)), float(np.nan_to_num(p, nan=1.0)), float(diff.mean())


def wilcoxon_user(diff):
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0 or np.allclose(diff, 0.0):
        return 1.0
    if np.count_nonzero(np.abs(diff) > 1e-12) == 0:
        return 1.0
    try:
        p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="asymptotic").pvalue
    except TypeError:
        p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="approx").pvalue
    return float(np.nan_to_num(p, nan=1.0))


def runlevel_pvalue(deltas):
    deltas = np.asarray(deltas, dtype=np.float64)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size < 2 or np.allclose(deltas, deltas[0]):
        return 1.0
    t, p = stats.ttest_1samp(deltas, popmean=0.0)
    return float(np.nan_to_num(p, nan=1.0))


def disjoint_pair_deltas(values, perm, idx=None):
    if idx is None:
        return np.array([(values[a] - values[b]).mean() for a, b in zip(perm[::2], perm[1::2])], dtype=np.float64)
    return np.array([(values[a, idx] - values[b, idx]).mean() for a, b in zip(perm[::2], perm[1::2])], dtype=np.float64)


def build_null_specs(arrays):
    specs = [
        {
            "model": "SASRec",
            "model_path": "SASRec",
            "dataset": dataset,
            "config": "C0",
            "seeds": SEEDS,
            "values": arrays[(dataset, "C0")]["values"],
        }
        for dataset in DATASETS
    ]
    for spec in EXTRA_NULL_MODELS:
        loaded = load_metric_matrix(
            spec["dataset"],
            spec["config"],
            metric=METRIC,
            model=spec["path"],
            seeds=SEEDS,
            require=False,
        )
        if loaded is None:
            continue
        specs.append(
            {
                "model": spec["label"],
                "model_path": spec["path"],
                "dataset": spec["dataset"],
                "config": spec["config"],
                "seeds": loaded["seeds"],
                "values": loaded["values"],
            }
        )
    return specs


def same_config_null(null_specs):
    rows = []
    partition_rows = []
    run_partition_rows = []
    rng = np.random.default_rng(20260216)
    for spec in null_specs:
        dataset = spec["dataset"]
        model = spec["model"]
        seeds = spec["seeds"]
        values = spec["values"]
        for i, j in itertools.combinations(range(len(seeds)), 2):
            diff = values[i] - values[j]
            _, p, mean = ttest_user(diff)
            p_w = wilcoxon_user(diff)
            nonzero = int(np.count_nonzero(np.abs(diff) > 1e-12))
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "seed_i": seeds[i],
                    "seed_j": seeds[j],
                    "mean_diff": mean,
                    "p_user": p,
                    "p_wilcoxon_user": p_w,
                    "nonzero_users": nonzero,
                    "nonzero_frac": nonzero / values.shape[1],
                    "sign_positive": mean > 0,
                    "reject_005": p < 0.05,
                    "reject_0001": p < 0.001,
                    "wilcoxon_reject_005": p_w < 0.05,
                    "wilcoxon_reject_0001": p_w < 0.001,
                }
            )

        for rep in range(NULL_PARTITIONS):
            perm = rng.permutation(len(seeds))
            rejections = []
            pair_means = []
            paired_perm = perm[: 2 * (len(seeds) // 2)]
            for a, b in zip(paired_perm[::2], paired_perm[1::2]):
                _, p, mean = ttest_user(values[a] - values[b])
                pair_means.append(mean)
                rejections.append(p < 0.05)
                partition_rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "rep": rep,
                        "seed_i": seeds[a],
                        "seed_j": seeds[b],
                        "mean_diff": mean,
                        "p_user": p,
                        "reject_005": p < 0.05,
                    }
                )
            p_run = runlevel_pvalue(pair_means)
            run_partition_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "rep": rep,
                    "p_run": p_run,
                    "reject_005": p_run < 0.05,
                    "n_pairs": len(pair_means),
                }
            )
    pairs = pd.DataFrame(rows)
    partitions = pd.DataFrame(partition_rows)
    run_partitions = pd.DataFrame(run_partition_rows)
    summary = (
        pairs.groupby(["model", "dataset"])
        .agg(
            seed_pairs=("seed_i", "size"),
            reject_005=("reject_005", "mean"),
            reject_0001=("reject_0001", "mean"),
            wilcoxon_reject_005=("wilcoxon_reject_005", "mean"),
            wilcoxon_reject_0001=("wilcoxon_reject_0001", "mean"),
            nonzero_users_med=("nonzero_users", "median"),
            nonzero_frac_med=("nonzero_frac", "median"),
            positive=("sign_positive", "mean"),
            max_abs_diff=("mean_diff", lambda x: np.max(np.abs(x))),
            median_p=("p_user", "median"),
        )
        .reset_index()
    )
    u_lookup = {(spec["model"], spec["dataset"]): spec["values"].shape[1] for spec in null_specs}
    summary["U"] = summary.apply(lambda r: u_lookup[(r["model"], r["dataset"])], axis=1)
    summary = summary[
        [
            "model",
            "dataset",
            "U",
            "seed_pairs",
            "reject_005",
            "reject_0001",
            "wilcoxon_reject_005",
            "wilcoxon_reject_0001",
            "nonzero_users_med",
            "nonzero_frac_med",
            "positive",
            "max_abs_diff",
            "median_p",
        ]
    ]
    part_summary = (
        partitions.groupby(["model", "dataset", "rep"])["reject_005"]
        .mean()
        .groupby(["model", "dataset"])
        .agg(partition_reject_mean="mean", partition_reject_sd="std")
        .reset_index()
    )
    run_part_summary = (
        run_partitions.groupby(["model", "dataset"])
        .agg(runlevel_reject_005=("reject_005", "mean"), runlevel_median_p=("p_run", "median"))
        .reset_index()
    )
    return pairs, partitions, run_partitions, summary.merge(part_summary, on=["model", "dataset"]).merge(run_part_summary, on=["model", "dataset"])


def close_config_verdicts(arrays):
    rows = []
    runlevel = []
    for dataset in DATASETS:
        for a, b in [("C0", "C1"), ("C0", "C2")]:
            va = arrays[(dataset, a)]["values"]
            vb = arrays[(dataset, b)]["values"]
            for i, seed_a in enumerate(SEEDS):
                for j, seed_b in enumerate(SEEDS):
                    _, p, mean = ttest_user(va[i] - vb[j])
                    if p < 0.05 and mean > 0:
                        verdict = "left_wins"
                    elif p < 0.05 and mean < 0:
                        verdict = "right_wins"
                    else:
                        verdict = "ns"
                    rows.append(
                        {
                            "dataset": dataset,
                            "comparison": f"{a}-{b}",
                            "left": a,
                            "right": b,
                            "seed_a": seed_a,
                            "seed_b": seed_b,
                            "mean_diff": mean,
                            "p_user": p,
                            "verdict": verdict,
                        }
                    )
            deltas = (va - vb).mean(axis=1)
            mean_delta = float(deltas.mean())
            sd = float(deltas.std(ddof=1))
            se = sd / math.sqrt(len(deltas))
            tcrit = stats.t.ppf(0.975, df=len(deltas) - 1)
            tstat = mean_delta / se if se > 0 else 0.0
            p_run = 2 * stats.t.sf(abs(tstat), df=len(deltas) - 1) if se > 0 else 1.0
            seed_snr = abs(mean_delta) / sd if sd > 0 else np.inf
            ci_contains_zero = mean_delta - tcrit * se <= 0 <= mean_delta + tcrit * se
            if seed_snr >= 2.5 and not ci_contains_zero:
                regime = "stable-control"
            elif seed_snr < 1.0 and not ci_contains_zero:
                regime = "unstable-resolved"
            elif seed_snr < 1.0 and ci_contains_zero:
                regime = "unstable-indet."
            else:
                regime = "borderline"
            runlevel.append(
                {
                    "dataset": dataset,
                    "comparison": f"{a}-{b}",
                    "mean_delta": mean_delta,
                    "ci_low": mean_delta - tcrit * se,
                    "ci_high": mean_delta + tcrit * se,
                    "p_run": p_run,
                    "sign_stability": float((deltas > 0).mean()),
                    "sd_delta": sd,
                    "seed_snr": seed_snr,
                    "regime": regime,
                    "S": len(deltas),
                }
            )
    loaded_gru = load_metric_matrix("Movielens-20m", "C0", metric=METRIC, model="RNN", seeds=SEEDS, require=False)
    if loaded_gru is not None:
        dataset = "Movielens-20m"
        a = "SASRec"
        b = "GRU4Rec"
        va = arrays[(dataset, "C0")]["values"]
        vb = loaded_gru["values"]
        for i, seed_a in enumerate(SEEDS):
            for j, seed_b in enumerate(SEEDS):
                _, p, mean = ttest_user(va[i] - vb[j])
                if p < 0.05 and mean > 0:
                    verdict = "left_wins"
                elif p < 0.05 and mean < 0:
                    verdict = "right_wins"
                else:
                    verdict = "ns"
                rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{a}-{b}",
                        "left": a,
                        "right": b,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "mean_diff": mean,
                        "p_user": p,
                        "verdict": verdict,
                    }
                )
        deltas = (va - vb).mean(axis=1)
        mean_delta = float(deltas.mean())
        sd = float(deltas.std(ddof=1))
        se = sd / math.sqrt(len(deltas))
        tcrit = stats.t.ppf(0.975, df=len(deltas) - 1)
        tstat = mean_delta / se if se > 0 else 0.0
        p_run = 2 * stats.t.sf(abs(tstat), df=len(deltas) - 1) if se > 0 else 1.0
        seed_snr = abs(mean_delta) / sd if sd > 0 else np.inf
        ci_contains_zero = mean_delta - tcrit * se <= 0 <= mean_delta + tcrit * se
        if seed_snr >= 2.5 and not ci_contains_zero:
            regime = "stable-control"
        elif seed_snr < 1.0 and not ci_contains_zero:
            regime = "unstable-resolved"
        elif seed_snr < 1.0 and ci_contains_zero:
            regime = "unstable-indet."
        else:
            regime = "borderline"
        runlevel.append(
            {
                "dataset": dataset,
                "comparison": f"{a}-{b}",
                "mean_delta": mean_delta,
                "ci_low": mean_delta - tcrit * se,
                "ci_high": mean_delta + tcrit * se,
                "p_run": p_run,
                "sign_stability": float((deltas > 0).mean()),
                "sd_delta": sd,
                "seed_snr": seed_snr,
                "regime": regime,
                "S": len(deltas),
            }
        )
    verdicts = pd.DataFrame(rows)
    verdict_summary = (
        verdicts.assign(
            left_win=lambda df: df["verdict"].eq("left_wins"),
            right_win=lambda df: df["verdict"].eq("right_wins"),
            no_sig=lambda df: df["verdict"].eq("ns"),
        )
        .groupby(["dataset", "comparison", "left", "right"])
        .agg(
            left_wins=("left_win", "mean"),
            right_wins=("right_win", "mean"),
            ns=("no_sig", "mean"),
        )
        .reset_index()
    )
    return verdicts, verdict_summary, pd.DataFrame(runlevel)


def user_sensitivity(arrays):
    rows = []
    run_rows = []
    rng = np.random.default_rng(20260216)
    for dataset in DATASETS:
        users = arrays[(dataset, "C0")]["users"]
        values = arrays[(dataset, "C0")]["values"]
        all_sizes = [u for u in SUBSAMPLE_SIZES if u < len(users)] + [len(users)]
        for n_users in all_sizes:
            reps = 1 if n_users == len(users) else SENS_USER_REPS
            for rep in range(reps):
                if n_users == len(users):
                    idx = np.arange(len(users))
                else:
                    idx = np.sort(rng.choice(len(users), size=n_users, replace=False))
                for i, j in itertools.combinations(range(len(SEEDS)), 2):
                    _, p, mean = ttest_user(values[i, idx] - values[j, idx])
                    rows.append(
                        {
                            "dataset": dataset,
                            "users": n_users,
                            "rep": rep,
                            "seed_i": SEEDS[i],
                            "seed_j": SEEDS[j],
                            "mean_diff": mean,
                            "p_user": p,
                            "reject_005": p < 0.05,
                            "neglog_p_user": -math.log10(max(p, 1e-300)),
                        }
                    )
                for run_rep in range(SENS_RUN_PARTITIONS):
                    perm = rng.permutation(len(SEEDS))[:14]
                    deltas = disjoint_pair_deltas(values, perm, idx)
                    p_run = runlevel_pvalue(deltas)
                    run_rows.append(
                        {
                            "dataset": dataset,
                            "users": n_users,
                            "rep": rep,
                            "run_rep": run_rep,
                            "p_run": p_run,
                            "reject_005": p_run < 0.05,
                            "neglog_p_run": -math.log10(max(p_run, 1e-300)),
                        }
                    )
    sens = pd.DataFrame(rows)
    run_sens = pd.DataFrame(run_rows)
    user_summary = (
        sens.groupby(["dataset", "users"])
        .agg(
            reject_005=("reject_005", "mean"),
            median_p=("p_user", "median"),
            user_neglog_med=("neglog_p_user", "median"),
            user_neglog_q25=("neglog_p_user", lambda x: np.quantile(x, 0.25)),
            user_neglog_q75=("neglog_p_user", lambda x: np.quantile(x, 0.75)),
        )
        .reset_index()
    )
    run_summary = (
        run_sens.groupby(["dataset", "users"])
        .agg(
            runlevel_reject_005=("reject_005", "mean"),
            runlevel_median_p=("p_run", "median"),
            run_neglog_med=("neglog_p_run", "median"),
            run_neglog_q25=("neglog_p_run", lambda x: np.quantile(x, 0.25)),
            run_neglog_q75=("neglog_p_run", lambda x: np.quantile(x, 0.75)),
        )
        .reset_index()
    )
    summary = user_summary.merge(run_summary, on=["dataset", "users"])
    return sens, run_sens, summary


def variance_components(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    s, u = matrix.shape
    grand = matrix.mean()
    seed_means = matrix.mean(axis=1)
    user_means = matrix.mean(axis=0)
    residual = matrix - seed_means[:, None] - user_means[None, :] + grand
    ms_seed = u * np.sum((seed_means - grand) ** 2) / (s - 1)
    ms_user = s * np.sum((user_means - grand) ** 2) / (u - 1)
    ms_resid = np.sum(residual ** 2) / ((s - 1) * (u - 1))
    sigma_a2 = max((ms_seed - ms_resid) / u, 0.0)
    sigma_b2 = max((ms_user - ms_resid) / s, 0.0)
    sigma_eps2 = max(ms_resid, 0.0)
    sigma_id2 = sigma_b2 + sigma_eps2
    rho = sigma_a2 / (sigma_a2 + sigma_id2) if sigma_a2 + sigma_id2 > 0 else 0.0
    vif = 1.0 + u * sigma_a2 / sigma_id2 if sigma_id2 > 0 else np.inf
    u_eff = u / (1.0 + (u - 1) * rho) if rho > 0 else float(u)
    return {
        "U": u,
        "S": s,
        "sigma_a2": sigma_a2,
        "sigma_id2": sigma_id2,
        "rho": rho,
        "VIF": vif,
        "SE_underestimate": math.sqrt(vif),
        "U_eff": u_eff,
    }


def same_procedure_diagnostics(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    s, u = matrix.shape
    grand = matrix.mean()
    seed_means = matrix.mean(axis=1)
    user_means = matrix.mean(axis=0)
    residual = matrix - seed_means[:, None] - user_means[None, :] + grand
    ms_resid = np.sum(residual ** 2) / ((s - 1) * (u - 1))
    seed_mean_var = np.var(seed_means, ddof=1)
    sigma_run2 = max(seed_mean_var - ms_resid / u, 0.0)
    pair_vars = []
    for i, j in itertools.combinations(range(s), 2):
        pair_vars.append(np.var(matrix[i] - matrix[j], ddof=1))
    sigma_id_pair2 = float(np.mean(pair_vars))
    vif_pair = 1.0 + 2.0 * u * sigma_run2 / sigma_id_pair2 if sigma_id_pair2 > 0 else np.inf
    sqrt_vif_pair = math.sqrt(vif_pair)
    rho_pair = (2.0 * sigma_run2) / (2.0 * sigma_run2 + sigma_id_pair2) if sigma_id_pair2 > 0 else 0.0
    u_eff_pair = u / (1.0 + (u - 1.0) * rho_pair) if rho_pair > 0 else float(u)

    alpha = 0.05
    low_total = (s - 1) * seed_mean_var / stats.chi2.ppf(1.0 - alpha / 2.0, df=s - 1)
    high_total = (s - 1) * seed_mean_var / stats.chi2.ppf(alpha / 2.0, df=s - 1)
    sigma_run2_low = max(low_total - ms_resid / u, 0.0)
    sigma_run2_high = max(high_total - ms_resid / u, 0.0)
    sqrt_vif_low = math.sqrt(1.0 + 2.0 * u * sigma_run2_low / sigma_id_pair2) if sigma_id_pair2 > 0 else np.inf
    sqrt_vif_high = math.sqrt(1.0 + 2.0 * u * sigma_run2_high / sigma_id_pair2) if sigma_id_pair2 > 0 else np.inf
    rho_low = (2.0 * sigma_run2_low) / (2.0 * sigma_run2_low + sigma_id_pair2) if sigma_id_pair2 > 0 else 0.0
    rho_high = (2.0 * sigma_run2_high) / (2.0 * sigma_run2_high + sigma_id_pair2) if sigma_id_pair2 > 0 else 0.0

    z005 = stats.norm.ppf(1.0 - 0.05 / 2.0)
    z0001 = stats.norm.ppf(1.0 - 0.001 / 2.0)
    pred_005 = 2.0 * stats.norm.sf(z005 / sqrt_vif_pair) if np.isfinite(sqrt_vif_pair) else 1.0
    pred_0001 = 2.0 * stats.norm.sf(z0001 / sqrt_vif_pair) if np.isfinite(sqrt_vif_pair) else 1.0

    return {
        "U": u,
        "S": s,
        "sigma_run2": sigma_run2,
        "sigma_id_pair2": sigma_id_pair2,
        "rho_pair": rho_pair,
        "rho_low": rho_low,
        "rho_high": rho_high,
        "VIF_pair": vif_pair,
        "sqrt_VIF_pair": sqrt_vif_pair,
        "sqrt_VIF_low": sqrt_vif_low,
        "sqrt_VIF_high": sqrt_vif_high,
        "U_eff_pair": u_eff_pair,
        "pred_reject_005": pred_005,
        "pred_reject_0001": pred_0001,
        "resid_correction": ms_resid / u,
    }


def diagnostics(null_specs, null_summary):
    rows = []
    for spec in null_specs:
        row = {"model": spec["model"], "dataset": spec["dataset"]}
        row.update(same_procedure_diagnostics(spec["values"]))
        rows.append(row)
    diag = pd.DataFrame(rows)
    cols = [
        "model",
        "dataset",
        "reject_005",
        "reject_0001",
        "wilcoxon_reject_005",
        "wilcoxon_reject_0001",
        "nonzero_users_med",
        "nonzero_frac_med",
        "runlevel_reject_005",
        "runlevel_median_p",
    ]
    return diag.merge(null_summary[cols], on=["model", "dataset"])


def metric_null_check(dataset="Movielens-20m", metrics=("HR@20", "MRR@20")):
    rows = []
    rng = np.random.default_rng(20260216)
    for metric in metrics:
        loaded = load_metric_matrix(dataset, "C0", metric=metric, model="SASRec", seeds=SEEDS, require=True)
        values = loaded["values"]
        pair_reject = []
        for i, j in itertools.combinations(range(len(SEEDS)), 2):
            _, p, _ = ttest_user(values[i] - values[j])
            pair_reject.append(p < 0.05)
        run_reject = []
        for _ in range(NULL_PARTITIONS):
            perm = rng.permutation(len(SEEDS))[:14]
            deltas = disjoint_pair_deltas(values, perm)
            run_reject.append(runlevel_pvalue(deltas) < 0.05)
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "user_t_reject_005": float(np.mean(pair_reject)),
                "run_t_reject_005": float(np.mean(run_reject)),
            }
        )
    return pd.DataFrame(rows)


def seed_noise_floor(run_summary, runlevel):
    base = (
        run_summary[(run_summary["config"] == "C0")]
        .groupby("dataset")[METRIC]
        .mean()
        .rename("c0_mean")
        .reset_index()
    )
    noise = (
        runlevel.groupby("dataset")
        .agg(sd_min=("sd_delta", "min"), sd_max=("sd_delta", "max"))
        .reset_index()
        .merge(base, on="dataset")
    )
    noise["rel_min"] = 100.0 * noise["sd_min"] / noise["c0_mean"]
    noise["rel_max"] = 100.0 * noise["sd_max"] / noise["c0_mean"]
    return noise


def leave_one_seed_out_diagnostics(null_specs):
    rows = []
    for spec in null_specs:
        values = spec["values"]
        if values.shape[0] < 4:
            continue
        estimates = []
        for leave_idx, seed in enumerate(spec["seeds"]):
            keep = [idx for idx in range(values.shape[0]) if idx != leave_idx]
            diag = same_procedure_diagnostics(values[keep])
            estimates.append(diag["sqrt_VIF_pair"])
            rows.append(
                {
                    "model": spec["model"],
                    "dataset": spec["dataset"],
                    "left_out_seed": seed,
                    "sqrt_VIF_pair": diag["sqrt_VIF_pair"],
                }
            )
        rows.append(
            {
                "model": spec["model"],
                "dataset": spec["dataset"],
                "left_out_seed": "range",
                "sqrt_VIF_pair": np.nan,
                "sqrt_VIF_min": float(np.min(estimates)),
                "sqrt_VIF_max": float(np.max(estimates)),
            }
        )
    return pd.DataFrame(rows)


def write_tables(null_summary, verdict_summary, runlevel, diag):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    null_for_table = null_summary.merge(
        diag[
            [
                "model",
                "dataset",
                "pred_reject_005",
                "sqrt_VIF_pair",
                "sqrt_VIF_low",
                "sqrt_VIF_high",
            ]
        ],
        on=["model", "dataset"],
    )
    null_for_table["dataset_order"] = null_for_table["dataset"].map(dataset_order)
    null_for_table["model_order"] = null_for_table["model"].map({"SASRec": 0}).fillna(1)
    null_for_table = null_for_table.sort_values(["model_order", "dataset_order", "model"])
    null_rows = []
    for _, r in null_for_table.iterrows():
        null_rows.append(
            [
                r["model"],
                label_dataset(r["dataset"]),
                f"{100*r['reject_005']:.1f}\\%",
                f"{100*r['wilcoxon_reject_005']:.1f}\\%",
                f"{100*r['nonzero_frac_med']:.1f}\\%",
                f"{100*r['runlevel_reject_005']:.1f}\\%",
                f"{100*r['pred_reject_005']:.1f}\\%",
                f"{r['sqrt_VIF_pair']:.2f} [{r['sqrt_VIF_low']:.2f},{r['sqrt_VIF_high']:.2f}]",
            ]
        )
    null_tex = (
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrrrl@{}}\n"
        "\\toprule\n"
        "Model & Data & user $t$ & user W & nz users & run $t$ & VIF pred. & $\\sqrt{\\mathrm{VIF}_{pair}}$ [95\\% CI]\\\\\n"
        "\\midrule\n"
        + "\n".join(" & ".join(row) + "\\\\" for row in null_rows)
        + "\n\\bottomrule\n\\end{tabular*}\n"
    )
    (TABLE_DIR / "null_calibration.tex").write_text(null_tex)
    (TABLE_DIR / "null_diagnostics_compact.tex").write_text(null_tex)

    close = verdict_summary.merge(runlevel, on=["dataset", "comparison"])
    close["dataset_order"] = close["dataset"].map(dataset_order)
    close = close.sort_values(["dataset_order", "comparison"])
    close_rows = []
    for _, r in close.iterrows():
        single = f"{100*r['left_wins']:.0f}/{100*r['right_wins']:.0f}/{100*r['ns']:.0f}\\%"
        close_rows.append(
            [
                label_dataset(r["dataset"]),
                r["comparison"].replace("-", "--"),
                r["regime"],
                f"{r['seed_snr']:.2f}",
                single,
                f"{r['mean_delta']:.5f} [{r['ci_low']:.5f},{r['ci_high']:.5f}]",
                f"{100*r['sign_stability']:.0f}\\%",
            ]
        )
    close_tex = (
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lllrrll@{}}\n"
        "\\toprule\n"
        "Data & Pair & regime & SNR & first/second/n.s. & run-level $\\bar\\Delta$ [95\\% CI] & first seed-win\\\\\n"
        "\\midrule\n"
        + "\n".join(" & ".join(row) + "\\\\" for row in close_rows)
        + "\n\\bottomrule\n\\end{tabular*}\n"
    )
    (TABLE_DIR / "close_pair.tex").write_text(close_tex)
    (TABLE_DIR / "close_pair_compact.tex").write_text(close_tex)

    diag_rows = []
    diag_for_table = diag[diag["model"] == "SASRec"].copy()
    diag_for_table["dataset_order"] = diag_for_table["dataset"].map(dataset_order)
    diag_for_table = diag_for_table.sort_values("dataset_order")
    for _, r in diag_for_table.iterrows():
        diag_rows.append(
            [
                label_dataset(r["dataset"]),
                f"{int(r['U']):,}",
                f"{r['sigma_run2']:.2e}",
                f"{r['rho_pair']:.2e}",
                fmt(r["VIF_pair"], 2),
                fmt(r["sqrt_VIF_pair"], 2),
                fmt(r["U_eff_pair"], 1),
                f"{100*r['reject_005']:.1f}\\%",
            ]
        )
    diag_tex = (
        "\\begin{tabular}{lrrrrrrr}\n"
        "\\toprule\n"
        "Dataset & $U$ & $\\hat\\sigma_{run}^2$ & $\\hat\\rho_{pair}$ & $\\mathrm{VIF}_{pair}$ & $\\sqrt{\\mathrm{VIF}_{pair}}$ & $U_{\\rm eff}$ & null rej.\\\\\n"
        "\\midrule\n"
        + "\n".join(" & ".join(row) + "\\\\" for row in diag_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (TABLE_DIR / "diagnostics.tex").write_text(diag_tex)


def plot_figures(arrays, sens_summary, diag, runlevel):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 6.3,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {"Movielens-1m": "#3b6fb6", "Beauty": "#c45a30", "Movielens-20m": "#2b8a5f"}

    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    for dataset in DATASETS:
        part = sens_summary[sens_summary.dataset == dataset].sort_values("users")
        ax.fill_between(
            part["users"].to_numpy(),
            part["user_neglog_q25"].to_numpy(),
            part["user_neglog_q75"].to_numpy(),
            color=colors[dataset],
            alpha=0.10,
            linewidth=0,
        )
        ax.plot(
            part["users"],
            part["user_neglog_med"],
            marker="o",
            ms=3,
            color=colors[dataset],
            label=f"{label_dataset(dataset)} user",
        )
        ax.plot(
            part["users"],
            part["run_neglog_med"],
            marker="s",
            ms=2.6,
            ls="--",
            color=colors[dataset],
            label=f"{label_dataset(dataset)} run",
        )
    ax.axhline(-math.log10(0.05), color="0.25", lw=0.65, ls=":", label="$p=.05$")
    ax.axhline(-math.log10(0.001), color="0.25", lw=0.65, ls="-.", label="$p=.001$")
    ax.set_xscale("log")
    ax.set_xlabel("subsampled test users")
    ax.set_ylabel("median $-\\log_{10}(p)$")
    ax.set_title("Same-procedure null")
    ax.legend(frameon=False, ncol=2, columnspacing=0.7, handlelength=1.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / "u_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    scatter_specs = [
        ("stable control\nML-1M C0--C2", "Movielens-1m", "C0", "C2"),
        ("unstable-indet.\nML-20M C0--C2", "Movielens-20m", "C0", "C2"),
    ]
    for x, (label, dataset, a, b) in enumerate(scatter_specs):
        diff = arrays[(dataset, a)]["values"].mean(axis=1) - arrays[(dataset, b)]["values"].mean(axis=1)
        jitter = np.linspace(-0.14, 0.14, len(diff))
        ax.scatter(np.full(len(diff), x) + jitter, diff, s=12, color=colors[dataset], alpha=0.9)
        row = runlevel[(runlevel.dataset == dataset) & (runlevel.comparison == f"{a}-{b}")].iloc[0]
        ax.errorbar(
            x,
            row["mean_delta"],
            yerr=[[row["mean_delta"] - row["ci_low"]], [row["ci_high"] - row["mean_delta"]]],
            color="black",
            capsize=2.5,
            marker="s",
            ms=2.7,
            lw=0.9,
        )
    ax.axhline(0, color="0.35", lw=0.75)
    ax.set_xticks(np.arange(len(scatter_specs)), [s[0] for s in scatter_specs])
    ax.set_ylabel("$\\Delta_s$ NDCG@20")
    ax.set_title("Run-level signal vs. seed noise")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / "seed_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    for _, r in diag[diag["model"] == "SASRec"].iterrows():
        max_u = int(r["U"])
        xs = np.unique(np.round(np.geomspace(100, max_u, 80)).astype(int))
        sigma_ratio = (r["VIF_pair"] - 1.0) / max_u
        ax.plot(xs, 1.0 + xs * sigma_ratio, label=label_dataset(r["dataset"]), color=colors[r["dataset"]])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("test users $U$")
    ax.set_ylabel("$\\mathrm{VIF}_{pair}$")
    ax.set_title("Naive variance inflation")
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.25)
    fig.savefig(FIG_DIR / "vif_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 2.35))
    ax = axes[0, 0]
    for dataset in DATASETS:
        part = sens_summary[sens_summary.dataset == dataset].sort_values("users")
        ax.fill_between(
            part["users"].to_numpy(),
            part["user_neglog_q25"].to_numpy(),
            part["user_neglog_q75"].to_numpy(),
            color=colors[dataset],
            alpha=0.10,
            linewidth=0,
        )
        ax.plot(part["users"], part["user_neglog_med"], marker="o", ms=2.5, color=colors[dataset], label=f"{label_dataset(dataset)} user")
        ax.plot(part["users"], part["run_neglog_med"], marker="s", ms=2.3, ls="--", color=colors[dataset], label=f"{label_dataset(dataset)} run")
    ax.axhline(-math.log10(0.05), color="0.25", lw=0.6, ls=":")
    ax.axhline(-math.log10(0.001), color="0.25", lw=0.6, ls="-.")
    ax.set_xscale("log")
    ax.set_xlabel("subsampled test users")
    ax.set_ylabel("median $-\\log_{10}(p)$")
    ax.set_title("(a) More users, smaller user $p$", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=5.6, ncol=2, columnspacing=0.6, handlelength=1.2)

    ax = axes[0, 1]
    for x, (label, dataset, a, b) in enumerate(scatter_specs):
        diff = arrays[(dataset, a)]["values"].mean(axis=1) - arrays[(dataset, b)]["values"].mean(axis=1)
        jitter = np.linspace(-0.13, 0.13, len(diff))
        ax.scatter(np.full(len(diff), x) + jitter, diff, s=10, color=colors[dataset], alpha=0.9)
        row = runlevel[(runlevel.dataset == dataset) & (runlevel.comparison == f"{a}-{b}")].iloc[0]
        ax.errorbar(
            x,
            row["mean_delta"],
            yerr=[[row["mean_delta"] - row["ci_low"]], [row["ci_high"] - row["mean_delta"]]],
            color="black",
            capsize=2.4,
            marker="s",
            ms=2.5,
            lw=0.85,
        )
    ax.axhline(0, color="0.35", lw=0.7)
    ax.set_xticks(np.arange(len(scatter_specs)), ["stable\nML-1M C0--C2", "unstable-indet.\nML-20M C0--C2"])
    ax.set_ylabel("$\\Delta_s$ NDCG@20")
    ax.set_title("(b) Stable vs. unstable-indet.", loc="left", fontweight="bold")

    ax = axes[1, 0]
    for _, r in diag[diag["model"] == "SASRec"].iterrows():
        max_u = int(r["U"])
        xs = np.unique(np.round(np.geomspace(100, max_u, 80)).astype(int))
        sigma_ratio = (r["VIF_pair"] - 1.0) / max_u
        ax.plot(xs, 1.0 + xs * sigma_ratio, label=label_dataset(r["dataset"]), color=colors[r["dataset"]])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("test users $U$")
    ax.set_ylabel("$\\mathrm{VIF}_{pair}$")
    ax.set_title("(c) Pair-null variance inflation", loc="left", fontweight="bold")

    ax = axes[1, 1]
    methods = [("reject_005", "user $t$"), ("wilcoxon_reject_005", "user W"), ("runlevel_reject_005", "run $t$"), ("pred_reject_005", "VIF pred.")]
    x = np.arange(len(DATASETS))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(methods))
    for offset, (col, label) in zip(offsets, methods):
        vals = [100 * diag.loc[(diag.model == "SASRec") & (diag.dataset == dataset), col].iloc[0] for dataset in DATASETS]
        ax.bar(x + offset, vals, width=width, label=label)
    ax.axhline(5, color="0.25", lw=0.7, ls=":")
    ax.set_xticks(x, [label_dataset(d) for d in DATASETS])
    ax.set_ylabel("null reject rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(d) Same-procedure calibration", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=5.8, ncol=2, columnspacing=0.7)

    fig.tight_layout(pad=0.30, h_pad=0.45, w_pad=0.75)
    fig.savefig(FIG_DIR / "diagnostics_grid.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(run_summary, null_summary, verdict_summary, runlevel, diag, metric_robustness, noise_floor, loo_diag):
    def md_table(df, floatfmt=".5g"):
        try:
            return df.to_markdown(index=False, floatfmt=floatfmt)
        except Exception:
            return "```\n" + df.to_string(index=False) + "\n```"

    lines = [
        "# Seed-aware significance experiment summary",
        "",
        "## Artifacts",
        "",
        "- Paper PDF: `paper/_main.pdf` (ACM double-column, 3 pages: 2 pages of note text plus a third page with tables, figures, and references).",
        "- Paper source: `paper/_main.tex`, `paper/sections/`, `paper/tables/`, `paper/figures/`.",
        "- Experiment runner: `experiments/run_seed_matrix.py`.",
        "- GRU4Rec null runner: `experiments/run_rnn_null.py`.",
        "- Analysis script: `experiments/analyze_seed_matrix.py`.",
        "- Raw per-user outputs: `outputs/seed_matrix/`.",
        "- Analysis CSVs: `outputs/analysis/`.",
        "- Run-level null partitions: `outputs/analysis/same_config_runlevel_partitions.csv`.",
        "- User-count run-level sensitivity: `outputs/analysis/u_sensitivity_runlevel.csv`.",
        "- Metric robustness: `outputs/analysis/metric_robustness.csv`.",
        "- Relative seed-noise floor: `outputs/analysis/seed_noise_floor.csv`.",
        "- Leave-one-seed-out VIF diagnostics: `outputs/analysis/leave_one_seed_out_vif.csv`.",
        "- Training logs: `logs/seed_matrix/` and `logs/rnn_null/`.",
        "",
        "## Headline",
        "",
        "The main empirical result matches the theory: the same SASRec configuration compared against itself across different training seeds is rejected by the naive per-user paired t-test far above the nominal 5% rate, while a run-level test over disjoint seed-pair means stays calibrated. On ML-20M, the same-procedure null is rejected in 80.0% of user-level t-test seed-pair comparisons and 4.3% of run-level partitions. The pair-null VIF is 85.1, corresponding to a 9.23x standard-error underestimate.",
        "",
        "GRU4Rec repeats the same-procedure null on ML-20M: user-level t rejects 44.8% of seed-pair comparisons, the run-level test rejects 4.3%, and VIF predicts 41.1%.",
        "",
        f"Metric: `{METRIC}`. Seeds: {len(SEEDS)}. Split: `{SPLIT}`.",
        "",
        "## Run means",
        "",
        md_table(
            run_summary.groupby(["dataset", "config"])[METRIC]
            .agg(["mean", "std", "min", "max"])
            .reset_index(),
            floatfmt=".5f",
        ),
        "",
        "## Same-config null calibration",
        "",
        md_table(null_summary, floatfmt=".5g"),
        "",
        "## Metric robustness",
        "",
        md_table(metric_robustness, floatfmt=".5g"),
        "",
        "## Relative seed-noise floor",
        "",
        md_table(noise_floor, floatfmt=".5g"),
        "",
        "## Leave-one-seed-out VIF",
        "",
        md_table(loo_diag, floatfmt=".5g"),
        "",
        "## Close-config run-level tests",
        "",
        md_table(runlevel, floatfmt=".5g"),
        "",
        "## Variance diagnostics",
        "",
        md_table(diag, floatfmt=".5g"),
        "",
    ]
    (ROOT / "FINAL_SUMMARY.md").write_text("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--summary-path", type=Path, default=ROOT / "FINAL_SUMMARY.md")
    return parser.parse_args()


def main():
    global INPUT_ROOT, ANALYSIS_DIR, FIG_DIR, TABLE_DIR

    args = parse_args()
    INPUT_ROOT = args.input_root
    ANALYSIS_DIR = args.analysis_dir
    FIG_DIR = args.figure_dir
    TABLE_DIR = args.table_dir

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    arrays, run_summary = load_arrays()
    run_summary.to_csv(ANALYSIS_DIR / "run_summary.csv", index=False)
    null_specs = build_null_specs(arrays)

    null_pairs, null_partitions, null_run_partitions, null_summary = same_config_null(null_specs)
    null_pairs.to_csv(ANALYSIS_DIR / "same_config_seed_pairs.csv", index=False)
    null_partitions.to_csv(ANALYSIS_DIR / "same_config_independent_partitions.csv", index=False)
    null_run_partitions.to_csv(ANALYSIS_DIR / "same_config_runlevel_partitions.csv", index=False)
    null_summary.to_csv(ANALYSIS_DIR / "same_config_summary.csv", index=False)

    verdicts, verdict_summary, runlevel = close_config_verdicts(arrays)
    verdicts.to_csv(ANALYSIS_DIR / "close_config_single_run_verdicts.csv", index=False)
    verdict_summary.to_csv(ANALYSIS_DIR / "close_config_verdict_summary.csv", index=False)
    runlevel.to_csv(ANALYSIS_DIR / "close_config_runlevel.csv", index=False)

    sens, run_sens, sens_summary = user_sensitivity(arrays)
    sens.to_csv(ANALYSIS_DIR / "u_sensitivity_pairs.csv", index=False)
    run_sens.to_csv(ANALYSIS_DIR / "u_sensitivity_runlevel.csv", index=False)
    sens_summary.to_csv(ANALYSIS_DIR / "u_sensitivity_summary.csv", index=False)

    diag = diagnostics(null_specs, null_summary)
    diag.to_csv(ANALYSIS_DIR / "variance_diagnostics.csv", index=False)

    metric_robustness = metric_null_check()
    metric_robustness.to_csv(ANALYSIS_DIR / "metric_robustness.csv", index=False)
    noise_floor = seed_noise_floor(run_summary, runlevel)
    noise_floor.to_csv(ANALYSIS_DIR / "seed_noise_floor.csv", index=False)
    loo_diag = leave_one_seed_out_diagnostics(null_specs)
    loo_diag.to_csv(ANALYSIS_DIR / "leave_one_seed_out_vif.csv", index=False)

    write_tables(null_summary, verdict_summary, runlevel, diag)
    plot_figures(arrays, sens_summary, diag, runlevel)
    write_summary(run_summary, null_summary, verdict_summary, runlevel, diag, metric_robustness, noise_floor, loo_diag)
    if args.summary_path != ROOT / "FINAL_SUMMARY.md":
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text((ROOT / "FINAL_SUMMARY.md").read_text())

    print(f"wrote analysis to {ANALYSIS_DIR}")
    print(f"wrote tables to {TABLE_DIR}")
    print(f"wrote figures to {FIG_DIR}")
    print(f"wrote {ROOT / 'FINAL_SUMMARY.md'}")


if __name__ == "__main__":
    main()
