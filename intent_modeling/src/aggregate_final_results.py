from pathlib import Path

import pandas as pd


ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling"
)

EVAL_ROOT = (
    ROOT
    / "outputs"
    / "final_evaluation"
)

OUTPUT_PATH = (
    EVAL_ROOT
    / "master_evaluation_metrics.csv"
)


MODEL_NAMES = [
    "rgb_only",
    "hand_motion_only",
    "position_cross_attention",
    "motion_cross_attention",
]


rows = []


for model_name in MODEL_NAMES:

    model_dir = EVAL_ROOT / model_name

    if not model_dir.exists():

        print(
            f"WARNING: missing directory: {model_dir}"
        )

        continue

    for metrics_file in sorted(
        model_dir.glob(
            "obs_*/seed_*/metrics.csv"
        )
    ):

        df = pd.read_csv(
            metrics_file
        )

        rows.append(
            df
        )


if not rows:

    raise RuntimeError(
        "No metrics.csv files found."
    )


all_results = pd.concat(
    rows,
    ignore_index=True
)


all_results = all_results.sort_values(
    [
        "model",
        "fraction",
        "seed"
    ]
)


all_results.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print("=" * 80)
print("MASTER RESULTS CREATED")
print("=" * 80)

print(
    f"Rows: {len(all_results)}"
)

print(
    f"Saved to: {OUTPUT_PATH}"
)

print()

print(
    all_results.groupby(
        "model"
    ).size()
)

print()


# =========================================================
# Aggregate across seeds
# =========================================================

aggregated = (
    all_results
    .groupby(
        [
            "model",
            "fraction"
        ]
    )
    .agg(
        top1_mean=(
            "top1_accuracy",
            "mean"
        ),
        top1_std=(
            "top1_accuracy",
            "std"
        ),
        top3_mean=(
            "top3_accuracy",
            "mean"
        ),
        top3_std=(
            "top3_accuracy",
            "std"
        ),
        macro_f1_mean=(
            "macro_f1",
            "mean"
        ),
        macro_f1_std=(
            "macro_f1",
            "std"
        ),
        macro_precision_mean=(
            "macro_precision",
            "mean"
        ),
        macro_precision_std=(
            "macro_precision",
            "std"
        ),
        macro_recall_mean=(
            "macro_recall",
            "mean"
        ),
        macro_recall_std=(
            "macro_recall",
            "std"
        ),
    )
    .reset_index()
)


aggregate_path = (
    EVAL_ROOT
    / "aggregated_evaluation_metrics.csv"
)


aggregated.to_csv(
    aggregate_path,
    index=False
)


print(
    f"Aggregated results saved to: "
    f"{aggregate_path}"
)

print()
print("=" * 80)
print("AGGREGATED RESULTS")
print("=" * 80)

print(
    aggregated.to_string(
        index=False
    )
)