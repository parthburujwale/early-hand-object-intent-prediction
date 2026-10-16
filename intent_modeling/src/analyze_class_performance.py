from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling"
)

EVAL_ROOT = (
    ROOT
    / "outputs"
    / "final_evaluation"
)

OUTPUT_ROOT = (
    EVAL_ROOT
    / "class_analysis"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


MODELS = {
    "rgb_only": "RGB only",
    "position_cross_attention":
        "RGB + hand position",
    "motion_cross_attention":
        "RGB + hand position + velocity",
}

SEEDS = [
    10,
    42,
    100
]

FRACTIONS = [
    10,
    100
]

NUM_CLASSES = 36


# =========================================================
# Load predictions across seeds
# =========================================================

def load_seed_predictions(
    model_name,
    fraction,
    seed
):

    path = (
        EVAL_ROOT
        / model_name
        / f"obs_{fraction:03d}"
        / f"seed_{seed}"
        / "predictions.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    data = np.load(path)

    return {
        "labels": data["labels"],
        "predictions": data["predictions"],
        "top3_predictions":
            data["top3_predictions"],
        "logits": data["logits"],
    }


# =========================================================
# Aggregate confusion matrix
# =========================================================

def aggregate_confusion_matrix(
    model_name,
    fraction
):

    cm_total = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    for seed in SEEDS:

        run_dir = (
            EVAL_ROOT
            / model_name
            / f"obs_{fraction:03d}"
            / f"seed_{seed}"
        )

        cm_path = (
            run_dir
            / "confusion_matrix.csv"
        )

        if not cm_path.exists():
            raise FileNotFoundError(
                f"Missing: {cm_path}"
            )

        cm = np.loadtxt(
            cm_path,
            delimiter=",",
            dtype=int
        )

        cm_total += cm

    return cm_total


# =========================================================
# Row-normalize confusion matrix
# =========================================================

def normalize_confusion_matrix(
    cm
):

    row_sums = cm.sum(
        axis=1,
        keepdims=True
    )

    normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(
            cm,
            dtype=float
        ),
        where=row_sums != 0
    )

    return normalized


# =========================================================
# Plot confusion matrix
# =========================================================

def plot_confusion_matrix(
    cm,
    model_name,
    fraction
):

    normalized = (
        normalize_confusion_matrix(
            cm
        )
    )

    fig = plt.figure(
        figsize=(12, 10)
    )

    ax = fig.add_subplot(111)

    image = ax.imshow(
        normalized,
        interpolation="nearest",
        aspect="auto"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Fraction of true class"
    )

    ax.set_xlabel(
        "Predicted Class",
        fontsize=12
    )

    ax.set_ylabel(
        "True Class",
        fontsize=12
    )

    ax.set_title(
        (
            f"{MODELS[model_name]} "
            f"Confusion Matrix "
            f"({fraction}% observation)"
        ),
        fontsize=14
    )

    ticks = np.arange(
        NUM_CLASSES
    )

    # Display dataset action labels 1-36,
    # while internal class indices remain 0-35.
    labels = [
        str(i + 1)
        for i in range(
            NUM_CLASSES
        )
    ]

    ax.set_xticks(
        ticks
    )

    ax.set_yticks(
        ticks
    )

    ax.set_xticklabels(
        labels,
        fontsize=7,
        rotation=90
    )

    ax.set_yticklabels(
        labels,
        fontsize=7
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / (
            f"confusion_"
            f"{model_name}_"
            f"{fraction:03d}.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# =========================================================
# Per-class accuracy
# =========================================================

def compute_per_class_accuracy(
    model_name,
    fraction
):

    correct = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    total = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    seed_accuracies = {
        class_idx: []
        for class_idx
        in range(NUM_CLASSES)
    }

    for seed in SEEDS:

        data = load_seed_predictions(
            model_name,
            fraction,
            seed
        )

        labels = data["labels"]
        preds = data["predictions"]

        for class_idx in range(
            NUM_CLASSES
        ):

            mask = (
                labels
                == class_idx
            )

            class_total = (
                mask.sum()
            )

            if class_total == 0:
                continue

            class_correct = (
                (
                    preds[mask]
                    == labels[mask]
                )
                .sum()
            )

            acc = (
                class_correct
                / class_total
            )

            seed_accuracies[
                class_idx
            ].append(
                acc
            )

            correct[class_idx] += (
                class_correct
            )

            total[class_idx] += (
                class_total
            )

    rows = []

    for class_idx in range(
        NUM_CLASSES
    ):

        values = (
            seed_accuracies[
                class_idx
            ]
        )

        if len(values) == 0:

            mean_acc = np.nan
            std_acc = np.nan

        else:

            mean_acc = np.mean(
                values
            )

            std_acc = np.std(
                values,
                ddof=1
            ) if len(values) > 1 else 0.0

        rows.append(
            {
                "class_index":
                    class_idx,

                "action_label":
                    class_idx + 1,

                "mean_accuracy":
                    mean_acc,

                "std_accuracy":
                    std_acc,

                "total_examples_across_seeds":
                    int(total[class_idx]),
            }
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# Build cross-model comparison
# =========================================================

def build_comparison(
    fraction
):

    rgb = (
        compute_per_class_accuracy(
            "rgb_only",
            fraction
        )
        .rename(
            columns={
                "mean_accuracy":
                    "rgb_accuracy",
                "std_accuracy":
                    "rgb_std",
            }
        )
    )

    position = (
        compute_per_class_accuracy(
            "position_cross_attention",
            fraction
        )
        .rename(
            columns={
                "mean_accuracy":
                    "position_accuracy",
                "std_accuracy":
                    "position_std",
            }
        )
    )

    motion = (
        compute_per_class_accuracy(
            "motion_cross_attention",
            fraction
        )
        .rename(
            columns={
                "mean_accuracy":
                    "motion_accuracy",
                "std_accuracy":
                    "motion_std",
            }
        )
    )

    comparison = rgb[
        [
            "class_index",
            "action_label",
            "rgb_accuracy",
            "rgb_std",
            "total_examples_across_seeds",
        ]
    ].copy()

    comparison = comparison.merge(
        position[
            [
                "class_index",
                "position_accuracy",
                "position_std",
            ]
        ],
        on="class_index"
    )

    comparison = comparison.merge(
        motion[
            [
                "class_index",
                "motion_accuracy",
                "motion_std",
            ]
        ],
        on="class_index"
    )

    comparison[
        "position_gain_over_rgb"
    ] = (
        comparison[
            "position_accuracy"
        ]
        -
        comparison[
            "rgb_accuracy"
        ]
    )

    comparison[
        "velocity_gain_over_position"
    ] = (
        comparison[
            "motion_accuracy"
        ]
        -
        comparison[
            "position_accuracy"
        ]
    )

    comparison[
        "full_gain_over_rgb"
    ] = (
        comparison[
            "motion_accuracy"
        ]
        -
        comparison[
            "rgb_accuracy"
        ]
    )

    return comparison


# =========================================================
# Plot per-class gain
# =========================================================

def plot_class_gain(
    comparison,
    fraction
):

    valid = comparison.dropna(
        subset=[
            "full_gain_over_rgb"
        ]
    ).copy()

    valid = valid.sort_values(
        "full_gain_over_rgb",
        ascending=False
    )

    x = np.arange(
        len(valid)
    )

    plt.figure(
        figsize=(14, 7)
    )

    plt.bar(
        x,
        valid[
            "full_gain_over_rgb"
        ] * 100
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.xticks(
        x,
        valid[
            "action_label"
        ],
        rotation=90,
        fontsize=8
    )

    plt.xlabel(
        "Action Label",
        fontsize=12
    )

    plt.ylabel(
        (
            "Top-1 Gain of Full Model "
            "over RGB-only "
            "(percentage points)"
        ),
        fontsize=12
    )

    plt.title(
        (
            "Per-Class Benefit of Hand Position "
            f"+ Velocity ({fraction}% observation)"
        ),
        fontsize=14
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / (
            f"class_gain_full_vs_rgb_"
            f"{fraction:03d}.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# =========================================================
# Top confusions
# =========================================================

def find_top_confusions(
    cm,
    top_k=20
):

    cm_copy = cm.copy()

    np.fill_diagonal(
        cm_copy,
        0
    )

    pairs = []

    for true_class in range(
        NUM_CLASSES
    ):

        for pred_class in range(
            NUM_CLASSES
        ):

            count = (
                cm_copy[
                    true_class,
                    pred_class
                ]
            )

            if count <= 0:
                continue

            pairs.append(
                {
                    "true_class_index":
                        true_class,

                    "true_action_label":
                        true_class + 1,

                    "pred_class_index":
                        pred_class,

                    "pred_action_label":
                        pred_class + 1,

                    "count":
                        int(count),
                }
            )

    pairs = sorted(
        pairs,
        key=lambda x: x["count"],
        reverse=True
    )

    return pd.DataFrame(
        pairs[:top_k]
    )


# =========================================================
# Main
# =========================================================

def main():

    all_comparisons = []

    for fraction in FRACTIONS:

        print()
        print("=" * 80)
        print(
            f"ANALYZING {fraction}% OBSERVATION"
        )
        print("=" * 80)

        # ---------------------------------------------
        # Confusion matrices
        # ---------------------------------------------

        for model_name in MODELS:

            cm = (
                aggregate_confusion_matrix(
                    model_name,
                    fraction
                )
            )

            cm_output = (
                plot_confusion_matrix(
                    cm,
                    model_name,
                    fraction
                )
            )

            print(
                "Saved:",
                cm_output
            )

            top_confusions = (
                find_top_confusions(
                    cm,
                    top_k=20
                )
            )

            confusion_csv = (
                OUTPUT_ROOT
                / (
                    f"top_confusions_"
                    f"{model_name}_"
                    f"{fraction:03d}.csv"
                )
            )

            top_confusions.to_csv(
                confusion_csv,
                index=False
            )

        # ---------------------------------------------
        # Class comparison
        # ---------------------------------------------

        comparison = (
            build_comparison(
                fraction
            )
        )

        comparison[
            "fraction"
        ] = fraction

        comparison_path = (
            OUTPUT_ROOT
            / (
                f"per_class_comparison_"
                f"{fraction:03d}.csv"
            )
        )

        comparison.to_csv(
            comparison_path,
            index=False
        )

        all_comparisons.append(
            comparison
        )

        plot_path = (
            plot_class_gain(
                comparison,
                fraction
            )
        )

        print(
            "Saved:",
            plot_path
        )

        # ---------------------------------------------
        # Print largest improvements
        # ---------------------------------------------

        ranked = (
            comparison
            .dropna(
                subset=[
                    "full_gain_over_rgb"
                ]
            )
            .sort_values(
                "full_gain_over_rgb",
                ascending=False
            )
        )

        print()
        print(
            "TOP 10 CLASSES: "
            "FULL MODEL vs RGB"
        )

        print(
            ranked[
                [
                    "action_label",
                    "rgb_accuracy",
                    "position_accuracy",
                    "motion_accuracy",
                    "full_gain_over_rgb",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

        print()
        print(
            "TOP 10 CLASSES: "
            "VELOCITY GAIN"
        )

        ranked_velocity = (
            comparison
            .dropna(
                subset=[
                    "velocity_gain_over_position"
                ]
            )
            .sort_values(
                "velocity_gain_over_position",
                ascending=False
            )
        )

        print(
            ranked_velocity[
                [
                    "action_label",
                    "position_accuracy",
                    "motion_accuracy",
                    "velocity_gain_over_position",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    combined = pd.concat(
        all_comparisons,
        ignore_index=True
    )

    combined_path = (
        OUTPUT_ROOT
        / "all_per_class_comparisons.csv"
    )

    combined.to_csv(
        combined_path,
        index=False
    )

    print()
    print("=" * 80)
    print("CLASS ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        "Outputs:",
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()