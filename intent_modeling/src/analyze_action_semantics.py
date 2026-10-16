from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling"
)

EVAL_ROOT = (
    ROOT
    / "outputs"
    / "final_evaluation"
)

CLASS_ANALYSIS_ROOT = (
    EVAL_ROOT
    / "class_analysis"
)

OUTPUT_ROOT = (
    EVAL_ROOT
    / "semantic_analysis"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# H2O action mapping
#
# IMPORTANT:
# Model class index 0 corresponds to H2O action label 1.
# Model class index 35 corresponds to H2O action label 36.
# ============================================================

ACTIONS = [
    ("grab", "book"),
    ("grab", "espresso"),
    ("grab", "lotion"),
    ("grab", "spray"),
    ("grab", "milk"),
    ("grab", "cocoa"),
    ("grab", "chips"),
    ("grab", "cappuccino"),

    ("place", "book"),
    ("place", "espresso"),
    ("place", "lotion"),
    ("place", "spray"),
    ("place", "milk"),
    ("place", "cocoa"),
    ("place", "chips"),
    ("place", "cappuccino"),

    ("open", "lotion"),
    ("open", "milk"),
    ("open", "chips"),

    ("close", "lotion"),
    ("close", "milk"),
    ("close", "chips"),

    ("pour", "milk"),

    ("take out", "espresso"),
    ("take out", "cocoa"),
    ("take out", "chips"),
    ("take out", "cappuccino"),

    ("put in", "espresso"),
    ("put in", "cocoa"),
    ("put in", "cappuccino"),

    ("apply", "lotion"),
    ("apply", "spray"),

    ("read", "book"),
    ("read", "espresso"),

    ("spray", "spray"),

    ("squeeze", "lotion"),
]


if len(ACTIONS) != 36:
    raise RuntimeError(
        f"Expected 36 H2O actions, got {len(ACTIONS)}"
    )


ACTION_NAMES = [
    f"{verb} {obj}"
    for verb, obj in ACTIONS
]


VERBS = [
    verb
    for verb, _ in ACTIONS
]

OBJECTS = [
    obj
    for _, obj in ACTIONS
]


# ============================================================
# Experiment setup
# ============================================================

MODELS = [
    "rgb_only",
    "hand_motion_only",
    "position_cross_attention",
    "motion_cross_attention",
]

MODEL_LABELS = {
    "rgb_only":
        "RGB only",

    "hand_motion_only":
        "Hand motion only",

    "position_cross_attention":
        "RGB + hand position",

    "motion_cross_attention":
        "RGB + hand position + velocity",
}

FRACTIONS = [
    10,
    20,
    30,
    50,
    75,
    100,
]

SEEDS = [
    10,
    42,
    100,
]

NUM_CLASSES = 36


# ============================================================
# Helpers
# ============================================================

def load_predictions(
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
            f"Missing predictions: {path}"
        )

    data = np.load(path)

    return (
        data["labels"].astype(int),
        data["predictions"].astype(int),
    )


def action_to_verb(
    class_indices
):

    return np.array(
        [
            VERBS[int(idx)]
            for idx in class_indices
        ]
    )


def action_to_object(
    class_indices
):

    return np.array(
        [
            OBJECTS[int(idx)]
            for idx in class_indices
        ]
    )


# ============================================================
# Semantic evaluation for one run
# ============================================================

def evaluate_semantics(
    labels,
    predictions
):

    true_verbs = action_to_verb(
        labels
    )

    pred_verbs = action_to_verb(
        predictions
    )

    true_objects = action_to_object(
        labels
    )

    pred_objects = action_to_object(
        predictions
    )

    action_correct = (
        labels == predictions
    )

    verb_correct = (
        true_verbs == pred_verbs
    )

    object_correct = (
        true_objects == pred_objects
    )

    exact_action = (
        verb_correct
        & object_correct
    )

    correct_object_wrong_verb = (
        object_correct
        & ~verb_correct
    )

    correct_verb_wrong_object = (
        verb_correct
        & ~object_correct
    )

    both_wrong = (
        ~verb_correct
        & ~object_correct
    )

    n = len(labels)

    # sanity check:
    # these four categories must partition all predictions
    partition_count = (
        exact_action.sum()
        + correct_object_wrong_verb.sum()
        + correct_verb_wrong_object.sum()
        + both_wrong.sum()
    )

    if partition_count != n:
        raise RuntimeError(
            "Semantic error categories "
            "do not partition all samples."
        )

    return {
        "action_accuracy":
            action_correct.mean(),

        "verb_accuracy":
            verb_correct.mean(),

        "object_accuracy":
            object_correct.mean(),

        "exact_action_rate":
            exact_action.mean(),

        "correct_object_wrong_verb_rate":
            correct_object_wrong_verb.mean(),

        "correct_verb_wrong_object_rate":
            correct_verb_wrong_object.mean(),

        "both_wrong_rate":
            both_wrong.mean(),
    }


# ============================================================
# Run semantic analysis for every checkpoint
# ============================================================

def create_semantic_metrics():

    rows = []

    for model_name in MODELS:

        for fraction in FRACTIONS:

            for seed in SEEDS:

                labels, predictions = (
                    load_predictions(
                        model_name,
                        fraction,
                        seed
                    )
                )

                metrics = (
                    evaluate_semantics(
                        labels,
                        predictions
                    )
                )

                row = {
                    "model":
                        model_name,

                    "fraction":
                        fraction,

                    "seed":
                        seed,
                }

                row.update(
                    metrics
                )

                rows.append(
                    row
                )

    df = pd.DataFrame(
        rows
    )

    output_path = (
        OUTPUT_ROOT
        / "semantic_metrics_per_run.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    return df


# ============================================================
# Aggregate across seeds
# ============================================================

def aggregate_semantic_metrics(
    df
):

    metric_columns = [
        "action_accuracy",
        "verb_accuracy",
        "object_accuracy",
        "exact_action_rate",
        "correct_object_wrong_verb_rate",
        "correct_verb_wrong_object_rate",
        "both_wrong_rate",
    ]

    agg_dict = {}

    for metric in metric_columns:

        agg_dict[
            f"{metric}_mean"
        ] = (
            metric,
            "mean"
        )

        agg_dict[
            f"{metric}_std"
        ] = (
            metric,
            "std"
        )

    aggregated = (
        df
        .groupby(
            [
                "model",
                "fraction",
            ]
        )
        .agg(
            **agg_dict
        )
        .reset_index()
    )

    output_path = (
        OUTPUT_ROOT
        / "semantic_metrics_aggregated.csv"
    )

    aggregated.to_csv(
        output_path,
        index=False
    )

    return aggregated


# ============================================================
# Add action names to existing class comparisons
# ============================================================

def add_names_to_class_analysis():

    for fraction in [
        10,
        100
    ]:

        input_path = (
            CLASS_ANALYSIS_ROOT
            / (
                f"per_class_comparison_"
                f"{fraction:03d}.csv"
            )
        )

        if not input_path.exists():

            print(
                "Skipping missing:",
                input_path
            )

            continue

        df = pd.read_csv(
            input_path
        )

        df["action_name"] = (
            df["class_index"]
            .apply(
                lambda x:
                    ACTION_NAMES[
                        int(x)
                    ]
            )
        )

        df["verb"] = (
            df["class_index"]
            .apply(
                lambda x:
                    VERBS[
                        int(x)
                    ]
            )
        )

        df["object"] = (
            df["class_index"]
            .apply(
                lambda x:
                    OBJECTS[
                        int(x)
                    ]
            )
        )

        # Put names near the front
        columns = [
            "class_index",
            "action_label",
            "action_name",
            "verb",
            "object",
        ]

        remaining = [
            column
            for column in df.columns
            if column not in columns
        ]

        df = df[
            columns + remaining
        ]

        output_path = (
            OUTPUT_ROOT
            / (
                f"per_action_named_"
                f"{fraction:03d}.csv"
            )
        )

        df.to_csv(
            output_path,
            index=False
        )

        print(
            "Saved:",
            output_path
        )


# ============================================================
# Aggregate confusion matrix across 3 seeds
# ============================================================

def aggregate_confusion(
    model_name,
    fraction
):

    total = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=int
    )

    for seed in SEEDS:

        path = (
            EVAL_ROOT
            / model_name
            / f"obs_{fraction:03d}"
            / f"seed_{seed}"
            / "confusion_matrix.csv"
        )

        cm = np.loadtxt(
            path,
            delimiter=",",
            dtype=int
        )

        total += cm

    return total


# ============================================================
# Named confusion matrices
# ============================================================

def make_named_confusion_matrices():

    models = [
        "rgb_only",
        "position_cross_attention",
        "motion_cross_attention",
    ]

    fractions = [
        10,
        100,
    ]

    for fraction in fractions:

        for model_name in models:

            cm = aggregate_confusion(
                model_name,
                fraction
            )

            row_sums = cm.sum(
                axis=1,
                keepdims=True
            )

            cm_normalized = np.divide(
                cm,
                row_sums,
                out=np.zeros_like(
                    cm,
                    dtype=float
                ),
                where=row_sums != 0
            )

            fig = plt.figure(
                figsize=(16, 14)
            )

            ax = fig.add_subplot(
                111
            )

            image = ax.imshow(
                cm_normalized,
                interpolation="nearest",
                aspect="auto"
            )

            fig.colorbar(
                image,
                ax=ax,
                label="Fraction of true class"
            )

            ticks = np.arange(
                NUM_CLASSES
            )

            ax.set_xticks(
                ticks
            )

            ax.set_yticks(
                ticks
            )

            ax.set_xticklabels(
                ACTION_NAMES,
                rotation=90,
                fontsize=7
            )

            ax.set_yticklabels(
                ACTION_NAMES,
                fontsize=7
            )

            ax.set_xlabel(
                "Predicted Action",
                fontsize=12
            )

            ax.set_ylabel(
                "True Action",
                fontsize=12
            )

            ax.set_title(
                (
                    f"{MODEL_LABELS[model_name]} "
                    f"— {fraction}% Observation"
                ),
                fontsize=14
            )

            plt.tight_layout()

            output_path = (
                OUTPUT_ROOT
                / (
                    f"named_confusion_"
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

            print(
                "Saved:",
                output_path
            )


# ============================================================
# Plot semantic metric curves
# ============================================================

def plot_metric_curve(
    aggregated,
    metric,
    ylabel,
    title,
    filename
):

    plt.figure(
        figsize=(9, 6)
    )

    for model_name in MODELS:

        model_df = (
            aggregated[
                aggregated["model"]
                == model_name
            ]
            .sort_values(
                "fraction"
            )
        )

        mean_column = (
            f"{metric}_mean"
        )

        std_column = (
            f"{metric}_std"
        )

        plt.errorbar(
            model_df["fraction"],
            model_df[mean_column]
            * 100,
            yerr=(
                model_df[std_column]
                * 100
            ),
            marker="o",
            linewidth=2,
            capsize=4,
            label=MODEL_LABELS[
                model_name
            ]
        )

    plt.xlabel(
        "Observed Portion of Action (%)"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.xticks(
        FRACTIONS
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# Plot semantic error decomposition
# ============================================================

def plot_error_decomposition(
    aggregated,
    fraction
):

    subset = (
        aggregated[
            aggregated["fraction"]
            == fraction
        ]
        .set_index(
            "model"
        )
        .loc[
            MODELS
        ]
    )

    exact = (
        subset[
            "exact_action_rate_mean"
        ].values
        * 100
    )

    object_only = (
        subset[
            "correct_object_wrong_verb_rate_mean"
        ].values
        * 100
    )

    verb_only = (
        subset[
            "correct_verb_wrong_object_rate_mean"
        ].values
        * 100
    )

    both_wrong = (
        subset[
            "both_wrong_rate_mean"
        ].values
        * 100
    )

    x = np.arange(
        len(MODELS)
    )

    plt.figure(
        figsize=(11, 7)
    )

    plt.bar(
        x,
        exact,
        label="Correct verb + object"
    )

    plt.bar(
        x,
        object_only,
        bottom=exact,
        label="Correct object, wrong verb"
    )

    plt.bar(
        x,
        verb_only,
        bottom=(
            exact
            + object_only
        ),
        label="Correct verb, wrong object"
    )

    plt.bar(
        x,
        both_wrong,
        bottom=(
            exact
            + object_only
            + verb_only
        ),
        label="Both wrong"
    )

    plt.xticks(
        x,
        [
            MODEL_LABELS[
                model
            ]
            for model in MODELS
        ],
        rotation=15
    )

    plt.ylabel(
        "Predictions (%)"
    )

    plt.title(
        (
            "Semantic Error Decomposition "
            f"at {fraction}% Observation"
        )
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / (
            f"semantic_error_decomposition_"
            f"{fraction:03d}.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# Rank action-level improvements with names
# ============================================================

def print_named_top_gains(
    fraction
):

    path = (
        OUTPUT_ROOT
        / (
            f"per_action_named_"
            f"{fraction:03d}.csv"
        )
    )

    if not path.exists():
        return

    df = pd.read_csv(
        path
    )

    print()
    print("=" * 80)

    print(
        f"TOP FULL-MODEL GAINS AT "
        f"{fraction}% OBSERVATION"
    )

    print("=" * 80)

    columns = [
        "action_label",
        "action_name",
        "rgb_accuracy",
        "position_accuracy",
        "motion_accuracy",
        "full_gain_over_rgb",
    ]

    ranked = (
        df
        .sort_values(
            "full_gain_over_rgb",
            ascending=False
        )
        [
            columns
        ]
        .head(15)
        .copy()
    )

    ranked[
        "rgb_accuracy"
    ] *= 100

    ranked[
        "position_accuracy"
    ] *= 100

    ranked[
        "motion_accuracy"
    ] *= 100

    ranked[
        "full_gain_over_rgb"
    ] *= 100

    print(
        ranked.to_string(
            index=False
        )
    )

    print()
    print(
        "TOP VELOCITY-SENSITIVE ACTIONS"
    )

    print("-" * 80)

    columns_velocity = [
        "action_label",
        "action_name",
        "position_accuracy",
        "motion_accuracy",
        "velocity_gain_over_position",
    ]

    ranked_velocity = (
        df
        .sort_values(
            "velocity_gain_over_position",
            ascending=False
        )
        [
            columns_velocity
        ]
        .head(15)
        .copy()
    )

    ranked_velocity[
        "position_accuracy"
    ] *= 100

    ranked_velocity[
        "motion_accuracy"
    ] *= 100

    ranked_velocity[
        "velocity_gain_over_position"
    ] *= 100

    print(
        ranked_velocity.to_string(
            index=False
        )
    )


# ============================================================
# Print high-level semantic table
# ============================================================

def print_semantic_summary(
    aggregated
):

    print()
    print("=" * 100)
    print(
        "SEMANTIC ACCURACY SUMMARY"
    )
    print("=" * 100)

    for fraction in [
        10,
        20,
        30,
        50,
        75,
        100
    ]:

        print()
        print(
            f"{fraction}% OBSERVATION"
        )

        print("-" * 100)

        subset = (
            aggregated[
                aggregated["fraction"]
                == fraction
            ]
            .copy()
        )

        subset[
            "model_name"
        ] = subset[
            "model"
        ].map(
            MODEL_LABELS
        )

        table = subset[
            [
                "model_name",
                "action_accuracy_mean",
                "verb_accuracy_mean",
                "object_accuracy_mean",
                "correct_object_wrong_verb_rate_mean",
                "correct_verb_wrong_object_rate_mean",
                "both_wrong_rate_mean",
            ]
        ].copy()

        for column in table.columns:

            if column == "model_name":
                continue

            table[
                column
            ] *= 100

        print(
            table.to_string(
                index=False
            )
        )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "H2O ACTION SEMANTIC ANALYSIS"
    )
    print("=" * 80)

    print()
    print(
        "Creating semantic metrics..."
    )

    per_run = (
        create_semantic_metrics()
    )

    aggregated = (
        aggregate_semantic_metrics(
            per_run
        )
    )

    print(
        "Created 72 semantic evaluations."
    )

    print()
    print(
        "Adding action names to "
        "existing per-class results..."
    )

    add_names_to_class_analysis()

    print()
    print(
        "Creating named confusion matrices..."
    )

    make_named_confusion_matrices()

    print()
    print(
        "Creating semantic metric curves..."
    )

    plot_metric_curve(
        aggregated,
        metric="action_accuracy",
        ylabel="Action Accuracy (%)",
        title="Action Recognition Across Observation Fractions",
        filename="action_accuracy_curve.png",
    )

    plot_metric_curve(
        aggregated,
        metric="verb_accuracy",
        ylabel="Verb Accuracy (%)",
        title="Verb Recognition Across Observation Fractions",
        filename="verb_accuracy_curve.png",
    )

    plot_metric_curve(
        aggregated,
        metric="object_accuracy",
        ylabel="Object Accuracy (%)",
        title="Object Recognition Across Observation Fractions",
        filename="object_accuracy_curve.png",
    )

    print()
    print(
        "Creating semantic error "
        "decomposition figures..."
    )

    plot_error_decomposition(
        aggregated,
        fraction=10
    )

    plot_error_decomposition(
        aggregated,
        fraction=100
    )

    print_semantic_summary(
        aggregated
    )

    print_named_top_gains(
        10
    )

    print_named_top_gains(
        100
    )

    print()
    print("=" * 80)
    print(
        "SEMANTIC ANALYSIS COMPLETE"
    )
    print("=" * 80)

    print(
        "Outputs:",
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()