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

OUTPUT_ROOT = (
    EVAL_ROOT
    / "semantic_analysis"
    / "confusion_reduction"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# H2O action definitions
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


NUM_CLASSES = 36

SEEDS = [
    10,
    42,
    100
]


# ============================================================
# Load aggregate confusion matrix
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

        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}"
            )

        cm = np.loadtxt(
            path,
            delimiter=",",
            dtype=int
        )

        total += cm

    return total


# ============================================================
# Build comparison table
# ============================================================

def build_reduction_table(
    fraction
):

    rgb = aggregate_confusion(
        "rgb_only",
        fraction
    )

    position = aggregate_confusion(
        "position_cross_attention",
        fraction
    )

    motion = aggregate_confusion(
        "motion_cross_attention",
        fraction
    )

    rows = []

    for true_idx in range(
        NUM_CLASSES
    ):

        for pred_idx in range(
            NUM_CLASSES
        ):

            # Ignore correct classifications
            if true_idx == pred_idx:
                continue

            rgb_count = int(
                rgb[
                    true_idx,
                    pred_idx
                ]
            )

            position_count = int(
                position[
                    true_idx,
                    pred_idx
                ]
            )

            motion_count = int(
                motion[
                    true_idx,
                    pred_idx
                ]
            )

            # Skip pairs that never occurred
            if (
                rgb_count == 0
                and position_count == 0
                and motion_count == 0
            ):
                continue

            true_verb = (
                VERBS[
                    true_idx
                ]
            )

            true_object = (
                OBJECTS[
                    true_idx
                ]
            )

            pred_verb = (
                VERBS[
                    pred_idx
                ]
            )

            pred_object = (
                OBJECTS[
                    pred_idx
                ]
            )

            same_object = (
                true_object
                == pred_object
            )

            same_verb = (
                true_verb
                == pred_verb
            )

            rows.append(
                {
                    "true_action_label":
                        true_idx + 1,

                    "true_action":
                        ACTION_NAMES[
                            true_idx
                        ],

                    "pred_action_label":
                        pred_idx + 1,

                    "predicted_action":
                        ACTION_NAMES[
                            pred_idx
                        ],

                    "true_verb":
                        true_verb,

                    "predicted_verb":
                        pred_verb,

                    "true_object":
                        true_object,

                    "predicted_object":
                        pred_object,

                    "same_object":
                        same_object,

                    "same_verb":
                        same_verb,

                    "rgb_errors":
                        rgb_count,

                    "position_errors":
                        position_count,

                    "motion_errors":
                        motion_count,

                    "reduction_rgb_to_position":
                        (
                            rgb_count
                            - position_count
                        ),

                    "reduction_position_to_motion":
                        (
                            position_count
                            - motion_count
                        ),

                    "reduction_rgb_to_motion":
                        (
                            rgb_count
                            - motion_count
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Print top reductions
# ============================================================

def print_top_reductions(
    df,
    fraction
):

    print()
    print("=" * 100)
    print(
        f"{fraction}% OBSERVATION"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Velocity-specific reductions
    # --------------------------------------------------------

    velocity = (
        df
        .sort_values(
            "reduction_position_to_motion",
            ascending=False
        )
    )

    print()
    print(
        "TOP CONFUSIONS REDUCED BY ADDING VELOCITY"
    )
    print("-" * 100)

    print(
        velocity[
            [
                "true_action",
                "predicted_action",
                "position_errors",
                "motion_errors",
                "reduction_position_to_motion",
                "same_object",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Overall RGB -> full model
    # --------------------------------------------------------

    full = (
        df
        .sort_values(
            "reduction_rgb_to_motion",
            ascending=False
        )
    )

    print()
    print(
        "TOP CONFUSIONS REDUCED: RGB -> FULL MODEL"
    )
    print("-" * 100)

    print(
        full[
            [
                "true_action",
                "predicted_action",
                "rgb_errors",
                "motion_errors",
                "reduction_rgb_to_motion",
                "same_object",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Same-object wrong-verb errors only
    # --------------------------------------------------------

    semantic = df[
        (
            df["same_object"]
            == True
        )
        &
        (
            df["same_verb"]
            == False
        )
    ].copy()

    semantic = (
        semantic
        .sort_values(
            "reduction_rgb_to_motion",
            ascending=False
        )
    )

    print()
    print(
        "SAME OBJECT / WRONG VERB CONFUSIONS "
        "REDUCED BY FULL MODEL"
    )
    print("-" * 100)

    print(
        semantic[
            [
                "true_action",
                "predicted_action",
                "rgb_errors",
                "position_errors",
                "motion_errors",
                "reduction_rgb_to_motion",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# Plot strongest velocity reductions
# ============================================================

def plot_velocity_reductions(
    df,
    fraction
):

    velocity = df[
        df[
            "reduction_position_to_motion"
        ] > 0
    ].copy()

    velocity = (
        velocity
        .sort_values(
            "reduction_position_to_motion",
            ascending=False
        )
        .head(15)
    )

    if len(velocity) == 0:
        return

    labels = [
        (
            f"{row.true_action}"
            f" → "
            f"{row.predicted_action}"
        )
        for row
        in velocity.itertuples()
    ]

    values = (
        velocity[
            "reduction_position_to_motion"
        ].values
    )

    y = np.arange(
        len(velocity)
    )

    plt.figure(
        figsize=(11, 7)
    )

    plt.barh(
        y,
        values
    )

    plt.yticks(
        y,
        labels
    )

    plt.xlabel(
        "Fewer Errors After Adding Velocity "
        "(aggregated over 3 seeds)"
    )

    plt.title(
        (
            "Confusions Resolved by Hand Velocity "
            f"at {fraction}% Observation"
        )
    )

    plt.gca().invert_yaxis()

    plt.grid(
        axis="x",
        alpha=0.25
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / (
            f"velocity_confusion_reduction_"
            f"{fraction:03d}.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Saved:",
        output_path
    )


# ============================================================
# Summarize semantic error counts
# ============================================================

def semantic_error_summary(
    fraction
):

    rows = []

    for model_name in [
        "rgb_only",
        "position_cross_attention",
        "motion_cross_attention",
    ]:

        cm = aggregate_confusion(
            model_name,
            fraction
        )

        same_object_wrong_verb = 0

        same_verb_wrong_object = 0

        both_wrong = 0

        total_errors = 0

        for true_idx in range(
            NUM_CLASSES
        ):

            for pred_idx in range(
                NUM_CLASSES
            ):

                if true_idx == pred_idx:
                    continue

                count = int(
                    cm[
                        true_idx,
                        pred_idx
                    ]
                )

                total_errors += count

                same_object = (
                    OBJECTS[
                        true_idx
                    ]
                    ==
                    OBJECTS[
                        pred_idx
                    ]
                )

                same_verb = (
                    VERBS[
                        true_idx
                    ]
                    ==
                    VERBS[
                        pred_idx
                    ]
                )

                if (
                    same_object
                    and not same_verb
                ):

                    same_object_wrong_verb += (
                        count
                    )

                elif (
                    same_verb
                    and not same_object
                ):

                    same_verb_wrong_object += (
                        count
                    )

                elif (
                    not same_object
                    and not same_verb
                ):

                    both_wrong += count

        rows.append(
            {
                "model":
                    model_name,

                "fraction":
                    fraction,

                "total_errors":
                    total_errors,

                "same_object_wrong_verb":
                    same_object_wrong_verb,

                "same_verb_wrong_object":
                    same_verb_wrong_object,

                "both_wrong":
                    both_wrong,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    all_reduction_tables = []

    all_error_summaries = []

    for fraction in [
        10,
        100
    ]:

        table = build_reduction_table(
            fraction
        )

        table[
            "fraction"
        ] = fraction

        output_csv = (
            OUTPUT_ROOT
            / (
                f"confusion_reduction_"
                f"{fraction:03d}.csv"
            )
        )

        table.to_csv(
            output_csv,
            index=False
        )

        print_top_reductions(
            table,
            fraction
        )

        plot_velocity_reductions(
            table,
            fraction
        )

        all_reduction_tables.append(
            table
        )

        error_summary = (
            semantic_error_summary(
                fraction
            )
        )

        all_error_summaries.append(
            error_summary
        )

    combined = pd.concat(
        all_reduction_tables,
        ignore_index=True
    )

    combined.to_csv(
        OUTPUT_ROOT
        / "all_confusion_reductions.csv",
        index=False
    )

    error_summary = pd.concat(
        all_error_summaries,
        ignore_index=True
    )

    error_summary.to_csv(
        OUTPUT_ROOT
        / "semantic_error_counts.csv",
        index=False
    )

    print()
    print("=" * 100)
    print(
        "SEMANTIC ERROR COUNTS"
    )
    print("=" * 100)

    print(
        error_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print(
        "CONFUSION REDUCTION ANALYSIS COMPLETE"
    )
    print("=" * 100)

    print(
        "Outputs:",
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()