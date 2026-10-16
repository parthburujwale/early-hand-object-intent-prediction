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

SEM_ROOT = (
    ROOT
    / "outputs"
    / "final_evaluation"
    / "semantic_analysis"
)

OUTPUT_ROOT = (
    SEM_ROOT
    / "summary_figures"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Model configuration
# ============================================================

MODEL_ORDER = [
    "rgb_only",
    "position_cross_attention",
    "motion_cross_attention",
]

MODEL_LABELS = {
    "rgb_only":
        "RGB only",

    "position_cross_attention":
        "RGB + hand position",

    "motion_cross_attention":
        "RGB + hand position + velocity",
}


# ============================================================
# Load semantic metrics
# ============================================================

semantic_path = (
    SEM_ROOT
    / "semantic_metrics_aggregated.csv"
)

if not semantic_path.exists():
    raise FileNotFoundError(
        f"Missing semantic metrics file: {semantic_path}"
    )

df = pd.read_csv(
    semantic_path
)


# ============================================================
# Select 10% observation results
# ============================================================

fraction = 10

subset = (
    df[
        df["fraction"] == fraction
    ]
    .set_index("model")
    .loc[MODEL_ORDER]
)


# ============================================================
# Prepare metrics
# ============================================================

action = (
    subset[
        "action_accuracy_mean"
    ].values
    * 100
)

verb = (
    subset[
        "verb_accuracy_mean"
    ].values
    * 100
)

obj = (
    subset[
        "object_accuracy_mean"
    ].values
    * 100
)


x = np.arange(
    len(MODEL_ORDER)
)

width = 0.24


# ============================================================
# FIGURE 1
# Action / Verb / Object accuracy at 10%
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    x - width,
    action,
    width,
    label="Action accuracy"
)

plt.bar(
    x,
    verb,
    width,
    label="Verb accuracy"
)

plt.bar(
    x + width,
    obj,
    width,
    label="Object accuracy"
)

plt.xticks(
    x,
    [
        MODEL_LABELS[m]
        for m in MODEL_ORDER
    ],
    rotation=10
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "What Does Each Modality Contribute at 10% Observation?"
)

plt.ylim(
    0,
    105
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_ROOT
    / "verb_object_action_10pct.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_ROOT
    / "verb_object_action_10pct.pdf",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
# Semantic error decomposition at 10%
# ============================================================

exact = (
    subset[
        "exact_action_rate_mean"
    ].values
    * 100
)

correct_object_wrong_verb = (
    subset[
        "correct_object_wrong_verb_rate_mean"
    ].values
    * 100
)

correct_verb_wrong_object = (
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


plt.figure(
    figsize=(10, 6)
)

plt.bar(
    x,
    exact,
    label="Correct action"
)

plt.bar(
    x,
    correct_object_wrong_verb,
    bottom=exact,
    label="Correct object, wrong verb"
)

plt.bar(
    x,
    correct_verb_wrong_object,
    bottom=(
        exact
        + correct_object_wrong_verb
    ),
    label="Correct verb, wrong object"
)

plt.bar(
    x,
    both_wrong,
    bottom=(
        exact
        + correct_object_wrong_verb
        + correct_verb_wrong_object
    ),
    label="Both wrong"
)

plt.xticks(
    x,
    [
        MODEL_LABELS[m]
        for m in MODEL_ORDER
    ],
    rotation=10
)

plt.ylabel(
    "Predictions (%)"
)

plt.title(
    "Semantic Error Breakdown at 10% Observation"
)

plt.ylim(
    0,
    100
)

plt.legend(
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.12
    ),
    ncol=2
)

plt.tight_layout()

plt.savefig(
    OUTPUT_ROOT
    / "semantic_error_breakdown_10pct.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_ROOT
    / "semantic_error_breakdown_10pct.pdf",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3
# Verb accuracy across all observation fractions
# ============================================================

plt.figure(
    figsize=(9, 6)
)

for model_name in MODEL_ORDER:

    model_df = (
        df[
            df["model"]
            == model_name
        ]
        .sort_values(
            "fraction"
        )
    )

    plt.errorbar(
        model_df["fraction"],
        model_df[
            "verb_accuracy_mean"
        ] * 100,
        yerr=(
            model_df[
                "verb_accuracy_std"
            ] * 100
        ),
        marker="o",
        linewidth=2,
        capsize=4,
        label=MODEL_LABELS[
            model_name
        ],
    )

plt.xlabel(
    "Observed Portion of Action (%)"
)

plt.ylabel(
    "Verb Accuracy (%)"
)

plt.title(
    "Verb Recognition Across Observation Fractions"
)

plt.xticks(
    [
        10,
        20,
        30,
        50,
        75,
        100
    ]
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_ROOT
    / "verb_accuracy_main.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_ROOT
    / "verb_accuracy_main.pdf",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Print main 10% semantic numbers
# ============================================================

print()
print("=" * 90)
print("10% SEMANTIC SUMMARY")
print("=" * 90)

for model_name in MODEL_ORDER:

    row = subset.loc[
        model_name
    ]

    print()
    print(
        MODEL_LABELS[
            model_name
        ]
    )

    print(
        f"  Action accuracy: "
        f"{row['action_accuracy_mean'] * 100:.2f}%"
    )

    print(
        f"  Verb accuracy: "
        f"{row['verb_accuracy_mean'] * 100:.2f}%"
    )

    print(
        f"  Object accuracy: "
        f"{row['object_accuracy_mean'] * 100:.2f}%"
    )

    print(
        f"  Correct object / wrong verb: "
        f"{row['correct_object_wrong_verb_rate_mean'] * 100:.2f}%"
    )

    print(
        f"  Correct verb / wrong object: "
        f"{row['correct_verb_wrong_object_rate_mean'] * 100:.2f}%"
    )

    print(
        f"  Both wrong: "
        f"{row['both_wrong_rate_mean'] * 100:.2f}%"
    )


# ============================================================
# Compute semantic gains safely
# ============================================================

rgb = subset.loc[
    "rgb_only"
]

position = subset.loc[
    "position_cross_attention"
]

motion = subset.loc[
    "motion_cross_attention"
]


position_vs_rgb_verb = (
    position["verb_accuracy_mean"]
    - rgb["verb_accuracy_mean"]
) * 100

velocity_vs_position_verb = (
    motion["verb_accuracy_mean"]
    - position["verb_accuracy_mean"]
) * 100

full_vs_rgb_verb = (
    motion["verb_accuracy_mean"]
    - rgb["verb_accuracy_mean"]
) * 100

full_vs_rgb_object = (
    motion["object_accuracy_mean"]
    - rgb["object_accuracy_mean"]
) * 100

full_vs_rgb_action = (
    motion["action_accuracy_mean"]
    - rgb["action_accuracy_mean"]
) * 100

position_vs_rgb_action = (
    position["action_accuracy_mean"]
    - rgb["action_accuracy_mean"]
) * 100

velocity_vs_position_action = (
    motion["action_accuracy_mean"]
    - position["action_accuracy_mean"]
) * 100

rgb_object_wrong_verb = (
    rgb[
        "correct_object_wrong_verb_rate_mean"
    ]
    * 100
)

motion_object_wrong_verb = (
    motion[
        "correct_object_wrong_verb_rate_mean"
    ]
    * 100
)

reduction_object_wrong_verb = (
    motion_object_wrong_verb
    - rgb_object_wrong_verb
)


# ============================================================
# Print semantic gains
# ============================================================

print()
print("=" * 90)
print("10% GAINS")
print("=" * 90)

print(
    f"Position vs RGB - Action: "
    f"{position_vs_rgb_action:+.2f} pp"
)

print(
    f"Velocity vs Position - Action: "
    f"{velocity_vs_position_action:+.2f} pp"
)

print(
    f"Full vs RGB - Action: "
    f"{full_vs_rgb_action:+.2f} pp"
)

print()

print(
    f"Position vs RGB - Verb: "
    f"{position_vs_rgb_verb:+.2f} pp"
)

print(
    f"Velocity vs Position - Verb: "
    f"{velocity_vs_position_verb:+.2f} pp"
)

print(
    f"Full vs RGB - Verb: "
    f"{full_vs_rgb_verb:+.2f} pp"
)

print()

print(
    f"Full vs RGB - Object: "
    f"{full_vs_rgb_object:+.2f} pp"
)

print()

print(
    f"RGB correct-object / wrong-verb errors: "
    f"{rgb_object_wrong_verb:.2f}%"
)

print(
    f"Full-model correct-object / wrong-verb errors: "
    f"{motion_object_wrong_verb:.2f}%"
)

print(
    f"Change in correct-object / wrong-verb errors: "
    f"{reduction_object_wrong_verb:+.2f} pp"
)


# ============================================================
# Save publication-style 10% summary table
# ============================================================

table_rows = []

for model_name in MODEL_ORDER:

    row = subset.loc[
        model_name
    ]

    table_rows.append(
        {
            "Model":
                MODEL_LABELS[
                    model_name
                ],

            "Action Accuracy (%)":
                row[
                    "action_accuracy_mean"
                ] * 100,

            "Verb Accuracy (%)":
                row[
                    "verb_accuracy_mean"
                ] * 100,

            "Object Accuracy (%)":
                row[
                    "object_accuracy_mean"
                ] * 100,

            "Correct Object, Wrong Verb (%)":
                row[
                    "correct_object_wrong_verb_rate_mean"
                ] * 100,

            "Correct Verb, Wrong Object (%)":
                row[
                    "correct_verb_wrong_object_rate_mean"
                ] * 100,

            "Both Wrong (%)":
                row[
                    "both_wrong_rate_mean"
                ] * 100,
        }
    )


summary_table = pd.DataFrame(
    table_rows
)

summary_table.to_csv(
    OUTPUT_ROOT
    / "semantic_summary_10pct.csv",
    index=False
)


# ============================================================
# Save semantic gain table
# ============================================================

gain_table = pd.DataFrame(
    [
        {
            "Comparison":
                "Position vs RGB",
            "Action Gain (pp)":
                position_vs_rgb_action,
            "Verb Gain (pp)":
                position_vs_rgb_verb,
        },
        {
            "Comparison":
                "Velocity vs Position",
            "Action Gain (pp)":
                velocity_vs_position_action,
            "Verb Gain (pp)":
                velocity_vs_position_verb,
        },
        {
            "Comparison":
                "Full vs RGB",
            "Action Gain (pp)":
                full_vs_rgb_action,
            "Verb Gain (pp)":
                full_vs_rgb_verb,
        },
    ]
)

gain_table.to_csv(
    OUTPUT_ROOT
    / "semantic_gains_10pct.csv",
    index=False
)


# ============================================================
# Final output
# ============================================================

print()
print("=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    OUTPUT_ROOT
)

print()
print(
    "Created:"
)

print(
    "  verb_object_action_10pct.png"
)

print(
    "  verb_object_action_10pct.pdf"
)

print(
    "  semantic_error_breakdown_10pct.png"
)

print(
    "  semantic_error_breakdown_10pct.pdf"
)

print(
    "  verb_accuracy_main.png"
)

print(
    "  verb_accuracy_main.pdf"
)

print(
    "  semantic_summary_10pct.csv"
)

print(
    "  semantic_gains_10pct.csv"
)