from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling"
)

RESULTS_PATH = (
    ROOT
    / "outputs"
    / "final_evaluation"
    / "aggregated_evaluation_metrics.csv"
)

FIGURE_DIR = (
    ROOT
    / "outputs"
    / "final_evaluation"
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Load results
# =========================================================

df = pd.read_csv(
    RESULTS_PATH
)


MODEL_ORDER = [
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
    100
]


# =========================================================
# Utility
# =========================================================

def get_model_data(
    model_name
):

    model_df = (
        df[
            df["model"]
            == model_name
        ]
        .sort_values(
            "fraction"
        )
    )

    return model_df


# =========================================================
# Figure 1
# Top-1 anticipation curve
# =========================================================

plt.figure(
    figsize=(9, 6)
)

for model_name in MODEL_ORDER:

    model_df = get_model_data(
        model_name
    )

    plt.errorbar(
        model_df["fraction"],
        model_df["top1_mean"] * 100,
        yerr=(
            model_df["top1_std"]
            * 100
        ),
        marker="o",
        capsize=4,
        linewidth=2,
        label=MODEL_LABELS[
            model_name
        ],
    )


plt.xlabel(
    "Observed Portion of Action (%)",
    fontsize=12
)

plt.ylabel(
    "Top-1 Accuracy (%)",
    fontsize=12
)

plt.title(
    "Early Action Prediction Performance",
    fontsize=14
)

plt.xticks(
    FRACTIONS
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "top1_anticipation_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    FIGURE_DIR
    / "top1_anticipation_curve.pdf",
    bbox_inches="tight"
)

plt.close()


# =========================================================
# Figure 2
# Top-3 anticipation curve
# =========================================================

plt.figure(
    figsize=(9, 6)
)

for model_name in MODEL_ORDER:

    model_df = get_model_data(
        model_name
    )

    plt.errorbar(
        model_df["fraction"],
        model_df["top3_mean"] * 100,
        yerr=(
            model_df["top3_std"]
            * 100
        ),
        marker="o",
        capsize=4,
        linewidth=2,
        label=MODEL_LABELS[
            model_name
        ],
    )


plt.xlabel(
    "Observed Portion of Action (%)",
    fontsize=12
)

plt.ylabel(
    "Top-3 Accuracy (%)",
    fontsize=12
)

plt.title(
    "Top-3 Early Action Prediction Performance",
    fontsize=14
)

plt.xticks(
    FRACTIONS
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "top3_anticipation_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    FIGURE_DIR
    / "top3_anticipation_curve.pdf",
    bbox_inches="tight"
)

plt.close()


# =========================================================
# Figure 3
# Macro-F1 anticipation curve
# =========================================================

plt.figure(
    figsize=(9, 6)
)

for model_name in MODEL_ORDER:

    model_df = get_model_data(
        model_name
    )

    plt.errorbar(
        model_df["fraction"],
        model_df["macro_f1_mean"] * 100,
        yerr=(
            model_df["macro_f1_std"]
            * 100
        ),
        marker="o",
        capsize=4,
        linewidth=2,
        label=MODEL_LABELS[
            model_name
        ],
    )


plt.xlabel(
    "Observed Portion of Action (%)",
    fontsize=12
)

plt.ylabel(
    "Macro-F1 (%)",
    fontsize=12
)

plt.title(
    "Macro-F1 Across Observation Fractions",
    fontsize=14
)

plt.xticks(
    FRACTIONS
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "macro_f1_anticipation_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    FIGURE_DIR
    / "macro_f1_anticipation_curve.pdf",
    bbox_inches="tight"
)

plt.close()


# =========================================================
# Figure 4
# Gain over RGB-only
# =========================================================

rgb = (
    get_model_data(
        "rgb_only"
    )
    .set_index(
        "fraction"
    )
)


comparison_models = [
    "position_cross_attention",
    "motion_cross_attention",
]


plt.figure(
    figsize=(9, 6)
)


for model_name in comparison_models:

    model_df = (
        get_model_data(
            model_name
        )
        .set_index(
            "fraction"
        )
    )

    gains = (
        (
            model_df[
                "top1_mean"
            ]
            -
            rgb[
                "top1_mean"
            ]
        )
        * 100
    )

    plt.plot(
        gains.index,
        gains.values,
        marker="o",
        linewidth=2,
        label=MODEL_LABELS[
            model_name
        ],
    )


plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Observed Portion of Action (%)",
    fontsize=12
)

plt.ylabel(
    "Top-1 Gain over RGB-only "
    "(percentage points)",
    fontsize=12
)

plt.title(
    "Benefit of 3D Hand Information",
    fontsize=14
)

plt.xticks(
    FRACTIONS
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "gain_over_rgb.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    FIGURE_DIR
    / "gain_over_rgb.pdf",
    bbox_inches="tight"
)

plt.close()


# =========================================================
# Figure 5
# Velocity contribution
# =========================================================

position = (
    get_model_data(
        "position_cross_attention"
    )
    .set_index(
        "fraction"
    )
)

motion = (
    get_model_data(
        "motion_cross_attention"
    )
    .set_index(
        "fraction"
    )
)


velocity_gain = (
    (
        motion[
            "top1_mean"
        ]
        -
        position[
            "top1_mean"
        ]
    )
    * 100
)


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    velocity_gain.index,
    velocity_gain.values,
    marker="o",
    linewidth=2,
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Observed Portion of Action (%)",
    fontsize=12
)

plt.ylabel(
    "Top-1 Gain from Velocity "
    "(percentage points)",
    fontsize=12
)

plt.title(
    "Contribution of Explicit Hand Velocity",
    fontsize=14
)

plt.xticks(
    FRACTIONS
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "velocity_contribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    FIGURE_DIR
    / "velocity_contribution.pdf",
    bbox_inches="tight"
)

plt.close()


# =========================================================
# Print useful numerical comparisons
# =========================================================

print()
print("=" * 80)
print("TOP-1 GAINS OVER RGB")
print("=" * 80)

motion_df = (
    get_model_data(
        "motion_cross_attention"
    )
    .set_index(
        "fraction"
    )
)

position_df = (
    get_model_data(
        "position_cross_attention"
    )
    .set_index(
        "fraction"
    )
)

rgb_df = (
    get_model_data(
        "rgb_only"
    )
    .set_index(
        "fraction"
    )
)


for fraction in FRACTIONS:

    fused_gain = (
        motion_df.loc[
            fraction,
            "top1_mean"
        ]
        -
        rgb_df.loc[
            fraction,
            "top1_mean"
        ]
    ) * 100

    cross_gain = (
        position_df.loc[
            fraction,
            "top1_mean"
        ]
        -
        rgb_df.loc[
            fraction,
            "top1_mean"
        ]
    ) * 100

    velocity_gain_value = (
        motion_df.loc[
            fraction,
            "top1_mean"
        ]
        -
        position_df.loc[
            fraction,
            "top1_mean"
        ]
    ) * 100

    print(
        f"{fraction:>3}% | "
        f"Position vs RGB: "
        f"{cross_gain:+.2f} pp | "
        f"Velocity addition: "
        f"{velocity_gain_value:+.2f} pp | "
        f"Full vs RGB: "
        f"{fused_gain:+.2f} pp"
    )


print()
print("=" * 80)
print("FIGURES SAVED")
print("=" * 80)

print(
    FIGURE_DIR
)