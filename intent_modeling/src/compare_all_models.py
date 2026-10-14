from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/results"
)

FRACTIONS = [10, 20, 30, 50, 75, 100]


def load_spatiotemporal():
    rows = []

    for fraction in FRACTIONS:
        path = (
            RESULTS_ROOT
            / f"spatiotemporal_transformer_results_{fraction:03d}.csv"
        )

        df = pd.read_csv(path)

        row = df.iloc[0]

        rows.append({
            "fraction": fraction,
            "best_epoch": int(row["best_epoch"]),
            "top1_accuracy": float(row["top1_accuracy"]),
            "top3_accuracy": float(row["top3_accuracy"]),
        })

    return pd.DataFrame(rows)


def load_existing_summary(filename):
    path = RESULTS_ROOT / filename

    df = pd.read_csv(path)

    return df


def main():

    st = load_spatiotemporal()

    gru = load_existing_summary(
        "gru_summary.csv"
    )

    transformer = load_existing_summary(
        "transformer_summary.csv"
    )

    # --------------------------------------------------
    # Save spatiotemporal summary
    # --------------------------------------------------

    st_summary = st.copy()

    st_summary["top1_percent"] = (
        st_summary["top1_accuracy"] * 100
    )

    st_summary["top3_percent"] = (
        st_summary["top3_accuracy"] * 100
    )

    summary_path = (
        RESULTS_ROOT
        / "spatiotemporal_transformer_summary.csv"
    )

    st_summary.to_csv(
        summary_path,
        index=False
    )

    print()
    print("Spatiotemporal Transformer")
    print(st_summary.to_string(index=False))

    # --------------------------------------------------
    # Combined comparison
    # --------------------------------------------------

    comparison = pd.DataFrame({
        "Observation": FRACTIONS,

        "GRU Top-1":
            gru["top1_percent"].values,

        "Pose Transformer Top-1":
            transformer["top1_percent"].values,

        "Spatiotemporal Transformer Top-1":
            st_summary["top1_percent"].values,

        "GRU Top-3":
            gru["top3_percent"].values,

        "Pose Transformer Top-3":
            transformer["top3_percent"].values,

        "Spatiotemporal Transformer Top-3":
            st_summary["top3_percent"].values,
    })

    comparison[
        "ST Gain vs GRU"
    ] = (
        comparison[
            "Spatiotemporal Transformer Top-1"
        ]
        - comparison["GRU Top-1"]
    )

    comparison[
        "ST Gain vs Pose Transformer"
    ] = (
        comparison[
            "Spatiotemporal Transformer Top-1"
        ]
        - comparison[
            "Pose Transformer Top-1"
        ]
    )

    comparison_path = (
        RESULTS_ROOT
        / "all_models_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print()
    print("=" * 100)
    print("ALL MODEL COMPARISON")
    print("=" * 100)

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Top-1 figure
    # --------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        FRACTIONS,
        comparison["GRU Top-1"],
        marker="o",
        linewidth=2,
        label="Pose-only GRU"
    )

    plt.plot(
        FRACTIONS,
        comparison[
            "Pose Transformer Top-1"
        ],
        marker="o",
        linewidth=2,
        label="Pose-only Transformer"
    )

    plt.plot(
        FRACTIONS,
        comparison[
            "Spatiotemporal Transformer Top-1"
        ],
        marker="o",
        linewidth=2.5,
        label="Spatiotemporal Transformer"
    )

    plt.axhline(
        y=100 / 36,
        linestyle="--",
        linewidth=1,
        label="Random chance"
    )

    plt.xlabel(
        "Observed Interaction (%)"
    )

    plt.ylabel(
        "Top-1 Accuracy (%)"
    )

    plt.title(
        "Early Human Intent Prediction"
    )

    plt.xticks(
        FRACTIONS
    )

    plt.ylim(
        0,
        100
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    top1_path = (
        RESULTS_ROOT
        / "all_models_top1_comparison.png"
    )

    plt.savefig(
        top1_path,
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # Top-3 figure
    # --------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        FRACTIONS,
        comparison["GRU Top-3"],
        marker="o",
        linewidth=2,
        label="Pose-only GRU"
    )

    plt.plot(
        FRACTIONS,
        comparison[
            "Pose Transformer Top-3"
        ],
        marker="o",
        linewidth=2,
        label="Pose-only Transformer"
    )

    plt.plot(
        FRACTIONS,
        comparison[
            "Spatiotemporal Transformer Top-3"
        ],
        marker="o",
        linewidth=2.5,
        label="Spatiotemporal Transformer"
    )

    plt.axhline(
        y=300 / 36,
        linestyle="--",
        linewidth=1,
        label="Random Top-3"
    )

    plt.xlabel(
        "Observed Interaction (%)"
    )

    plt.ylabel(
        "Top-3 Accuracy (%)"
    )

    plt.title(
        "Early Human Intent Prediction — Top-3"
    )

    plt.xticks(
        FRACTIONS
    )

    plt.ylim(
        0,
        100
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    top3_path = (
        RESULTS_ROOT
        / "all_models_top3_comparison.png"
    )

    plt.savefig(
        top3_path,
        dpi=300
    )

    plt.close()

    print()
    print("Saved:")
    print(summary_path)
    print(comparison_path)
    print(top1_path)
    print(top3_path)


if __name__ == "__main__":
    main()