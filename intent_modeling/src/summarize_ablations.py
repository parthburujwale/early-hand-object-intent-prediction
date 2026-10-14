from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/results"
)

ABLATION_ROOT = ROOT / "ablations"


DISPLAY_NAMES = {
    "hand": "Hand only",
    "object": "Object only",
    "relative": "Relative only",
    "hand_relative": "Hand + Relative",
    "hand_object": "Hand + Object",
}


def main():

    rows = []

    for mode in DISPLAY_NAMES:

        path = (
            ABLATION_ROOT
            / f"{mode}_010.csv"
        )

        df = pd.read_csv(path)

        row = df.iloc[0]

        rows.append({
            "model": DISPLAY_NAMES[mode],
            "top1": (
                float(
                    row["top1_accuracy"]
                )
                * 100
            ),
            "top3": (
                float(
                    row["top3_accuracy"]
                )
                * 100
            )
        })

    # Existing full model
    full = pd.read_csv(
        ROOT
        / "spatiotemporal_transformer_results_010.csv"
    ).iloc[0]

    rows.append({
        "model":
            "Hand + Object + Relative",

        "top1":
            float(
                full["top1_accuracy"]
            )
            * 100,

        "top3":
            float(
                full["top3_accuracy"]
            )
            * 100,
    })

    results = pd.DataFrame(
        rows
    )

    results = results.sort_values(
        "top1",
        ascending=False
    )

    output_csv = (
        ABLATION_ROOT
        / "ablation_summary_010.csv"
    )

    results.to_csv(
        output_csv,
        index=False
    )

    print()
    print(
        results.to_string(
            index=False
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        results["model"],
        results["top1"]
    )

    plt.ylabel(
        "Top-1 Accuracy (%)"
    )

    plt.xlabel(
        "Input modalities"
    )

    plt.title(
        "10% Observation Ablation Study"
    )

    plt.ylim(
        0,
        100
    )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()

    output_plot = (
        ABLATION_ROOT
        / "ablation_top1_010.png"
    )

    plt.savefig(
        output_plot,
        dpi=300
    )

    plt.close()

    print()
    print("Saved:")
    print(output_csv)
    print(output_plot)


if __name__ == "__main__":
    main()