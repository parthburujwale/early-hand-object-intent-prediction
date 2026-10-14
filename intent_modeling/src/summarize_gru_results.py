from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/results"
)

FRACTIONS = [10, 20, 30, 50, 75, 100]


rows = []

for fraction in FRACTIONS:
    path = RESULTS_ROOT / f"gru_results_{fraction:03d}.csv"

    df = pd.read_csv(path)

    row = df.iloc[0].to_dict()
    rows.append(row)


summary = pd.DataFrame(rows)

summary = summary.sort_values("fraction")

summary["top1_percent"] = summary["top1_accuracy"] * 100
summary["top3_percent"] = summary["top3_accuracy"] * 100


summary_path = RESULTS_ROOT / "gru_summary.csv"

summary.to_csv(
    summary_path,
    index=False
)


print()
print("GRU RESULTS")
print()

print(
    summary[
        [
            "fraction",
            "best_epoch",
            "top1_percent",
            "top3_percent",
        ]
    ].to_string(index=False)
)


plt.figure(figsize=(8, 5))

plt.plot(
    summary["fraction"],
    summary["top1_percent"],
    marker="o",
    label="Top-1 Accuracy"
)

plt.plot(
    summary["fraction"],
    summary["top3_percent"],
    marker="o",
    label="Top-3 Accuracy"
)

plt.xlabel("Observed Portion of Action (%)")
plt.ylabel("Validation Accuracy (%)")
plt.title("GRU Early Intent Prediction on H2O")

plt.xticks(FRACTIONS)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plot_path = RESULTS_ROOT / "gru_intent_curve.png"

plt.savefig(
    plot_path,
    dpi=300
)

print()
print("Saved summary:")
print(summary_path)

print()
print("Saved plot:")
print(plot_path)