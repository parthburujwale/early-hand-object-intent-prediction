from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/results"
)


gru = pd.read_csv(
    RESULTS_ROOT / "gru_summary.csv"
)

transformer = pd.read_csv(
    RESULTS_ROOT / "transformer_summary.csv"
)


print()
print("MODEL COMPARISON")
print()

comparison = pd.DataFrame({
    "Observation": gru["fraction"],
    "GRU Top-1": gru["top1_percent"],
    "Transformer Top-1": transformer["top1_percent"],
    "GRU Top-3": gru["top3_percent"],
    "Transformer Top-3": transformer["top3_percent"],
})

print(
    comparison.to_string(
        index=False
    )
)


plt.figure(figsize=(8, 5))

plt.plot(
    gru["fraction"],
    gru["top1_percent"],
    marker="o",
    label="GRU"
)

plt.plot(
    transformer["fraction"],
    transformer["top1_percent"],
    marker="o",
    label="Transformer"
)

plt.xlabel(
    "Observed Portion of Action (%)"
)

plt.ylabel(
    "Top-1 Validation Accuracy (%)"
)

plt.title(
    "Early Intent Prediction: GRU vs Transformer"
)

plt.xticks(
    [10, 20, 30, 50, 75, 100]
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

output_path = (
    RESULTS_ROOT
    / "gru_vs_transformer.png"
)

plt.savefig(
    output_path,
    dpi=300
)

print()
print(
    "Saved comparison plot:"
)

print(
    output_path
)