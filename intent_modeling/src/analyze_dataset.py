from pathlib import Path
from collections import Counter

import numpy as np


PROCESSED_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/processed"
)


def summarize_split(split_name, percentage):
    path = PROCESSED_ROOT / f"{split_name}_{percentage:03d}.npz"

    data = np.load(path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    observed_frames = data["observed_frames"]

    print()
    print("=" * 70)
    print(f"{split_name.upper()} — {percentage}% observation")
    print("=" * 70)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("dtype:", X.dtype)

    print()
    print("Feature statistics")
    print("Min:", float(X.min()))
    print("Max:", float(X.max()))
    print("Mean:", float(X.mean()))
    print("Std:", float(X.std()))

    print()
    print("NaN values:", int(np.isnan(X).sum()))
    print("Inf values:", int(np.isinf(X).sum()))

    print()
    print("Observed frame statistics")
    print("Min:", int(observed_frames.min()))
    print("Max:", int(observed_frames.max()))
    print("Mean:", float(observed_frames.mean()))
    print("Median:", float(np.median(observed_frames)))

    counts = Counter(y.tolist())

    print()
    print("Number of classes:", len(counts))

    print()
    print("Class distribution")

    for label in sorted(counts):
        print(
            f"Action {label:2d}: "
            f"{counts[label]:3d} samples"
        )

    return X, y


def check_label_consistency():
    print()
    print("=" * 70)
    print("CHECKING LABEL CONSISTENCY ACROSS OBSERVATION FRACTIONS")
    print("=" * 70)

    fractions = [10, 20, 30, 50, 75, 100]

    for split_name in ["train", "val"]:
        reference = None

        for percentage in fractions:
            path = PROCESSED_ROOT / f"{split_name}_{percentage:03d}.npz"
            data = np.load(path, allow_pickle=True)

            y = data["y"]

            if reference is None:
                reference = y
            else:
                if not np.array_equal(reference, y):
                    print(
                        f"WARNING: label mismatch in "
                        f"{split_name}_{percentage:03d}"
                    )
                    break
        else:
            print(
                f"{split_name}: labels are consistent "
                f"across all observation fractions"
            )


def check_sequence_differences():
    print()
    print("=" * 70)
    print("CHECKING WHETHER PARTIAL OBSERVATIONS DIFFER")
    print("=" * 70)

    train_10 = np.load(
        PROCESSED_ROOT / "train_010.npz",
        allow_pickle=True
    )["X"]

    train_100 = np.load(
        PROCESSED_ROOT / "train_100.npz",
        allow_pickle=True
    )["X"]

    difference = np.abs(train_10 - train_100)

    print(
        "Mean absolute difference "
        "between 10% and 100% sequences:",
        float(difference.mean())
    )

    identical_samples = np.all(
        np.isclose(train_10, train_100),
        axis=(1, 2)
    )

    print(
        "Completely identical samples:",
        int(identical_samples.sum()),
        "/",
        len(identical_samples)
    )


def inspect_validity_flags():
    print()
    print("=" * 70)
    print("HAND VALIDITY ANALYSIS")
    print("=" * 70)

    data = np.load(
        PROCESSED_ROOT / "train_100.npz",
        allow_pickle=True
    )

    X = data["X"]

    # Feature layout:
    # 0     = left hand validity
    # 1:64  = left hand XYZ
    # 64    = right hand validity
    # 65:   = right hand XYZ

    left_valid = X[:, :, 0]
    right_valid = X[:, :, 64]

    print(
        "Left-hand valid ratio:",
        float(left_valid.mean())
    )

    print(
        "Right-hand valid ratio:",
        float(right_valid.mean())
    )

    both_valid = (
        (left_valid > 0.5)
        & (right_valid > 0.5)
    )

    print(
        "Both hands valid ratio:",
        float(both_valid.mean())
    )


if __name__ == "__main__":

    print("H2O Dataset Sanity Analysis")

    summarize_split(
        "train",
        100
    )

    summarize_split(
        "val",
        100
    )

    summarize_split(
        "train",
        10
    )

    summarize_split(
        "val",
        10
    )

    check_label_consistency()

    check_sequence_differences()

    inspect_validity_flags()