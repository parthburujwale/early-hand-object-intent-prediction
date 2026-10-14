from pathlib import Path
import numpy as np


PROCESSED_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/processed"
)

NORMALIZED_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/normalized"
)

FRACTIONS = [10, 20, 30, 50, 75, 100]


def normalize_sequence(X):
    """
    X shape:
        [N, T, 128]

    Layout:
        0      left_valid
        1:64   left hand, 21 x 3
        64     right_valid
        65:128 right hand, 21 x 3

    We normalize each hand independently relative to its wrist.
    """

    X = X.copy()

    left_valid = X[:, :, 0:1]
    left_hand = X[:, :, 1:64].reshape(
        X.shape[0],
        X.shape[1],
        21,
        3
    )

    right_valid = X[:, :, 64:65]
    right_hand = X[:, :, 65:128].reshape(
        X.shape[0],
        X.shape[1],
        21,
        3
    )

    # Wrist joint = joint 0
    left_wrist = left_hand[:, :, 0:1, :]
    right_wrist = right_hand[:, :, 0:1, :]

    # Translate every joint so wrist becomes origin
    left_hand = left_hand - left_wrist
    right_hand = right_hand - right_wrist

    # Flatten again
    left_hand = left_hand.reshape(
        X.shape[0],
        X.shape[1],
        63
    )

    right_hand = right_hand.reshape(
        X.shape[0],
        X.shape[1],
        63
    )

    normalized = np.concatenate(
        [
            left_valid,
            left_hand,
            right_valid,
            right_hand,
        ],
        axis=2
    )

    return normalized.astype(np.float32)


def process_file(split, fraction):
    input_path = (
        PROCESSED_ROOT
        / f"{split}_{fraction:03d}.npz"
    )

    data = np.load(
        input_path,
        allow_pickle=True
    )

    X = data["X"]

    X_normalized = normalize_sequence(X)

    NORMALIZED_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        NORMALIZED_ROOT
        / f"{split}_{fraction:03d}.npz"
    )

    np.savez_compressed(
        output_path,
        X=X_normalized,
        y=data["y"],
        ids=data["ids"],
        paths=data["paths"],
        start_frames=data["start_frames"],
        end_frames=data["end_frames"],
        observed_frames=data["observed_frames"],
        observation_fraction=data["observation_fraction"],
    )

    print(
        f"{split}_{fraction:03d}: "
        f"{X.shape} -> {X_normalized.shape}"
    )

    print(
        "Mean:",
        float(X_normalized.mean())
    )

    print(
        "Std:",
        float(X_normalized.std())
    )

    print()


if __name__ == "__main__":

    print("Normalizing H2O hand poses")
    print()

    for split in ["train", "val"]:
        for fraction in FRACTIONS:
            process_file(
                split,
                fraction
            )

    print("Normalization complete.")