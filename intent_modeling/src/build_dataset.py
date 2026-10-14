from pathlib import Path
import csv
import math

import numpy as np


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/intent_prediction/data/h2o"
)

OUTPUT_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/processed"
)

OBSERVATION_FRACTIONS = [
    0.10,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
]

SEQUENCE_LENGTH = 32


# ============================================================
# Load one hand-pose frame
# ============================================================

def load_hand_pose(path):
    """
    H2O hand-pose format:

    left_valid
    21 x 3 left hand coordinates
    right_valid
    21 x 3 right hand coordinates

    Total values = 128
    """

    values = np.loadtxt(
        path,
        dtype=np.float32
    )

    values = np.asarray(values).reshape(-1)

    if len(values) != 128:
        raise ValueError(
            f"Expected 128 values in {path}, "
            f"found {len(values)}"
        )

    left_valid = values[0]

    left_hand = values[
        1:64
    ].reshape(21, 3)

    right_valid = values[64]

    right_hand = values[
        65:128
    ].reshape(21, 3)

    # 128-dimensional frame feature
    feature = np.concatenate(
        [
            np.array(
                [left_valid],
                dtype=np.float32
            ),
            left_hand.reshape(-1),

            np.array(
                [right_valid],
                dtype=np.float32
            ),
            right_hand.reshape(-1),
        ]
    )

    return feature.astype(np.float32)


# ============================================================
# Temporal resampling
# ============================================================

def resample_sequence(sequence, target_length):
    """
    Resample a variable-length sequence to a fixed number
    of frames using nearest-neighbor temporal sampling.

    Input:
        sequence:
            [T, D]

    Output:
        [target_length, D]
    """

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    num_frames = len(sequence)

    if num_frames == 0:
        raise ValueError(
            "Cannot resample an empty sequence."
        )

    if num_frames == target_length:
        return sequence

    indices = np.linspace(
        0,
        num_frames - 1,
        target_length
    )

    indices = np.round(
        indices
    ).astype(int)

    return sequence[indices]


# ============================================================
# Read action split file
# ============================================================

def read_action_split(split_path):
    """
    Expected format:

    id path action_label start_act end_act start_frame end_frame
    """

    samples = []

    with open(
        split_path,
        "r"
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter=" "
        )

        for row in reader:

            # csv.DictReader can occasionally produce
            # empty entries with repeated whitespace.
            row = {
                key: value
                for key, value in row.items()
                if key is not None
            }

            sample = {
                "id": int(row["id"]),
                "path": row["path"],
                "action_label": int(
                    row["action_label"]
                ),
                "start_act": int(
                    row["start_act"]
                ),
                "end_act": int(
                    row["end_act"]
                ),
                "start_frame": int(
                    row["start_frame"]
                ),
                "end_frame": int(
                    row["end_frame"]
                ),
            }

            samples.append(sample)

    return samples


# ============================================================
# Load an action segment
# ============================================================

def load_action_segment(sample):
    """
    Loads all hand-pose frames belonging to one
    annotated H2O action.
    """

    sequence_path = (
        DATA_ROOT
        / sample["path"]
        / "cam4"
        / "hand_pose"
    )

    frames = []

    missing_frames = []

    for frame_id in range(
        sample["start_act"],
        sample["end_act"] + 1
    ):

        frame_path = (
            sequence_path
            / f"{frame_id:06d}.txt"
        )

        if not frame_path.exists():
            missing_frames.append(
                frame_id
            )
            continue

        try:
            feature = load_hand_pose(
                frame_path
            )

            frames.append(feature)

        except Exception as e:

            print(
                f"\nWarning: failed to load "
                f"{frame_path}"
            )

            print(e)

    if missing_frames:

        print(
            f"\nWarning: {len(missing_frames)} "
            f"missing frames for "
            f"{sample['path']}"
        )

    if len(frames) == 0:
        return None

    return np.stack(
        frames,
        axis=0
    )


# ============================================================
# Create partial observation
# ============================================================

def create_partial_sequence(
    full_sequence,
    fraction
):
    """
    Keep only the first fraction of an action.

    Example:

    fraction = 0.20

    means the model only observes the first
    20% of the action.
    """

    total_frames = len(
        full_sequence
    )

    observed_frames = max(
        1,
        math.ceil(
            total_frames * fraction
        )
    )

    partial_sequence = (
        full_sequence[
            :observed_frames
        ]
    )

    fixed_sequence = (
        resample_sequence(
            partial_sequence,
            SEQUENCE_LENGTH
        )
    )

    return (
        fixed_sequence,
        observed_frames
    )


# ============================================================
# Build one complete split
# ============================================================

def build_split(split_name):
    """
    Creates datasets for one split:

    train
    val
    """

    split_path = (
        DATA_ROOT
        / "label_split"
        / f"action_{split_name}.txt"
    )

    print()
    print("=" * 70)
    print(
        f"Building split: {split_name}"
    )
    print("=" * 70)

    samples = read_action_split(
        split_path
    )

    print(
        f"Action segments: {len(samples)}"
    )

    datasets = {
        fraction: {
            "X": [],
            "y": [],
            "ids": [],
            "paths": [],
            "start_frames": [],
            "end_frames": [],
            "observed_frames": [],
        }
        for fraction
        in OBSERVATION_FRACTIONS
    }

    skipped = 0

    for index, sample in enumerate(
        samples
    ):

        full_sequence = (
            load_action_segment(
                sample
            )
        )

        if full_sequence is None:

            skipped += 1

            print(
                f"Skipping sample "
                f"{sample['id']}: "
                f"no hand-pose frames"
            )

            continue

        for fraction in (
            OBSERVATION_FRACTIONS
        ):

            (
                sequence,
                observed_frames,
            ) = create_partial_sequence(
                full_sequence,
                fraction
            )

            data = datasets[
                fraction
            ]

            data["X"].append(
                sequence
            )

            data["y"].append(
                sample["action_label"]
            )

            data["ids"].append(
                sample["id"]
            )

            data["paths"].append(
                sample["path"]
            )

            data["start_frames"].append(
                sample["start_act"]
            )

            data["end_frames"].append(
                sample["end_act"]
            )

            data[
                "observed_frames"
            ].append(
                observed_frames
            )

        if (
            (index + 1) % 100 == 0
            or
            index + 1 == len(samples)
        ):

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(samples)}"
            )

    # --------------------------------------------------------
    # Save each observation fraction separately
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    for fraction, data in (
        datasets.items()
    ):

        X = np.stack(
            data["X"],
            axis=0
        ).astype(
            np.float32
        )

        y = np.array(
            data["y"],
            dtype=np.int64
        )

        ids = np.array(
            data["ids"],
            dtype=np.int64
        )

        paths = np.array(
            data["paths"]
        )

        start_frames = np.array(
            data["start_frames"],
            dtype=np.int64
        )

        end_frames = np.array(
            data["end_frames"],
            dtype=np.int64
        )

        observed_frames = np.array(
            data["observed_frames"],
            dtype=np.int64
        )

        percentage = int(
            fraction * 100
        )

        output_path = (
            OUTPUT_ROOT
            / (
                f"{split_name}_"
                f"{percentage:03d}.npz"
            )
        )

        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            ids=ids,
            paths=paths,
            start_frames=start_frames,
            end_frames=end_frames,
            observed_frames=observed_frames,
            observation_fraction=fraction,
        )

        print()
        print(
            f"Saved: {output_path}"
        )

        print(
            "X shape:",
            X.shape
        )

        print(
            "y shape:",
            y.shape
        )

        print(
            "Unique actions:",
            len(np.unique(y))
        )

    print()
    print(
        f"Skipped samples: {skipped}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "H2O Partial-Observation "
        "Dataset Builder"
    )

    print(
        f"Data root: {DATA_ROOT}"
    )

    print(
        f"Output root: {OUTPUT_ROOT}"
    )

    print(
        f"Fixed sequence length: "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        "Observation fractions:",
        OBSERVATION_FRACTIONS
    )

    build_split(
        "train"
    )

    build_split(
        "val"
    )

    print()
    print("=" * 70)
    print(
        "Dataset construction complete."
    )
    print("=" * 70)