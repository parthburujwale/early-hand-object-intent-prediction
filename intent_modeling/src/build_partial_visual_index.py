from pathlib import Path
import json
import math
import numpy as np


INDEX_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/rgbd_hand_index"
)

OUTPUT_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/partial_visual_index"
)

FRACTIONS = [
    0.10,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
]

NUM_SAMPLED_FRAMES = 8


def sample_indices(
    num_available,
    num_samples
):
    if num_available <= 0:
        raise ValueError(
            "num_available must be > 0"
        )

    indices = np.linspace(
        0,
        num_available - 1,
        num_samples
    )

    return np.round(
        indices
    ).astype(int)


def build_fraction(
    records,
    fraction
):
    output = []

    for record in records:

        frames = record["frames"]

        total_frames = len(frames)

        observed_count = max(
            1,
            math.ceil(
                total_frames * fraction
            )
        )

        observed_frames = frames[
            :observed_count
        ]

        selected_indices = sample_indices(
            len(observed_frames),
            NUM_SAMPLED_FRAMES
        )

        selected_frames = [
            observed_frames[i]
            for i in selected_indices
        ]

        output.append({
            "id": record["id"],
            "pose_path": record["pose_path"],
            "ego_path": record["ego_path"],
            "action_label": record["action_label"],

            "observation_fraction": fraction,

            "total_action_frames":
                total_frames,

            "observed_action_frames":
                observed_count,

            "sampled_frames":
                selected_frames,
        })

    return output


def process_split(
    split_name
):
    input_path = (
        INDEX_ROOT
        / f"{split_name}.json"
    )

    with open(
        input_path,
        "r"
    ) as f:
        records = json.load(f)

    print()
    print("=" * 70)

    print(
        f"Processing {split_name}"
    )

    print("=" * 70)

    print(
        "Segments:",
        len(records)
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    for fraction in FRACTIONS:

        percentage = int(
            fraction * 100
        )

        samples = build_fraction(
            records,
            fraction
        )

        output_path = (
            OUTPUT_ROOT
            / (
                f"{split_name}_"
                f"{percentage:03d}.json"
            )
        )

        with open(
            output_path,
            "w"
        ) as f:

            json.dump(
                samples,
                f,
                indent=2
            )

        observed_counts = [
            x["observed_action_frames"]
            for x in samples
        ]

        print()
        print(
            f"{percentage}% observation"
        )

        print(
            "Segments:",
            len(samples)
        )

        print(
            "Frames/model input:",
            NUM_SAMPLED_FRAMES
        )

        print(
            "Observed frames min:",
            min(observed_counts)
        )

        print(
            "Observed frames max:",
            max(observed_counts)
        )

        print(
            "Observed frames mean:",
            round(
                float(
                    np.mean(
                        observed_counts
                    )
                ),
                2
            )
        )

        print(
            "Saved:",
            output_path
        )


if __name__ == "__main__":

    process_split(
        "train"
    )

    process_split(
        "val"
    )

    print()
    print(
        "Partial visual indexes complete."
    )