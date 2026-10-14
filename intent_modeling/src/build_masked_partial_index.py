from pathlib import Path
import json
import math

import numpy as np


INDEX_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/"
    "outputs/rgbd_hand_index"
)

OUTPUT_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/"
    "outputs/masked_partial_index"
)


FRACTION_CONFIG = {
    10: {
        "fraction": 0.10,
        "num_samples": 4,
    },
    20: {
        "fraction": 0.20,
        "num_samples": 8,
    },
    30: {
        "fraction": 0.30,
        "num_samples": 12,
    },
    50: {
        "fraction": 0.50,
        "num_samples": 16,
    },
    75: {
        "fraction": 0.75,
        "num_samples": 24,
    },
    100: {
        "fraction": 1.00,
        "num_samples": 32,
    },
}


MAX_SEQUENCE_LENGTH = 32


def uniformly_sample_frames(
    observed_frames,
    requested_samples,
):
    """
    Uniformly sample up to requested_samples
    real frames from the observed portion.

    If fewer real frames exist than requested,
    use all available real frames.
    """

    num_available = len(
        observed_frames
    )

    if num_available == 0:
        raise ValueError(
            "Observed frame list is empty."
        )

    num_real_samples = min(
        requested_samples,
        num_available,
    )

    if num_real_samples == 1:

        selected_indices = np.array(
            [0],
            dtype=int,
        )

    else:

        selected_indices = np.linspace(
            0,
            num_available - 1,
            num_real_samples,
        )

        selected_indices = np.round(
            selected_indices
        ).astype(int)

    selected_frames = [
        observed_frames[i]
        for i in selected_indices
    ]

    return (
        selected_frames,
        selected_indices.tolist(),
    )


def pad_sequence(
    selected_frames,
):
    """
    Pad the sequence to MAX_SEQUENCE_LENGTH.

    Padding repeats the last real frame so all
    paths remain valid, but the valid_mask tells
    the model which positions should be ignored.
    """

    num_real = len(
        selected_frames
    )

    if num_real == 0:
        raise ValueError(
            "Cannot pad empty sequence."
        )

    padded_frames = list(
        selected_frames
    )

    valid_mask = (
        [True] * num_real
    )

    last_frame = selected_frames[-1]

    while (
        len(padded_frames)
        < MAX_SEQUENCE_LENGTH
    ):

        padded_frames.append(
            last_frame
        )

        valid_mask.append(
            False
        )

    if (
        len(padded_frames)
        != MAX_SEQUENCE_LENGTH
    ):
        raise RuntimeError(
            "Incorrect padded sequence length."
        )

    return (
        padded_frames,
        valid_mask,
    )


def build_fraction(
    records,
    percentage,
):
    config = FRACTION_CONFIG[
        percentage
    ]

    fraction = config[
        "fraction"
    ]

    requested_samples = config[
        "num_samples"
    ]

    output = []

    for record in records:

        frames = record[
            "frames"
        ]

        total_frames = len(
            frames
        )

        observed_count = max(
            1,
            math.ceil(
                total_frames
                * fraction
            ),
        )

        observed_frames = frames[
            :observed_count
        ]

        (
            selected_frames,
            selected_indices,
        ) = uniformly_sample_frames(
            observed_frames,
            requested_samples,
        )

        (
            padded_frames,
            valid_mask,
        ) = pad_sequence(
            selected_frames
        )

        selected_frame_ids = [
            frame["frame_id"]
            for frame in selected_frames
        ]

        output.append({
            "id":
                record["id"],

            "pose_path":
                record["pose_path"],

            "ego_path":
                record["ego_path"],

            "action_label":
                record["action_label"],

            "observation_percentage":
                percentage,

            "observation_fraction":
                fraction,

            "total_action_frames":
                total_frames,

            "observed_action_frames":
                observed_count,

            "requested_temporal_samples":
                requested_samples,

            "num_real_temporal_samples":
                len(selected_frames),

            "selected_indices_within_observed":
                selected_indices,

            "selected_frame_ids":
                selected_frame_ids,

            "frames":
                padded_frames,

            "valid_mask":
                valid_mask,
        })

    return output


def process_split(
    split_name,
):
    input_path = (
        INDEX_ROOT
        / f"{split_name}.json"
    )

    with open(
        input_path,
        "r",
    ) as f:

        records = json.load(
            f
        )

    print()
    print("=" * 72)
    print(
        f"Processing {split_name}"
    )
    print("=" * 72)

    print(
        "Segments:",
        len(records),
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for percentage in (
        10,
        20,
        30,
        50,
        75,
        100,
    ):

        samples = build_fraction(
            records,
            percentage,
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
            "w",
        ) as f:

            json.dump(
                samples,
                f,
                indent=2,
            )

        real_samples = [
            sample[
                "num_real_temporal_samples"
            ]
            for sample in samples
        ]

        observed_counts = [
            sample[
                "observed_action_frames"
            ]
            for sample in samples
        ]

        requested = (
            FRACTION_CONFIG[
                percentage
            ][
                "num_samples"
            ]
        )

        print()
        print(
            f"{percentage}% observation"
        )

        print(
            "Requested temporal samples:",
            requested,
        )

        print(
            "Sequence length:",
            MAX_SEQUENCE_LENGTH,
        )

        print(
            "Real temporal samples min:",
            min(real_samples),
        )

        print(
            "Real temporal samples max:",
            max(real_samples),
        )

        print(
            "Real temporal samples mean:",
            round(
                float(
                    np.mean(
                        real_samples
                    )
                ),
                2,
            ),
        )

        print(
            "Observed frames min:",
            min(observed_counts),
        )

        print(
            "Observed frames max:",
            max(observed_counts),
        )

        print(
            "Observed frames mean:",
            round(
                float(
                    np.mean(
                        observed_counts
                    )
                ),
                2,
            ),
        )

        print(
            "Saved:",
            output_path,
        )


def main():

    process_split(
        "train"
    )

    process_split(
        "val"
    )

    print()
    print("=" * 72)
    print(
        "Masked partial indexes complete."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()