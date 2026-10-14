from pathlib import Path
import csv
import math

import numpy as np


DATA_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/intent_prediction/data/h2o"
)

OUTPUT_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/multimodal"
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


def load_hand_pose(path):
    values = np.loadtxt(
        path,
        dtype=np.float32
    ).reshape(-1)

    if len(values) != 128:
        raise ValueError(
            f"Expected 128 hand values in {path}, "
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

    left_wrist = left_hand[0].copy()
    right_wrist = right_hand[0].copy()

    # Wrist-relative normalization
    left_normalized = (
        left_hand - left_wrist
    )

    right_normalized = (
        right_hand - right_wrist
    )

    hand_feature = np.concatenate(
        [
            np.array(
                [left_valid],
                dtype=np.float32
            ),
            left_normalized.reshape(-1),

            np.array(
                [right_valid],
                dtype=np.float32
            ),
            right_normalized.reshape(-1),
        ]
    )

    return (
        hand_feature.astype(np.float32),
        left_wrist.astype(np.float32),
        right_wrist.astype(np.float32),
    )


def load_object_rt(path):
    values = np.loadtxt(
        path,
        dtype=np.float32
    ).reshape(-1)

    if len(values) != 17:
        raise ValueError(
            f"Expected 17 object RT values in {path}, "
            f"found {len(values)}"
        )

    object_id = int(values[0])

    transform = values[
        1:
    ].reshape(4, 4)

    rotation = transform[
        :3,
        :3
    ]

    translation = transform[
        :3,
        3
    ]

    object_feature = np.concatenate(
        [
            rotation.reshape(-1),
            translation
        ]
    ).astype(np.float32)

    return (
        object_id,
        object_feature,
        translation.astype(np.float32)
    )


def load_frame(
    sequence_path,
    frame_id
):
    base = (
        DATA_ROOT
        / sequence_path
        / "cam4"
    )

    frame_name = (
        f"{frame_id:06d}.txt"
    )

    hand_path = (
        base
        / "hand_pose"
        / frame_name
    )

    object_path = (
        base
        / "obj_pose_rt"
        / frame_name
    )

    if (
        not hand_path.exists()
        or not object_path.exists()
    ):
        return None

    (
        hand_feature,
        left_wrist,
        right_wrist,
    ) = load_hand_pose(
        hand_path
    )

    (
        object_id,
        object_feature,
        object_center,
    ) = load_object_rt(
        object_path
    )

    left_to_object = (
        object_center - left_wrist
    )

    right_to_object = (
        object_center - right_wrist
    )

    left_distance = np.linalg.norm(
        left_to_object
    )

    right_distance = np.linalg.norm(
        right_to_object
    )

    relative_feature = np.concatenate(
        [
            left_to_object,
            right_to_object,
            np.array(
                [
                    left_distance,
                    right_distance
                ],
                dtype=np.float32
            )
        ]
    ).astype(np.float32)

    return {
        "hand": hand_feature,
        "object": object_feature,
        "relative": relative_feature,
        "object_id": object_id
    }


def read_action_split(path):
    samples = []

    with open(path, "r") as f:
        reader = csv.DictReader(
            f,
            delimiter=" "
        )

        for row in reader:
            if not row:
                continue

            sample = {
                "id": int(
                    row["id"]
                ),
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
            }

            samples.append(
                sample
            )

    return samples


def load_action_segment(sample):
    hands = []
    objects = []
    relatives = []
    object_ids = []

    for frame_id in range(
        sample["start_act"],
        sample["end_act"] + 1
    ):
        frame = load_frame(
            sample["path"],
            frame_id
        )

        if frame is None:
            continue

        hands.append(
            frame["hand"]
        )

        objects.append(
            frame["object"]
        )

        relatives.append(
            frame["relative"]
        )

        object_ids.append(
            frame["object_id"]
        )

    if len(hands) == 0:
        return None

    return {
        "hand": np.stack(
            hands
        ),
        "object": np.stack(
            objects
        ),
        "relative": np.stack(
            relatives
        ),
        "object_ids": np.array(
            object_ids,
            dtype=np.int64
        )
    }


def resample_sequence(
    sequence,
    target_length
):
    num_frames = len(
        sequence
    )

    if num_frames == 0:
        raise ValueError(
            "Cannot resample empty sequence."
        )

    indices = np.linspace(
        0,
        num_frames - 1,
        target_length
    )

    indices = np.round(
        indices
    ).astype(int)

    return sequence[
        indices
    ]


def create_partial_segment(
    segment,
    fraction
):
    total_frames = len(
        segment["hand"]
    )

    observed_frames = max(
        1,
        math.ceil(
            total_frames * fraction
        )
    )

    hand = segment[
        "hand"
    ][:observed_frames]

    object_features = segment[
        "object"
    ][:observed_frames]

    relative = segment[
        "relative"
    ][:observed_frames]

    hand = resample_sequence(
        hand,
        SEQUENCE_LENGTH
    )

    object_features = resample_sequence(
        object_features,
        SEQUENCE_LENGTH
    )

    relative = resample_sequence(
        relative,
        SEQUENCE_LENGTH
    )

    return (
        hand,
        object_features,
        relative,
        observed_frames
    )


def build_split(split_name):
    split_path = (
        DATA_ROOT
        / "label_split"
        / f"action_{split_name}.txt"
    )

    samples = read_action_split(
        split_path
    )

    print()
    print("=" * 70)
    print(
        f"Building multimodal split: {split_name}"
    )
    print("=" * 70)

    print(
        "Segments:",
        len(samples)
    )

    datasets = {}

    for fraction in (
        OBSERVATION_FRACTIONS
    ):
        datasets[fraction] = {
            "hand": [],
            "object": [],
            "relative": [],
            "y": [],
            "object_id": [],
            "ids": [],
            "paths": [],
            "observed_frames": [],
        }

    skipped = 0

    for index, sample in enumerate(
        samples
    ):
        segment = load_action_segment(
            sample
        )

        if segment is None:
            skipped += 1
            continue

        unique_objects = np.unique(
            segment["object_ids"]
        )

        if len(unique_objects) != 1:
            print(
                "Warning: multiple object IDs in",
                sample["path"],
                unique_objects
            )

        object_id = int(
            segment["object_ids"][0]
        )

        for fraction in (
            OBSERVATION_FRACTIONS
        ):
            (
                hand,
                object_features,
                relative,
                observed_frames
            ) = create_partial_segment(
                segment,
                fraction
            )

            data = datasets[
                fraction
            ]

            data["hand"].append(
                hand
            )

            data["object"].append(
                object_features
            )

            data["relative"].append(
                relative
            )

            data["y"].append(
                sample["action_label"]
            )

            data["object_id"].append(
                object_id
            )

            data["ids"].append(
                sample["id"]
            )

            data["paths"].append(
                sample["path"]
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

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    for fraction, data in (
        datasets.items()
    ):
        X_hand = np.stack(
            data["hand"]
        ).astype(
            np.float32
        )

        X_object = np.stack(
            data["object"]
        ).astype(
            np.float32
        )

        X_relative = np.stack(
            data["relative"]
        ).astype(
            np.float32
        )

        y = np.array(
            data["y"],
            dtype=np.int64
        )

        object_id = np.array(
            data["object_id"],
            dtype=np.int64
        )

        ids = np.array(
            data["ids"],
            dtype=np.int64
        )

        paths = np.array(
            data["paths"]
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
            X_hand=X_hand,
            X_object=X_object,
            X_relative=X_relative,
            y=y,
            object_id=object_id,
            ids=ids,
            paths=paths,
            observed_frames=observed_frames,
            observation_fraction=fraction,
        )

        print()
        print(
            "Saved:",
            output_path
        )

        print(
            "Hand:",
            X_hand.shape
        )

        print(
            "Object:",
            X_object.shape
        )

        print(
            "Relative:",
            X_relative.shape
        )

        print(
            "Labels:",
            y.shape
        )

    print()
    print(
        "Skipped:",
        skipped
    )


if __name__ == "__main__":
    print(
        "Building H2O multimodal intent dataset"
    )

    build_split(
        "train"
    )

    build_split(
        "val"
    )

    print()
    print(
        "Multimodal dataset complete."
    )