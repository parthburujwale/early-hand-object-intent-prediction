from pathlib import Path
import csv
import json


POSE_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/intent_prediction/data/h2o"
)

EGO_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/intent_prediction/data/h2o_ego"
)

OUTPUT_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/rgbd_hand_index"
)


def read_split(split_name):
    split_path = (
        POSE_ROOT
        / "label_split"
        / f"action_{split_name}.txt"
    )

    rows = []

    with open(split_path, "r") as f:
        reader = csv.DictReader(
            f,
            delimiter=" "
        )

        for row in reader:
            if not row:
                continue

            rows.append({
                "id": int(row["id"]),
                "path": row["path"],
                "action_label": int(row["action_label"]),
                "start_act": int(row["start_act"]),
                "end_act": int(row["end_act"]),
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
            })

    return rows


def convert_sequence_path(pose_path):
    """
    pose path example:
        subject1/h1/0

    ego path becomes:
        subject1_ego/h1/0
    """

    parts = pose_path.split("/")

    subject = parts[0]
    rest = "/".join(parts[1:])

    return f"{subject}_ego/{rest}"


def build_segment_record(sample):
    ego_sequence = convert_sequence_path(
        sample["path"]
    )

    cam4 = (
        EGO_ROOT
        / ego_sequence
        / "cam4"
    )

    rgb256_dir = cam4 / "rgb256"
    rgb_dir = cam4 / "rgb"
    depth_dir = cam4 / "depth"
    hand_dir = cam4 / "hand_pose"

    frames = []

    missing = []

    for frame_id in range(
        sample["start_act"],
        sample["end_act"] + 1
    ):
        frame_name = f"{frame_id:06d}"

        rgb256 = (
            rgb256_dir
            / f"{frame_name}.jpg"
        )

        rgb = (
            rgb_dir
            / f"{frame_name}.png"
        )

        depth = (
            depth_dir
            / f"{frame_name}.png"
        )

        hand = (
            hand_dir
            / f"{frame_name}.txt"
        )

        exists = {
            "rgb256": rgb256.exists(),
            "rgb": rgb.exists(),
            "depth": depth.exists(),
            "hand_pose": hand.exists(),
        }

        if not all(exists.values()):
            missing.append({
                "frame": frame_id,
                "exists": exists
            })

            continue

        frames.append({
            "frame_id": frame_id,
            "rgb256": str(rgb256),
            "rgb": str(rgb),
            "depth": str(depth),
            "hand_pose": str(hand),
        })

    if len(frames) == 0:
        return None, missing

    record = {
        "id": sample["id"],
        "pose_path": sample["path"],
        "ego_path": ego_sequence,
        "action_label": sample["action_label"],
        "start_act": sample["start_act"],
        "end_act": sample["end_act"],
        "num_expected_frames": (
            sample["end_act"]
            - sample["start_act"]
            + 1
        ),
        "num_available_frames": len(frames),
        "frames": frames,
    }

    return record, missing


def build_split(split_name):
    samples = read_split(
        split_name
    )

    print()
    print("=" * 70)
    print(
        f"Building {split_name} RGB-D + hand index"
    )
    print("=" * 70)

    print(
        "Action segments:",
        len(samples)
    )

    records = []

    missing_segments = []
    missing_frame_count = 0

    for i, sample in enumerate(
        samples
    ):
        record, missing = (
            build_segment_record(
                sample
            )
        )

        if record is not None:
            records.append(
                record
            )

        if missing:
            missing_segments.append({
                "id": sample["id"],
                "path": sample["path"],
                "missing": missing,
            })

            missing_frame_count += len(
                missing
            )

        if (
            (i + 1) % 100 == 0
            or i + 1 == len(samples)
        ):
            print(
                f"Processed "
                f"{i + 1}/"
                f"{len(samples)}"
            )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_ROOT
        / f"{split_name}.json"
    )

    with open(
        output_path,
        "w"
    ) as f:
        json.dump(
            records,
            f,
            indent=2
        )

    missing_path = (
        OUTPUT_ROOT
        / f"{split_name}_missing.json"
    )

    with open(
        missing_path,
        "w"
    ) as f:
        json.dump(
            missing_segments,
            f,
            indent=2
        )

    total_available_frames = sum(
        r["num_available_frames"]
        for r in records
    )

    print()
    print(
        "Saved:",
        output_path
    )

    print(
        "Segments indexed:",
        len(records)
    )

    print(
        "Available frames:",
        total_available_frames
    )

    print(
        "Segments with missing frames:",
        len(missing_segments)
    )

    print(
        "Total missing frames:",
        missing_frame_count
    )

    print(
        "Missing report:",
        missing_path
    )


if __name__ == "__main__":
    build_split("train")
    build_split("val")