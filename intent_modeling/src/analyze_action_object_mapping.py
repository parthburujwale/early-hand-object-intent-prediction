from pathlib import Path
from collections import defaultdict, Counter

import numpy as np


DATA_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/intent_prediction/data/h2o"
)


def read_split(path):
    rows = []

    with open(path, "r") as f:
        header = f.readline()

        for line in f:
            parts = line.strip().split()

            if not parts:
                continue

            rows.append({
                "id": int(parts[0]),
                "path": parts[1],
                "action": int(parts[2]),
                "start": int(parts[3]),
                "end": int(parts[4]),
            })

    return rows


def get_object_id(sequence_path, frame_id):
    path = (
        DATA_ROOT
        / sequence_path
        / "cam4"
        / "obj_pose"
        / f"{frame_id:06d}.txt"
    )

    if not path.exists():
        return None

    values = np.loadtxt(
        path,
        dtype=np.float32
    ).reshape(-1)

    return int(values[0])


def analyze(split_name):
    split_path = (
        DATA_ROOT
        / "label_split"
        / f"action_{split_name}.txt"
    )

    rows = read_split(split_path)

    action_to_objects = defaultdict(Counter)
    object_to_actions = defaultdict(Counter)

    inconsistent_segments = 0

    for row in rows:
        start_object = get_object_id(
            row["path"],
            row["start"]
        )

        end_object = get_object_id(
            row["path"],
            row["end"]
        )

        if (
            start_object is not None
            and end_object is not None
            and start_object != end_object
        ):
            inconsistent_segments += 1

        if start_object is None:
            continue

        action_to_objects[
            row["action"]
        ][start_object] += 1

        object_to_actions[
            start_object
        ][row["action"]] += 1

    print()
    print("=" * 70)
    print(split_name.upper())
    print("=" * 70)

    print(
        "Segments:",
        len(rows)
    )

    print(
        "Object changed within action segment:",
        inconsistent_segments
    )

    print()
    print("ACTION -> OBJECT")

    for action in sorted(action_to_objects):
        print(
            f"Action {action:2d}: "
            f"{dict(action_to_objects[action])}"
        )

    print()
    print("OBJECT -> ACTIONS")

    for obj in sorted(object_to_actions):
        print(
            f"Object {obj:2d}: "
            f"{dict(object_to_actions[obj])}"
        )


if __name__ == "__main__":
    analyze("train")
    analyze("val")