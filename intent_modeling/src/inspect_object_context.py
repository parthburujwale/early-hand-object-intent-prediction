from pathlib import Path
import numpy as np


DATA_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/intent_prediction/data/h2o"
)


SEQUENCE = "subject1/h1/0"

FRAMES = [
    0,
    30,
    60,
    62,
    64,
    90,
    127,
    130,
    133,
    180,
    255,
    300,
    317,
    370,
    417,
    440,
]


def inspect_frame(frame_id):

    frame_name = f"{frame_id:06d}.txt"

    base = (
        DATA_ROOT
        / SEQUENCE
        / "cam4"
    )

    obj_path = (
        base
        / "obj_pose"
        / frame_name
    )

    obj_rt_path = (
        base
        / "obj_pose_rt"
        / frame_name
    )

    action_path = (
        base
        / "action_label"
        / frame_name
    )

    verb_path = (
        base
        / "verb_label"
        / frame_name
    )

    print()
    print("=" * 60)
    print("FRAME:", frame_id)
    print("=" * 60)

    if action_path.exists():
        print(
            "Action:",
            action_path.read_text().strip()
        )

    if verb_path.exists():
        print(
            "Verb:",
            verb_path.read_text().strip()
        )

    if obj_path.exists():

        values = np.loadtxt(
            obj_path,
            dtype=np.float32
        ).reshape(-1)

        object_id = int(
            values[0]
        )

        points = values[1:].reshape(
            -1,
            3
        )

        print(
            "Object ID:",
            object_id
        )

        print(
            "Object points:",
            points.shape
        )

        print(
            "Object center:",
            points.mean(axis=0)
        )

    else:
        print(
            "Object pose: MISSING"
        )

    if obj_rt_path.exists():

        rt = np.loadtxt(
            obj_rt_path,
            dtype=np.float32
        ).reshape(-1)

        print(
            "Object RT values:",
            len(rt)
        )

        print(
            "Object RT:",
            rt
        )


if __name__ == "__main__":

    print(
        "Inspecting object context for:"
    )

    print(
        SEQUENCE
    )

    for frame in FRAMES:
        inspect_frame(
            frame
        )