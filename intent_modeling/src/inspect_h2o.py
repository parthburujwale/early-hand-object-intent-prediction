from pathlib import Path
import numpy as np


DATA_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/intent_prediction/data/h2o"
)


def load_hand_pose(path):
    values = np.loadtxt(path, dtype=np.float32)

    left_valid = values[0]

    left_hand = values[
        1:64
    ].reshape(21, 3)

    right_valid = values[64]

    right_hand = values[
        65:128
    ].reshape(21, 3)

    return (
        left_valid,
        left_hand,
        right_valid,
        right_hand,
    )


def load_object_pose(path):
    values = np.loadtxt(path, dtype=np.float32)

    object_class = int(values[0])

    object_points = values[1:].reshape(
        -1,
        3
    )

    return (
        object_class,
        object_points
    )


hand_path = (
    DATA_ROOT
    / "subject1/h1/0/cam4/hand_pose/000000.txt"
)

object_path = (
    DATA_ROOT
    / "subject1/h1/0/cam4/obj_pose/000000.txt"
)

action_path = (
    DATA_ROOT
    / "subject1/h1/0/cam4/action_label/000000.txt"
)

verb_path = (
    DATA_ROOT
    / "subject1/h1/0/cam4/verb_label/000000.txt"
)


(
    left_valid,
    left_hand,
    right_valid,
    right_hand,
) = load_hand_pose(hand_path)


(
    object_class,
    object_points,
) = load_object_pose(object_path)


action_label = int(
    action_path.read_text().strip()
)

verb_label = int(
    verb_path.read_text().strip()
)


print("LEFT HAND")
print("Valid:", left_valid)
print("Shape:", left_hand.shape)
print("Wrist:", left_hand[0])

print()

print("RIGHT HAND")
print("Valid:", right_valid)
print("Shape:", right_hand.shape)
print("Wrist:", right_hand[0])

print()

print("OBJECT")
print("Class:", object_class)
print("Shape:", object_points.shape)
print(
    "Approx center:",
    object_points.mean(axis=0)
)

print()

print("LABELS")
print("Action:", action_label)
print("Verb:", verb_label)