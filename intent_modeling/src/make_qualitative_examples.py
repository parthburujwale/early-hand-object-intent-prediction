from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling"
)

EVAL_ROOT = (
    ROOT
    / "outputs"
    / "final_evaluation"
)

INDEX_PATH = (
    ROOT
    / "outputs"
    / "masked_partial_index"
    / "val_010.json"
)

OUTPUT_ROOT = (
    EVAL_ROOT
    / "qualitative_examples"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# H2O action mapping
# ============================================================

ACTIONS = [
    ("grab", "book"),
    ("grab", "espresso"),
    ("grab", "lotion"),
    ("grab", "spray"),
    ("grab", "milk"),
    ("grab", "cocoa"),
    ("grab", "chips"),
    ("grab", "cappuccino"),

    ("place", "book"),
    ("place", "espresso"),
    ("place", "lotion"),
    ("place", "spray"),
    ("place", "milk"),
    ("place", "cocoa"),
    ("place", "chips"),
    ("place", "cappuccino"),

    ("open", "lotion"),
    ("open", "milk"),
    ("open", "chips"),

    ("close", "lotion"),
    ("close", "milk"),
    ("close", "chips"),

    ("pour", "milk"),

    ("take out", "espresso"),
    ("take out", "cocoa"),
    ("take out", "chips"),
    ("take out", "cappuccino"),

    ("put in", "espresso"),
    ("put in", "cocoa"),
    ("put in", "cappuccino"),

    ("apply", "lotion"),
    ("apply", "spray"),

    ("read", "book"),
    ("read", "espresso"),

    ("spray", "spray"),

    ("squeeze", "lotion"),
]

ACTION_NAMES = [
    f"{verb} {obj}"
    for verb, obj in ACTIONS
]

VERBS = [
    verb
    for verb, _ in ACTIONS
]

OBJECTS = [
    obj
    for _, obj in ACTIONS
]


# ============================================================
# Experiment setup
# ============================================================

MODELS = [
    "rgb_only",
    "position_cross_attention",
    "motion_cross_attention",
]

MODEL_NAMES = {
    "rgb_only":
        "RGB only",

    "position_cross_attention":
        "RGB + hand position",

    "motion_cross_attention":
        "RGB + hand position + velocity",
}

SEEDS = [
    10,
    42,
    100
]

FRACTION = 10


# ============================================================
# Load validation index
# ============================================================

with open(
    INDEX_PATH,
    "r"
) as f:

    samples = json.load(f)


print(
    f"Loaded {len(samples)} "
    f"validation samples."
)


# ============================================================
# Load predictions
# ============================================================

predictions = {}

labels_reference = None


for model_name in MODELS:

    predictions[
        model_name
    ] = {}

    for seed in SEEDS:

        path = (
            EVAL_ROOT
            / model_name
            / f"obs_{FRACTION:03d}"
            / f"seed_{seed}"
            / "predictions.npz"
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        data = np.load(
            path
        )

        labels = data[
            "labels"
        ].astype(int)

        preds = data[
            "predictions"
        ].astype(int)

        if labels_reference is None:

            labels_reference = labels

        else:

            if not np.array_equal(
                labels_reference,
                labels
            ):

                raise RuntimeError(
                    "Validation label order "
                    "does not match between runs."
                )

        predictions[
            model_name
        ][
            seed
        ] = preds


# ============================================================
# Verify masked index order
# ============================================================

if len(samples) != len(
    labels_reference
):

    raise RuntimeError(
        "Index and prediction lengths differ: "
        f"{len(samples)} vs "
        f"{len(labels_reference)}"
    )


for i, sample in enumerate(
    samples
):

    expected_action_label = (
        int(
            labels_reference[i]
        )
        + 1
    )

    index_action_label = int(
        sample[
            "action_label"
        ]
    )

    if (
        expected_action_label
        != index_action_label
    ):

        raise RuntimeError(
            "Sample ordering mismatch at "
            f"index {i}: "
            f"predictions say action "
            f"{expected_action_label}, "
            f"index says "
            f"{index_action_label}"
        )


print(
    "Prediction ordering matches "
    "masked validation index."
)


# ============================================================
# Utilities
# ============================================================

def majority_prediction(
    model_name,
    sample_index
):

    values = [
        int(
            predictions[
                model_name
            ][seed][
                sample_index
            ]
        )
        for seed in SEEDS
    ]

    counter = Counter(
        values
    )

    most_common = (
        counter.most_common()
    )

    majority_class = (
        most_common[0][0]
    )

    majority_count = (
        most_common[0][1]
    )

    return (
        majority_class,
        majority_count,
        values
    )


def correct_seed_count(
    model_name,
    sample_index,
    true_class
):

    return sum(
        int(
            predictions[
                model_name
            ][seed][
                sample_index
            ]
        )
        == true_class

        for seed in SEEDS
    )


def same_object_wrong_verb(
    true_class,
    predicted_class
):

    return (
        OBJECTS[
            true_class
        ]
        ==
        OBJECTS[
            predicted_class
        ]
        and
        VERBS[
            true_class
        ]
        !=
        VERBS[
            predicted_class
        ]
    )


# ============================================================
# Rank interesting examples
# ============================================================

candidate_rows = []


for sample_index in range(
    len(samples)
):

    true_class = int(
        labels_reference[
            sample_index
        ]
    )

    rgb_class, rgb_majority, rgb_all = (
        majority_prediction(
            "rgb_only",
            sample_index
        )
    )

    pos_class, pos_majority, pos_all = (
        majority_prediction(
            "position_cross_attention",
            sample_index
        )
    )

    motion_class, motion_majority, motion_all = (
        majority_prediction(
            "motion_cross_attention",
            sample_index
        )
    )

    rgb_correct_count = (
        correct_seed_count(
            "rgb_only",
            sample_index,
            true_class
        )
    )

    pos_correct_count = (
        correct_seed_count(
            "position_cross_attention",
            sample_index,
            true_class
        )
    )

    motion_correct_count = (
        correct_seed_count(
            "motion_cross_attention",
            sample_index,
            true_class
        )
    )

    rgb_wrong = (
        rgb_correct_count
        <= 1
    )

    pos_wrong = (
        pos_correct_count
        <= 1
    )

    motion_correct = (
        motion_correct_count
        >= 2
    )

    position_semantic_error = (
        same_object_wrong_verb(
            true_class,
            pos_class
        )
        if pos_class != true_class
        else False
    )

    rgb_semantic_error = (
        same_object_wrong_verb(
            true_class,
            rgb_class
        )
        if rgb_class != true_class
        else False
    )

    # Higher score = stronger qualitative example.
    score = 0

    if motion_correct:
        score += 5

    if rgb_wrong:
        score += 3

    if pos_wrong:
        score += 4

    if rgb_semantic_error:
        score += 2

    if position_semantic_error:
        score += 4

    # Reward unanimous motion correctness.
    score += motion_correct_count

    # Reward clear wrong consensus in baselines.
    if rgb_majority >= 2:
        score += 1

    if pos_majority >= 2:
        score += 1

    candidate_rows.append(
        {
            "sample_index":
                sample_index,

            "dataset_id":
                samples[
                    sample_index
                ].get(
                    "id",
                    sample_index
                ),

            "true_class_index":
                true_class,

            "true_action_label":
                true_class + 1,

            "true_action":
                ACTION_NAMES[
                    true_class
                ],

            "rgb_majority_action":
                ACTION_NAMES[
                    rgb_class
                ],

            "position_majority_action":
                ACTION_NAMES[
                    pos_class
                ],

            "motion_majority_action":
                ACTION_NAMES[
                    motion_class
                ],

            "rgb_correct_seeds":
                rgb_correct_count,

            "position_correct_seeds":
                pos_correct_count,

            "motion_correct_seeds":
                motion_correct_count,

            "rgb_same_object_wrong_verb":
                rgb_semantic_error,

            "position_same_object_wrong_verb":
                position_semantic_error,

            "rgb_predictions":
                " | ".join(
                    ACTION_NAMES[x]
                    for x in rgb_all
                ),

            "position_predictions":
                " | ".join(
                    ACTION_NAMES[x]
                    for x in pos_all
                ),

            "motion_predictions":
                " | ".join(
                    ACTION_NAMES[x]
                    for x in motion_all
                ),

            "score":
                score,
        }
    )


candidates = pd.DataFrame(
    candidate_rows
)


# ============================================================
# Preferred examples:
#
# Motion correct in >=2 seeds,
# RGB wrong in >=2 seeds,
# position wrong in >=2 seeds.
# ============================================================

strong = candidates[
    (
        candidates[
            "motion_correct_seeds"
        ] >= 2
    )
    &
    (
        candidates[
            "rgb_correct_seeds"
        ] <= 1
    )
    &
    (
        candidates[
            "position_correct_seeds"
        ] <= 1
    )
].copy()


strong = strong.sort_values(
    [
        "score",
        "motion_correct_seeds"
    ],
    ascending=False
)


# If there are fewer than 6 very strong examples,
# fall back to all motion-correct / RGB-wrong cases.

if len(strong) < 6:

    fallback = candidates[
        (
            candidates[
                "motion_correct_seeds"
            ] >= 2
        )
        &
        (
            candidates[
                "rgb_correct_seeds"
            ] <= 1
        )
    ].copy()

    fallback = fallback.sort_values(
        [
            "score",
            "motion_correct_seeds"
        ],
        ascending=False
    )

    strong = fallback


selected = strong.head(
    8
).copy()


# ============================================================
# Save candidate rankings
# ============================================================

candidates.sort_values(
    "score",
    ascending=False
).to_csv(
    OUTPUT_ROOT
    / "all_ranked_candidates.csv",
    index=False
)


selected.to_csv(
    OUTPUT_ROOT
    / "selected_qualitative_examples.csv",
    index=False
)


print()
print("=" * 100)
print(
    "SELECTED QUALITATIVE EXAMPLES"
)
print("=" * 100)

print(
    selected[
        [
            "sample_index",
            "true_action",
            "rgb_majority_action",
            "position_majority_action",
            "motion_majority_action",
            "rgb_correct_seeds",
            "position_correct_seeds",
            "motion_correct_seeds",
            "score",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Get real frames only
# ============================================================

def get_real_frames(
    sample
):

    frames = sample[
        "frames"
    ]

    valid_mask = sample[
        "valid_mask"
    ]

    real_frames = [
        frame
        for frame, valid
        in zip(
            frames,
            valid_mask
        )
        if bool(valid)
    ]

    return real_frames


# ============================================================
# Create one visualization per example
# ============================================================

for rank, row in enumerate(
    selected.itertuples(),
    start=1
):

    sample_index = int(
        row.sample_index
    )

    sample = samples[
        sample_index
    ]

    real_frames = (
        get_real_frames(
            sample
        )
    )

    num_frames = len(
        real_frames
    )

    if num_frames == 0:

        print(
            f"Skipping sample "
            f"{sample_index}: "
            "no valid frames."
        )

        continue

    fig, axes = plt.subplots(
        1,
        num_frames,
        figsize=(
            4 * num_frames,
            5
        )
    )

    if num_frames == 1:

        axes = [
            axes
        ]

    for ax, frame in zip(
        axes,
        real_frames
    ):

        image_path = Path(
            frame[
                "rgb256"
            ]
        )

        if not image_path.exists():

            image_path = Path(
                frame[
                    "rgb"
                ]
            )

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        ax.imshow(
            image
        )

        ax.set_title(
            f"Frame {frame['frame_id']}"
        )

        ax.axis(
            "off"
        )

    true_name = (
        row.true_action
    )

    rgb_name = (
        row.rgb_majority_action
    )

    pos_name = (
        row.position_majority_action
    )

    motion_name = (
        row.motion_majority_action
    )

    title = (
        f"True: {true_name}\n"
        f"RGB: {rgb_name} "
        f"({row.rgb_correct_seeds}/3 correct)   |   "
        f"Position: {pos_name} "
        f"({row.position_correct_seeds}/3 correct)   |   "
        f"Position + velocity: {motion_name} "
        f"({row.motion_correct_seeds}/3 correct)"
    )

    fig.suptitle(
        title,
        fontsize=13
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.88
        ]
    )

    output_path = (
        OUTPUT_ROOT
        / (
            f"example_{rank:02d}_"
            f"sample_{sample_index:03d}.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# Create a compact multi-example figure
# ============================================================

top_for_grid = selected.head(
    4
)


if len(
    top_for_grid
) > 0:

    # Four real frames at 10% observation.
    num_display_frames = 4

    fig, axes = plt.subplots(
        len(top_for_grid),
        num_display_frames,
        figsize=(
            14,
            3.7
            * len(top_for_grid)
        )
    )

    if len(
        top_for_grid
    ) == 1:

        axes = np.expand_dims(
            axes,
            axis=0
        )

    for row_idx, row in enumerate(
        top_for_grid.itertuples()
    ):

        sample = samples[
            int(
                row.sample_index
            )
        ]

        real_frames = (
            get_real_frames(
                sample
            )
        )

        for col_idx in range(
            num_display_frames
        ):

            ax = axes[
                row_idx,
                col_idx
            ]

            if col_idx < len(
                real_frames
            ):

                frame = real_frames[
                    col_idx
                ]

                image_path = Path(
                    frame[
                        "rgb256"
                    ]
                )

                if not image_path.exists():

                    image_path = Path(
                        frame[
                            "rgb"
                        ]
                    )

                image = Image.open(
                    image_path
                ).convert(
                    "RGB"
                )

                ax.imshow(
                    image
                )

                ax.set_title(
                    f"Frame "
                    f"{frame['frame_id']}",
                    fontsize=9
                )

            ax.axis(
                "off"
            )

        prediction_text = (
            f"True: {row.true_action}\n"
            f"RGB: {row.rgb_majority_action} | "
            f"Position: "
            f"{row.position_majority_action} | "
            f"Motion: "
            f"{row.motion_majority_action}"
        )

        axes[
            row_idx,
            0
        ].set_ylabel(
            prediction_text,
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=100
        )

    fig.suptitle(
        "Examples Where Hand Motion Improves "
        "Early Intent Prediction (10% Observation)",
        fontsize=15
    )

    plt.tight_layout(
        rect=[
            0.15,
            0,
            1,
            0.96
        ]
    )

    grid_path = (
        OUTPUT_ROOT
        / "qualitative_examples_grid.png"
    )

    plt.savefig(
        grid_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Saved grid:",
        grid_path
    )


print()
print("=" * 100)
print(
    "QUALITATIVE ANALYSIS COMPLETE"
)
print("=" * 100)

print(
    "Outputs:",
    OUTPUT_ROOT
)