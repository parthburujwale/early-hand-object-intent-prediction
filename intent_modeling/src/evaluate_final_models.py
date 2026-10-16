from pathlib import Path
import argparse
import csv
import importlib.util
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# =========================================================
# Project paths
# =========================================================

ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling"
)

SRC_ROOT = ROOT / "src"

OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "final_evaluation"
)

INDEX_ROOT = (
    ROOT
    / "outputs"
    / "masked_partial_index"
)


# =========================================================
# Dynamically import our training files
# =========================================================

def import_module_from_file(
    module_name,
    file_path
):

    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            file_path
        )
    )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


rgb_module = import_module_from_file(
    "rgb_masked_module",
    SRC_ROOT
    / "train_rgb_only_vit_masked.py"
)

motion_module = import_module_from_file(
    "motion_cross_module",
    SRC_ROOT
    / "train_motion_cross_attention_transformer.py"
)

position_module = import_module_from_file(
    "position_cross_module",
    SRC_ROOT
    / "train_position_cross_attention_transformer.py"
)

hand_module = import_module_from_file(
    "hand_motion_module",
    SRC_ROOT
    / "train_hand_motion_only.py"
)


# =========================================================
# Checkpoint utility
# =========================================================

def extract_state_dict(
    checkpoint
):

    if isinstance(
        checkpoint,
        dict
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            return checkpoint[
                "model_state_dict"
            ]

        if (
            "state_dict"
            in checkpoint
        ):

            return checkpoint[
                "state_dict"
            ]

    return checkpoint


# =========================================================
# Model creation
# =========================================================

def build_model(
    model_name,
    device
):

    if model_name == "rgb_only":

        model = (
            rgb_module
            .RGBTemporalTransformer()
        )

    elif (
        model_name
        == "hand_motion_only"
    ):

        model = (
            hand_module
            .HandMotionOnlyTransformer()
        )

    elif (
        model_name
        == "position_cross_attention"
    ):

        model = (
            position_module
            .PositionCrossAttentionTransformer()
        )

    elif (
        model_name
        == "motion_cross_attention"
    ):

        model = (
            motion_module
            .MotionCrossAttentionTransformer()
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model.to(
        device
    )


# =========================================================
# Checkpoint paths
# =========================================================

def get_checkpoint_path(
    model_name,
    fraction,
    seed
):

    fraction_string = (
        f"{fraction:03d}"
    )

    if model_name == "rgb_only":

        return (
            ROOT
            / "outputs"
            / "checkpoints"
            / "rgb_only_vit_masked"
            / (
                f"rgb_masked_"
                f"{fraction_string}_"
                f"seed{seed}.pt"
            )
        )

    if (
        model_name
        == "hand_motion_only"
    ):

        return (
            ROOT
            / "outputs"
            / "checkpoints"
            / "hand_motion_only"
            / (
                f"hand_motion_"
                f"{fraction_string}_"
                f"seed{seed}.pt"
            )
        )

    if (
        model_name
        == "position_cross_attention"
    ):

        return (
            ROOT
            / "outputs"
            / "checkpoints"
            / "position_cross_attention"
            / (
                f"position_cross_"
                f"{fraction_string}_"
                f"seed{seed}.pt"
            )
        )

    if (
        model_name
        == "motion_cross_attention"
    ):

        return (
            ROOT
            / "outputs"
            / "checkpoints"
            / "motion_cross_attention"
            / (
                f"motion_cross_"
                f"{fraction_string}_"
                f"seed{seed}.pt"
            )
        )

    raise ValueError(
        model_name
    )


# =========================================================
# Dataset construction
# =========================================================

def build_dataset(
    model_name,
    fraction
):

    val_path = (
        INDEX_ROOT
        / (
            f"val_"
            f"{fraction:03d}.json"
        )
    )

    if model_name == "rgb_only":

        weights = (
            rgb_module
            .ViT_B_16_Weights
            .IMAGENET1K_V1
        )

        transform = (
            weights.transforms()
        )

        dataset = (
            rgb_module
            .RGBMaskedDataset(
                val_path,
                transform
            )
        )

    elif (
        model_name
        == "hand_motion_only"
    ):

        dataset = (
            hand_module
            .HandMotionDataset(
                val_path
            )
        )

    elif (
        model_name
        == "position_cross_attention"
    ):

        weights = (
            position_module
            .ViT_B_16_Weights
            .IMAGENET1K_V1
        )

        transform = (
            weights.transforms()
        )

        dataset = (
            position_module
            .PositionVisualHandDataset(
                val_path,
                transform
            )
        )

    elif (
        model_name
        == "motion_cross_attention"
    ):

        weights = (
            motion_module
            .ViT_B_16_Weights
            .IMAGENET1K_V1
        )

        transform = (
            weights.transforms()
        )

        dataset = (
            motion_module
            .MotionVisualHandDataset(
                val_path,
                transform
            )
        )

    else:

        raise ValueError(
            model_name
        )

    return dataset


# =========================================================
# Forward pass adapter
# =========================================================

def forward_batch(
    model_name,
    model,
    batch,
    device
):

    if model_name == "rgb_only":

        (
            images,
            valid_mask,
            labels
        ) = batch

        images = images.to(
            device,
            non_blocking=True
        )

        valid_mask = valid_mask.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        logits = model(
            images,
            valid_mask
        )

    elif (
        model_name
        == "hand_motion_only"
    ):

        (
            hand_positions,
            hand_velocity,
            valid_mask,
            labels
        ) = batch

        hand_positions = (
            hand_positions.to(
                device,
                non_blocking=True
            )
        )

        hand_velocity = (
            hand_velocity.to(
                device,
                non_blocking=True
            )
        )

        valid_mask = valid_mask.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        logits = model(
            hand_positions,
            hand_velocity,
            valid_mask
        )

    elif (
        model_name
        == "position_cross_attention"
    ):

        (
            images,
            hand_positions,
            valid_mask,
            labels
        ) = batch

        images = images.to(
            device,
            non_blocking=True
        )

        hand_positions = (
            hand_positions.to(
                device,
                non_blocking=True
            )
        )

        valid_mask = valid_mask.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        logits = model(
            images,
            hand_positions,
            valid_mask
        )

    elif (
        model_name
        == "motion_cross_attention"
    ):

        (
            images,
            hand_positions,
            hand_velocity,
            valid_mask,
            labels
        ) = batch

        images = images.to(
            device,
            non_blocking=True
        )

        hand_positions = (
            hand_positions.to(
                device,
                non_blocking=True
            )
        )

        hand_velocity = (
            hand_velocity.to(
                device,
                non_blocking=True
            )
        )

        valid_mask = valid_mask.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        logits = model(
            images,
            hand_positions,
            hand_velocity,
            valid_mask
        )

    else:

        raise ValueError(
            model_name
        )

    return (
        logits,
        labels
    )


# =========================================================
# Evaluate one checkpoint
# =========================================================

def evaluate_checkpoint(
    model_name,
    fraction,
    seed,
    device,
    batch_size,
    num_workers
):

    checkpoint_path = (
        get_checkpoint_path(
            model_name,
            fraction,
            seed
        )
    )

    if not checkpoint_path.exists():

        print(
            f"SKIPPING - checkpoint missing:"
        )

        print(
            checkpoint_path
        )

        return None

    print()
    print("=" * 80)

    print(
        f"MODEL: {model_name}"
    )

    print(
        f"OBSERVATION: {fraction}%"
    )

    print(
        f"SEED: {seed}"
    )

    print(
        f"CHECKPOINT: {checkpoint_path}"
    )

    print("=" * 80)

    dataset = build_dataset(
        model_name,
        fraction
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = build_model(
        model_name,
        device
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    state_dict = (
        extract_state_dict(
            checkpoint
        )
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    all_labels = []

    all_predictions = []

    all_top3 = []

    all_logits = []

    with torch.inference_mode():

        for batch in loader:

            (
                logits,
                labels
            ) = forward_batch(
                model_name,
                model,
                batch,
                device
            )

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            top3 = (
                torch.topk(
                    logits,
                    k=3,
                    dim=1
                )
                .indices
            )

            all_labels.append(
                labels.cpu()
            )

            all_predictions.append(
                predictions.cpu()
            )

            all_top3.append(
                top3.cpu()
            )

            all_logits.append(
                logits.cpu()
            )

    labels = torch.cat(
        all_labels
    ).numpy()

    predictions = torch.cat(
        all_predictions
    ).numpy()

    top3_predictions = torch.cat(
        all_top3
    ).numpy()

    logits = torch.cat(
        all_logits
    ).numpy()

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    top1_accuracy = (
        accuracy_score(
            labels,
            predictions
        )
    )

    top3_accuracy = np.mean(
        np.any(
            top3_predictions
            == labels[:, None],
            axis=1
        )
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    (
        class_precision,
        class_recall,
        class_f1,
        class_support
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(36),
        average=None,
        zero_division=0
    )

    cm = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(36)
    )

    print(
        f"Top-1: "
        f"{top1_accuracy * 100:.2f}%"
    )

    print(
        f"Top-3: "
        f"{top3_accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision: "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall: "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )

    # -----------------------------------------------------
    # Output directory
    # -----------------------------------------------------

    run_output_dir = (
        OUTPUT_ROOT
        / model_name
        / (
            f"obs_{fraction:03d}"
        )
        / (
            f"seed_{seed}"
        )
    )

    run_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Save raw arrays
    # -----------------------------------------------------

    np.savez_compressed(
        run_output_dir
        / "predictions.npz",

        labels=labels,

        predictions=predictions,

        top3_predictions=(
            top3_predictions
        ),

        logits=logits,

        sample_indices=np.arange(
            len(labels)
        )
    )

    # -----------------------------------------------------
    # Save summary metrics
    # -----------------------------------------------------

    with open(
        run_output_dir
        / "metrics.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "model",
                "fraction",
                "seed",
                "top1_accuracy",
                "top3_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1"
            ]
        )

        writer.writerow(
            [
                model_name,
                fraction,
                seed,
                top1_accuracy,
                top3_accuracy,
                macro_precision,
                macro_recall,
                macro_f1
            ]
        )

    # -----------------------------------------------------
    # Per-class metrics
    # -----------------------------------------------------

    with open(
        run_output_dir
        / "per_class_metrics.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "class_index",
                "action_label",
                "precision",
                "recall",
                "f1",
                "support"
            ]
        )

        for class_index in range(
            36
        ):

            writer.writerow(
                [
                    class_index,
                    class_index + 1,
                    class_precision[
                        class_index
                    ],
                    class_recall[
                        class_index
                    ],
                    class_f1[
                        class_index
                    ],
                    int(
                        class_support[
                            class_index
                        ]
                    )
                ]
            )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    np.savetxt(
        run_output_dir
        / "confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d"
    )

    # -----------------------------------------------------
    # Per-sample CSV
    # -----------------------------------------------------

    with open(
        run_output_dir
        / "sample_predictions.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "sample_index",
                "true_class_index",
                "true_action_label",
                "pred_class_index",
                "pred_action_label",
                "top3_class_indices",
                "top3_action_labels"
            ]
        )

        for i in range(
            len(labels)
        ):

            writer.writerow(
                [
                    i,
                    int(labels[i]),
                    int(labels[i]) + 1,
                    int(predictions[i]),
                    int(predictions[i]) + 1,
                    " ".join(
                        str(int(x))
                        for x
                        in top3_predictions[i]
                    ),
                    " ".join(
                        str(int(x) + 1)
                        for x
                        in top3_predictions[i]
                    )
                ]
            )

    del model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "fraction": fraction,
        "seed": seed,
        "top1_accuracy": (
            top1_accuracy
        ),
        "top3_accuracy": (
            top3_accuracy
        ),
        "macro_precision": (
            macro_precision
        ),
        "macro_recall": (
            macro_recall
        ),
        "macro_f1": (
            macro_f1
        )
    }


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "rgb_only",
            "hand_motion_only",
            "position_cross_attention",
            "motion_cross_attention"
        ]
    )

    parser.add_argument(
        "--fractions",
        nargs="+",
        type=int,
        default=[
            10,
            20,
            30,
            50,
            75,
            100
        ]
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[
            10,
            42,
            100
        ]
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Device:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    for model_name in args.models:

        for fraction in (
            args.fractions
        ):

            for seed in args.seeds:

                result = (
                    evaluate_checkpoint(

                        model_name,

                        fraction,

                        seed,

                        device,

                        args.batch_size,

                        args.num_workers
                    )
                )

                if result is not None:

                    all_results.append(
                        result
                    )

    # -----------------------------------------------------
    # Combined summary
    # -----------------------------------------------------

    summary_path = (
        OUTPUT_ROOT
        / "all_evaluation_metrics.csv"
    )

    with open(
        summary_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "fraction",
                "seed",
                "top1_accuracy",
                "top3_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1"
            ]
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print(
        "Summary:",
        summary_path
    )


if __name__ == "__main__":

    main()