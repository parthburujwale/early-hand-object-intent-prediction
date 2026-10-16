from pathlib import Path
import argparse
import csv
import json
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


# =========================================================
# Paths
# =========================================================

INDEX_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/"
    "outputs/masked_partial_index"
)

RESULTS_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/"
    "outputs/results/hand_motion_only"
)

CHECKPOINT_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/"
    "outputs/checkpoints/hand_motion_only"
)


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Hand pose processing
# =========================================================

def load_hand_pose(path):

    values = np.loadtxt(
        path,
        dtype=np.float32
    ).reshape(-1)

    if len(values) != 128:
        raise ValueError(
            f"Expected 128 values, got {len(values)} in {path}"
        )

    left_valid = values[0]

    left_hand = values[
        1:64
    ].reshape(
        21,
        3
    )

    right_valid = values[64]

    right_hand = values[
        65:128
    ].reshape(
        21,
        3
    )

    # Wrist-relative geometry
    left_wrist = left_hand[0].copy()
    right_wrist = right_hand[0].copy()

    left_hand = left_hand - left_wrist
    right_hand = right_hand - right_wrist

    features = np.concatenate(
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

    return features.astype(
        np.float32
    )


def compute_velocity(
    hand_sequence,
    valid_mask
):
    """
    hand_sequence:
        [T, 128]

    valid_mask:
        [T]

    Returns:
        velocity [T, 128]
    """

    velocity = np.zeros_like(
        hand_sequence,
        dtype=np.float32
    )

    T = hand_sequence.shape[0]

    for t in range(
        1,
        T
    ):

        if (
            valid_mask[t]
            and valid_mask[t - 1]
        ):

            velocity[t] = (
                hand_sequence[t]
                - hand_sequence[t - 1]
            )

    return velocity


# =========================================================
# Dataset
# =========================================================

class HandMotionDataset(
    Dataset
):

    def __init__(
        self,
        index_path
    ):

        with open(
            index_path,
            "r"
        ) as f:

            self.samples = json.load(
                f
            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index
    ):

        sample = self.samples[
            index
        ]

        valid_mask_np = np.array(
            sample["valid_mask"],
            dtype=bool
        )

        hands = []

        for frame in sample[
            "frames"
        ]:

            hand = load_hand_pose(
                frame["hand_pose"]
            )

            hands.append(
                hand
            )

        hand_positions_np = np.stack(
            hands
        ).astype(
            np.float32
        )

        hand_velocity_np = compute_velocity(
            hand_positions_np,
            valid_mask_np
        )

        hand_positions = torch.tensor(
            hand_positions_np,
            dtype=torch.float32
        )

        hand_velocity = torch.tensor(
            hand_velocity_np,
            dtype=torch.float32
        )

        valid_mask = torch.tensor(
            valid_mask_np,
            dtype=torch.bool
        )

        label = torch.tensor(
            sample[
                "action_label"
            ] - 1,
            dtype=torch.long
        )

        return (
            hand_positions,
            hand_velocity,
            valid_mask,
            label
        )


# =========================================================
# Hand-motion model
# =========================================================

class HandMotionOnlyTransformer(
    nn.Module
):

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.30,
        num_classes=36,
        max_frames=32
    ):

        super().__init__()

        # Position 128 + velocity 128
        self.input_projection = (
            nn.Sequential(

                nn.Linear(
                    256,
                    d_model
                ),

                nn.LayerNorm(
                    d_model
                ),

                nn.GELU(),

                nn.Linear(
                    d_model,
                    d_model
                )
            )
        )

        self.time_embedding = (
            nn.Parameter(
                torch.randn(
                    max_frames,
                    d_model
                ) * 0.02
            )
        )

        self.cls_token = (
            nn.Parameter(
                torch.randn(
                    1,
                    1,
                    d_model
                ) * 0.02
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(

                d_model=d_model,

                nhead=nhead,

                dim_feedforward=(
                    dim_feedforward
                ),

                dropout=dropout,

                activation="gelu",

                batch_first=True,

                norm_first=True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(

                encoder_layer,

                num_layers=num_layers
            )
        )

        self.final_norm = (
            nn.LayerNorm(
                d_model
            )
        )

        self.dropout = (
            nn.Dropout(
                dropout
            )
        )

        self.classifier = (
            nn.Linear(
                d_model,
                num_classes
            )
        )

    def forward(
        self,
        hand_positions,
        hand_velocity,
        valid_mask
    ):

        B, T, _ = (
            hand_positions.shape
        )

        x = torch.cat(
            [
                hand_positions,
                hand_velocity
            ],
            dim=-1
        )

        x = self.input_projection(
            x
        )

        x = (
            x
            + self.time_embedding[
                :T
            ][
                None,
                :,
                :
            ]
        )

        cls = (
            self.cls_token.expand(
                B,
                -1,
                -1
            )
        )

        sequence = torch.cat(
            [
                cls,
                x
            ],
            dim=1
        )

        cls_padding = torch.zeros(
            B,
            1,
            dtype=torch.bool,
            device=valid_mask.device
        )

        padding_mask = torch.cat(
            [
                cls_padding,
                ~valid_mask
            ],
            dim=1
        )

        sequence = self.transformer(
            sequence,
            src_key_padding_mask=(
                padding_mask
            )
        )

        representation = (
            sequence[:, 0]
        )

        representation = (
            self.final_norm(
                representation
            )
        )

        representation = (
            self.dropout(
                representation
            )
        )

        logits = (
            self.classifier(
                representation
            )
        )

        return logits


# =========================================================
# Evaluation
# =========================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_top3 = 0
    total_samples = 0

    with torch.no_grad():

        for (
            hand_positions,
            hand_velocity,
            valid_mask,
            labels
        ) in loader:

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

            valid_mask = (
                valid_mask.to(
                    device,
                    non_blocking=True
                )
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

            loss = criterion(
                logits,
                labels
            )

            batch_size = (
                labels.size(0)
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            total_correct += (
                predictions
                == labels
            ).sum().item()

            top3 = torch.topk(
                logits,
                k=3,
                dim=1
            ).indices

            total_top3 += (
                top3
                == labels.unsqueeze(1)
            ).any(
                dim=1
            ).sum().item()

            total_samples += (
                batch_size
            )

    return (
        total_loss
        / total_samples,

        total_correct
        / total_samples,

        total_top3
        / total_samples
    )


# =========================================================
# Training
# =========================================================

def train_fraction(
    args,
    device
):

    train_path = (
        INDEX_ROOT
        / (
            f"train_"
            f"{args.fraction:03d}.json"
        )
    )

    val_path = (
        INDEX_ROOT
        / (
            f"val_"
            f"{args.fraction:03d}.json"
        )
    )

    train_dataset = (
        HandMotionDataset(
            train_path
        )
    )

    val_dataset = (
        HandMotionDataset(
            val_path
        )
    )

    train_loader = (
        DataLoader(

            train_dataset,

            batch_size=args.batch_size,

            shuffle=True,

            num_workers=(
                args.num_workers
            ),

            pin_memory=True
        )
    )

    val_loader = (
        DataLoader(

            val_dataset,

            batch_size=args.batch_size,

            shuffle=False,

            num_workers=(
                args.num_workers
            ),

            pin_memory=True
        )
    )

    model = (
        HandMotionOnlyTransformer(

            d_model=args.d_model,

            nhead=args.nhead,

            num_layers=(
                args.num_layers
            ),

            dim_feedforward=(
                args.dim_feedforward
            ),

            dropout=args.dropout

        ).to(
            device
        )
    )

    total_parameters = sum(

        p.numel()

        for p in model.parameters()
    )

    print()
    print("=" * 80)
    print(
        "HAND-MOTION-ONLY TRANSFORMER"
    )
    print(
        "Observation:",
        f"{args.fraction}%"
    )
    print(
        "Seed:",
        args.seed
    )
    print(
        "Train samples:",
        len(train_dataset)
    )
    print(
        "Val samples:",
        len(val_dataset)
    )
    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )
    print("=" * 80)

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = (
        torch.optim.AdamW(

            model.parameters(),

            lr=args.learning_rate,

            weight_decay=(
                args.weight_decay
            )
        )
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.5,

            patience=5
        )
    )

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        CHECKPOINT_ROOT
        / (
            f"hand_motion_"
            f"{args.fraction:03d}_"
            f"seed{args.seed}.pt"
        )
    )

    best_accuracy = 0.0
    best_top3 = 0.0
    best_epoch = 0

    patience_counter = 0

    for epoch in range(
        1,
        args.epochs + 1
    ):

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for (
            hand_positions,
            hand_velocity,
            valid_mask,
            labels
        ) in train_loader:

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

            valid_mask = (
                valid_mask.to(
                    device,
                    non_blocking=True
                )
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(

                hand_positions,

                hand_velocity,

                valid_mask
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            batch_size = (
                labels.size(0)
            )

            train_loss += (
                loss.item()
                * batch_size
            )

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            train_correct += (
                predictions
                == labels
            ).sum().item()

            train_samples += (
                batch_size
            )

        train_loss = (
            train_loss
            / train_samples
        )

        train_accuracy = (
            train_correct
            / train_samples
        )

        (
            val_loss,
            val_accuracy,
            val_top3
        ) = evaluate(

            model,

            val_loader,

            criterion,

            device
        )

        scheduler.step(
            val_accuracy
        )

        improved = (
            val_accuracy
            > best_accuracy
        )

        if improved:

            best_accuracy = (
                val_accuracy
            )

            best_top3 = (
                val_top3
            )

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "fraction":
                        args.fraction,

                    "seed":
                        args.seed,

                    "epoch":
                        best_epoch,

                    "top1_accuracy":
                        best_accuracy,

                    "top3_accuracy":
                        best_top3
                },

                checkpoint_path
            )

        else:

            patience_counter += 1

        if (
            epoch == 1
            or epoch % 2 == 0
            or improved
        ):

            print(

                f"Epoch {epoch:03d} | "

                f"Train Loss: "
                f"{train_loss:.4f} | "

                f"Train Acc: "
                f"{train_accuracy * 100:.2f}% | "

                f"Val Loss: "
                f"{val_loss:.4f} | "

                f"Val Acc: "
                f"{val_accuracy * 100:.2f}% | "

                f"Top-3: "
                f"{val_top3 * 100:.2f}%"
            )

        if (
            patience_counter
            >= args.early_stopping
        ):

            print()

            print(
                "Early stopping at "
                f"epoch {epoch}"
            )

            break

    result_path = (
        RESULTS_ROOT
        / (
            f"hand_motion_"
            f"{args.fraction:03d}_"
            f"seed{args.seed}.csv"
        )
    )

    with open(
        result_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "fraction",
                "seed",
                "best_epoch",
                "top1_accuracy",
                "top3_accuracy"
            ]
        )

        writer.writerow(
            [
                args.fraction,
                args.seed,
                best_epoch,
                best_accuracy,
                best_top3
            ]
        )

    print()
    print("=" * 80)
    print("BEST RESULT")
    print("=" * 80)

    print(
        "Observation:",
        f"{args.fraction}%"
    )

    print(
        "Seed:",
        args.seed
    )

    print(
        "Epoch:",
        best_epoch
    )

    print(
        "Top-1:",
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        "Top-3:",
        f"{best_top3 * 100:.2f}%"
    )

    print(
        "Checkpoint:",
        checkpoint_path
    )

    print(
        "Results:",
        result_path
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fraction",
        type=int,
        default=10,
        choices=[
            10,
            20,
            30,
            50,
            75,
            100
        ]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--d_model",
        type=int,
        default=256
    )

    parser.add_argument(
        "--nhead",
        type=int,
        default=8
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=3
    )

    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=512
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.30
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--early_stopping",
        type=int,
        default=15
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

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

    train_fraction(
        args,
        device
    )


if __name__ == "__main__":
    main()