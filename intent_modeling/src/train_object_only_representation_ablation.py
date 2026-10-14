from pathlib import Path
import argparse
import csv
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DATA_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/multimodal"
)

RESULTS_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/results/"
    "object_only_representation"
)

CHECKPOINT_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/checkpoints/"
    "object_only_representation"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_object_representation(X_object, mode):
    X_object = X_object.copy()

    n, t, _ = X_object.shape

    rotations = X_object[:, :, :9].reshape(
        n,
        t,
        3,
        3
    )

    translations = X_object[:, :, 9:12]

    if mode == "absolute":
        return X_object.astype(np.float32)

    if mode == "static":
        first_pose = X_object[:, 0:1, :]

        return np.repeat(
            first_pose,
            t,
            axis=1
        ).astype(np.float32)

    if mode == "motion":
        R0 = rotations[:, 0]
        T0 = translations[:, 0]

        relative_rotations = []

        for frame in range(t):
            Rt = rotations[:, frame]

            relative_R = np.matmul(
                np.transpose(
                    R0,
                    (0, 2, 1)
                ),
                Rt
            )

            relative_rotations.append(
                relative_R
            )

        relative_rotations = np.stack(
            relative_rotations,
            axis=1
        )

        relative_translation = (
            translations
            - T0[:, None, :]
        )

        motion = np.concatenate(
            [
                relative_rotations.reshape(
                    n,
                    t,
                    9
                ),
                relative_translation
            ],
            axis=-1
        )

        return motion.astype(np.float32)

    raise ValueError(
        f"Unknown mode: {mode}"
    )


class ObjectOnlyDataset(Dataset):

    def __init__(
        self,
        path,
        mode
    ):
        data = np.load(
            path,
            allow_pickle=True
        )

        X_object = convert_object_representation(
            data["X_object"],
            mode
        )

        self.object = torch.tensor(
            X_object,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            data["y"] - 1,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (
            self.object[index],
            self.y[index]
        )


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=32
    ):
        super().__init__()

        position = torch.arange(
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):
        return (
            x
            + self.pe[:, :x.size(1)]
        )


class ObjectOnlyTransformer(nn.Module):

    def __init__(
        self,
        d_model=128,
        nhead=4,
        temporal_layers=2,
        dim_feedforward=256,
        dropout=0.30,
        num_classes=36
    ):
        super().__init__()

        self.object_projection = nn.Sequential(
            nn.Linear(
                12,
                d_model
            ),
            nn.LayerNorm(
                d_model
            ),
            nn.GELU()
        )

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=32
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=temporal_layers
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, obj):

        x = self.object_projection(
            obj
        )

        x = self.positional_encoding(
            x
        )

        x = self.encoder(
            x
        )

        x = x.mean(
            dim=1
        )

        x = self.norm(
            x
        )

        x = self.dropout(
            x
        )

        return self.classifier(
            x
        )


def evaluate(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    total_correct = 0
    total_top3 = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():

        for obj, y in loader:

            obj = obj.to(
                device
            )

            y = y.to(
                device
            )

            logits = model(
                obj
            )

            loss = criterion(
                logits,
                y
            )

            total_loss += (
                loss.item()
                * y.size(0)
            )

            prediction = logits.argmax(
                dim=1
            )

            total_correct += (
                prediction == y
            ).sum().item()

            top3 = torch.topk(
                logits,
                k=3,
                dim=1
            ).indices

            total_top3 += (
                top3
                == y.unsqueeze(1)
            ).any(
                dim=1
            ).sum().item()

            total_samples += (
                y.size(0)
            )

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        total_top3 / total_samples
    )


def train(args):

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
            torch.cuda.get_device_name(0)
        )

    print(
        "Object-only representation:",
        args.mode
    )

    print(
        "Observation:",
        f"{args.fraction}%"
    )

    train_dataset = ObjectOnlyDataset(
        DATA_ROOT
        / f"train_{args.fraction:03d}.npz",
        args.mode
    )

    val_dataset = ObjectOnlyDataset(
        DATA_ROOT
        / f"val_{args.fraction:03d}.npz",
        args.mode
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = ObjectOnlyTransformer().to(
        device
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8
    )

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        CHECKPOINT_ROOT
        / f"{args.mode}_{args.fraction:03d}.pt"
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

        train_correct = 0
        train_samples = 0

        for obj, y in train_loader:

            obj = obj.to(
                device
            )

            y = y.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                obj
            )

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            train_correct += (
                logits.argmax(
                    dim=1
                )
                == y
            ).sum().item()

            train_samples += (
                y.size(0)
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

            best_accuracy = val_accuracy
            best_top3 = val_top3
            best_epoch = epoch

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "mode":
                        args.mode,

                    "fraction":
                        args.fraction,

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
            or epoch % 5 == 0
            or improved
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"Train Acc: "
                f"{100 * train_correct / train_samples:.2f}% | "
                f"Val Acc: "
                f"{100 * val_accuracy:.2f}% | "
                f"Top-3: "
                f"{100 * val_top3:.2f}%"
            )

        if (
            patience_counter
            >= args.early_stopping
        ):

            print(
                "Early stopping:",
                epoch
            )

            break

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RESULTS_ROOT
        / f"{args.mode}_{args.fraction:03d}.csv"
    )

    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([
            "mode",
            "fraction",
            "best_epoch",
            "top1_accuracy",
            "top3_accuracy"
        ])

        writer.writerow([
            args.mode,
            args.fraction,
            best_epoch,
            best_accuracy,
            best_top3
        ])

    print()
    print("=" * 60)

    print(
        "BEST RESULT"
    )

    print(
        "Object-only representation:",
        args.mode
    )

    print(
        "Observation:",
        f"{args.fraction}%"
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
        "Saved:",
        output_path
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "absolute",
            "static",
            "motion"
        ],
        required=True
    )

    parser.add_argument(
        "--fraction",
        type=int,
        default=10
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4
    )

    parser.add_argument(
        "--early_stopping",
        type=int,
        default=20
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

    train(
        args
    )


if __name__ == "__main__":
    main()