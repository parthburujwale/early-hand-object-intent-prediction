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

CHECKPOINT_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/checkpoints/"
    "spatiotemporal_transformer"
)

RESULTS_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/results"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultimodalIntentDataset(Dataset):
    def __init__(self, path):
        data = np.load(
            path,
            allow_pickle=True
        )

        self.hand = torch.tensor(
            data["X_hand"],
            dtype=torch.float32
        )

        self.object = torch.tensor(
            data["X_object"],
            dtype=torch.float32
        )

        self.relative = torch.tensor(
            data["X_relative"],
            dtype=torch.float32
        )

        # H2O labels are 1...36
        self.y = torch.tensor(
            data["y"] - 1,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (
            self.hand[index],
            self.object[index],
            self.relative[index],
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


class SpatiotemporalIntentTransformer(nn.Module):
    def __init__(
        self,
        d_model=128,
        nhead=4,
        spatial_layers=1,
        temporal_layers=2,
        dim_feedforward=256,
        dropout=0.30,
        num_classes=36
    ):
        super().__init__()

        # Separate modality projections
        self.hand_projection = nn.Sequential(
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        self.object_projection = nn.Sequential(
            nn.Linear(12, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        self.relative_projection = nn.Sequential(
            nn.Linear(8, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # Learnable modality identity embeddings
        self.modality_embeddings = nn.Parameter(
            torch.randn(
                3,
                d_model
            ) * 0.02
        )

        # ----------------------------------------------------
        # Spatial interaction Transformer
        # ----------------------------------------------------

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer,
            num_layers=spatial_layers
        )

        # ----------------------------------------------------
        # Temporal Transformer
        # ----------------------------------------------------

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=32
        )

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
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

    def forward(
        self,
        hand,
        object_features,
        relative
    ):
        """
        hand:
            [B, T, 128]

        object_features:
            [B, T, 12]

        relative:
            [B, T, 8]
        """

        B, T, _ = hand.shape

        hand_token = self.hand_projection(
            hand
        )

        object_token = self.object_projection(
            object_features
        )

        relative_token = self.relative_projection(
            relative
        )

        # [B, T, 3, D]
        tokens = torch.stack(
            [
                hand_token,
                object_token,
                relative_token
            ],
            dim=2
        )

        tokens = (
            tokens
            + self.modality_embeddings[
                None,
                None,
                :,
                :
            ]
        )

        # Treat each frame independently for spatial fusion
        tokens = tokens.reshape(
            B * T,
            3,
            -1
        )

        tokens = self.spatial_encoder(
            tokens
        )

        # Fuse the 3 modalities for each frame
        frame_embedding = tokens.mean(
            dim=1
        )

        frame_embedding = frame_embedding.reshape(
            B,
            T,
            -1
        )

        # Add temporal position information
        frame_embedding = self.positional_encoding(
            frame_embedding
        )

        # Reason over interaction evolution
        temporal_output = self.temporal_encoder(
            frame_embedding
        )

        # Global temporal pooling
        representation = temporal_output.mean(
            dim=1
        )

        representation = self.norm(
            representation
        )

        representation = self.dropout(
            representation
        )

        logits = self.classifier(
            representation
        )

        return logits


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
            hand,
            obj,
            relative,
            y
        ) in loader:

            hand = hand.to(device)
            obj = obj.to(device)
            relative = relative.to(device)
            y = y.to(device)

            logits = model(
                hand,
                obj,
                relative
            )

            loss = criterion(
                logits,
                y
            )

            total_loss += (
                loss.item()
                * y.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            total_correct += (
                predictions == y
            ).sum().item()

            top3 = torch.topk(
                logits,
                k=3,
                dim=1
            ).indices

            total_top3 += (
                top3 == y.unsqueeze(1)
            ).any(dim=1).sum().item()

            total_samples += y.size(0)

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        total_top3 / total_samples
    )


def train_fraction(
    fraction,
    args,
    device
):
    print()
    print("=" * 80)
    print(
        f"Spatiotemporal Transformer: "
        f"{fraction}% observation"
    )
    print("=" * 80)

    train_dataset = MultimodalIntentDataset(
        DATA_ROOT
        / f"train_{fraction:03d}.npz"
    )

    val_dataset = MultimodalIntentDataset(
        DATA_ROOT
        / f"val_{fraction:03d}.npz"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(
            device.type == "cuda"
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(
            device.type == "cuda"
        )
    )

    model = SpatiotemporalIntentTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        spatial_layers=args.spatial_layers,
        temporal_layers=args.temporal_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=8
        )
    )

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        CHECKPOINT_ROOT
        / f"spatiotemporal_{fraction:03d}.pt"
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

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for (
            hand,
            obj,
            relative,
            y
        ) in train_loader:

            hand = hand.to(device)
            obj = obj.to(device)
            relative = relative.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(
                hand,
                obj,
                relative
            )

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += (
                loss.item()
                * y.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            total_correct += (
                predictions == y
            ).sum().item()

            total_samples += y.size(0)

        train_loss = (
            total_loss
            / total_samples
        )

        train_accuracy = (
            total_correct
            / total_samples
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

                    "fraction":
                        fraction,

                    "epoch":
                        epoch,

                    "val_accuracy":
                        val_accuracy,

                    "val_top3_accuracy":
                        val_top3
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
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy * 100:.2f}% | "
                f"Top-3: {val_top3 * 100:.2f}%"
            )

        if (
            patience_counter
            >= args.early_stopping
        ):
            print(
                f"Early stopping at epoch {epoch}"
            )
            break

    print()
    print(
        f"BEST {fraction}%"
    )

    print(
        f"Epoch: {best_epoch}"
    )

    print(
        f"Top-1: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        f"Top-3: "
        f"{best_top3 * 100:.2f}%"
    )

    return {
        "fraction": fraction,
        "best_epoch": best_epoch,
        "top1_accuracy": best_accuracy,
        "top3_accuracy": best_top3
    }


def save_results(results):
    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    fraction = results[0][
        "fraction"
    ]

    path = (
        RESULTS_ROOT
        / (
            f"spatiotemporal_transformer_"
            f"results_{fraction:03d}.csv"
        )
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fraction",
                "best_epoch",
                "top1_accuracy",
                "top3_accuracy"
            ]
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    print()
    print(
        "Results saved to:"
    )
    print(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fraction",
        type=int,
        default=None
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
        "--d_model",
        type=int,
        default=128
    )

    parser.add_argument(
        "--nhead",
        type=int,
        default=4
    )

    parser.add_argument(
        "--spatial_layers",
        type=int,
        default=1
    )

    parser.add_argument(
        "--temporal_layers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=256
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.30
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
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

    if args.fraction is not None:
        fractions = [
            args.fraction
        ]
    else:
        fractions = [
            10,
            20,
            30,
            50,
            75,
            100
        ]

    results = []

    for fraction in fractions:
        results.append(
            train_fraction(
                fraction,
                args,
                device
            )
        )

    save_results(
        results
    )


if __name__ == "__main__":
    main()