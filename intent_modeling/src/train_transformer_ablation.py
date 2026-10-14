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
    "intent_prediction/intent_modeling/outputs/results/ablations"
)

CHECKPOINT_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/checkpoints/ablations"
)


MODES = {
    "hand": ["hand"],
    "object": ["object"],
    "relative": ["relative"],
    "hand_object": ["hand", "object"],
    "hand_relative": ["hand", "relative"],
    "full": ["hand", "object", "relative"],
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IntentDataset(Dataset):

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


class AblationTransformer(nn.Module):

    def __init__(
        self,
        mode,
        d_model=128,
        nhead=4,
        spatial_layers=1,
        temporal_layers=2,
        dim_feedforward=256,
        dropout=0.30,
        num_classes=36
    ):

        super().__init__()

        self.active_modalities = MODES[
            mode
        ]

        self.projections = nn.ModuleDict({

            "hand": nn.Sequential(
                nn.Linear(
                    128,
                    d_model
                ),
                nn.LayerNorm(
                    d_model
                ),
                nn.GELU()
            ),

            "object": nn.Sequential(
                nn.Linear(
                    12,
                    d_model
                ),
                nn.LayerNorm(
                    d_model
                ),
                nn.GELU()
            ),

            "relative": nn.Sequential(
                nn.Linear(
                    8,
                    d_model
                ),
                nn.LayerNorm(
                    d_model
                ),
                nn.GELU()
            )
        })

        self.modality_embeddings = nn.ParameterDict({

            name: nn.Parameter(
                torch.randn(
                    1,
                    d_model
                ) * 0.02
            )

            for name in [
                "hand",
                "object",
                "relative"
            ]
        })

        spatial_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )

        self.spatial_encoder = (
            nn.TransformerEncoder(
                spatial_layer,
                num_layers=spatial_layers
            )
        )

        self.positional_encoding = (
            PositionalEncoding(
                d_model=d_model,
                max_len=32
            )
        )

        temporal_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )

        self.temporal_encoder = (
            nn.TransformerEncoder(
                temporal_layer,
                num_layers=temporal_layers
            )
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
        obj,
        relative
    ):

        inputs = {
            "hand": hand,
            "object": obj,
            "relative": relative
        }

        tokens = []

        for modality in (
            self.active_modalities
        ):

            token = self.projections[
                modality
            ](
                inputs[
                    modality
                ]
            )

            token = (
                token
                + self.modality_embeddings[
                    modality
                ]
            )

            tokens.append(
                token
            )

        # B x T x M x D
        tokens = torch.stack(
            tokens,
            dim=2
        )

        B, T, M, D = tokens.shape

        # Perform spatial attention independently
        # at every timestep
        tokens = tokens.reshape(
            B * T,
            M,
            D
        )

        tokens = self.spatial_encoder(
            tokens
        )

        # Fuse modalities
        frame_embedding = tokens.mean(
            dim=1
        )

        frame_embedding = (
            frame_embedding.reshape(
                B,
                T,
                D
            )
        )

        frame_embedding = (
            self.positional_encoding(
                frame_embedding
            )
        )

        temporal_output = (
            self.temporal_encoder(
                frame_embedding
            )
        )

        representation = (
            temporal_output.mean(
                dim=1
            )
        )

        representation = self.norm(
            representation
        )

        representation = self.dropout(
            representation
        )

        return self.classifier(
            representation
        )


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

            hand = hand.to(
                device
            )

            obj = obj.to(
                device
            )

            relative = relative.to(
                device
            )

            y = y.to(
                device
            )

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
        "Mode:",
        args.mode
    )

    print(
        "Modalities:",
        MODES[args.mode]
    )

    train_dataset = IntentDataset(
        DATA_ROOT
        / f"train_{args.fraction:03d}.npz"
    )

    val_dataset = IntentDataset(
        DATA_ROOT
        / f"val_{args.fraction:03d}.npz"
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

    model = AblationTransformer(
        mode=args.mode
    ).to(
        device
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-4
        )
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=8
        )
    )

    best_accuracy = 0.0
    best_top3 = 0.0
    best_epoch = 0

    patience_counter = 0

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        CHECKPOINT_ROOT
        / (
            f"{args.mode}_"
            f"{args.fraction:03d}.pt"
        )
    )

    for epoch in range(
        1,
        args.epochs + 1
    ):

        model.train()

        train_correct = 0
        train_samples = 0
        train_loss = 0.0

        for (
            hand,
            obj,
            relative,
            y
        ) in train_loader:

            hand = hand.to(
                device
            )

            obj = obj.to(
                device
            )

            relative = relative.to(
                device
            )

            y = y.to(
                device
            )

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
                1.0
            )

            optimizer.step()

            train_loss += (
                loss.item()
                * y.size(0)
            )

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

                    "mode":
                        args.mode,

                    "fraction":
                        args.fraction,

                    "epoch":
                        epoch,

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
        / (
            f"{args.mode}_"
            f"{args.fraction:03d}.csv"
        )
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
        "Mode:",
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
        choices=list(
            MODES.keys()
        ),
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