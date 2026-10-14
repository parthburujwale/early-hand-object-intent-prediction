from pathlib import Path
import argparse
import csv
import json
import random

from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights


INDEX_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/"
    "outputs/partial_visual_index"
)

RESULTS_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/"
    "outputs/results/rgb_only_vit"
)

CHECKPOINT_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/"
    "outputs/checkpoints/rgb_only_vit"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RGBOnlyDataset(Dataset):

    def __init__(
        self,
        index_path,
        transform
    ):
        with open(index_path, "r") as f:
            self.samples = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        images = []

        for frame in sample["sampled_frames"]:

            image = Image.open(
                frame["rgb256"]
            ).convert("RGB")

            image = self.transform(image)

            images.append(image)

        images = torch.stack(images)

        label = torch.tensor(
            sample["action_label"] - 1,
            dtype=torch.long
        )

        return images, label


class FrozenViTEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        weights = ViT_B_16_Weights.IMAGENET1K_V1

        self.vit = vit_b_16(
            weights=weights
        )

        for p in self.vit.parameters():
            p.requires_grad = False

        self.output_dim = 768

    def forward(self, x):

        with torch.no_grad():

            x = self.vit._process_input(x)

            batch_size = x.shape[0]

            cls_token = self.vit.class_token.expand(
                batch_size,
                -1,
                -1
            )

            x = torch.cat(
                [cls_token, x],
                dim=1
            )

            x = self.vit.encoder(x)

            x = x[:, 0]

        return x


class RGBTemporalTransformer(nn.Module):

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.30,
        num_classes=36,
        num_frames=8
    ):
        super().__init__()

        self.vision_encoder = FrozenViTEncoder()

        self.visual_projection = nn.Sequential(
            nn.Linear(
                768,
                d_model
            ),
            nn.LayerNorm(
                d_model
            ),
            nn.GELU()
        )

        self.cls_token = nn.Parameter(
            torch.randn(
                1,
                1,
                d_model
            ) * 0.02
        )

        self.time_embedding = nn.Parameter(
            torch.randn(
                num_frames,
                d_model
            ) * 0.02
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

        self.temporal_transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
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

    def forward(self, images):

        B, T, C, H, W = images.shape

        images = images.reshape(
            B * T,
            C,
            H,
            W
        )

        features = self.vision_encoder(
            images
        )

        features = features.reshape(
            B,
            T,
            -1
        )

        tokens = self.visual_projection(
            features
        )

        tokens = (
            tokens
            + self.time_embedding[
                :T
            ][None, :, :]
        )

        cls = self.cls_token.expand(
            B,
            -1,
            -1
        )

        tokens = torch.cat(
            [
                cls,
                tokens
            ],
            dim=1
        )

        tokens = self.temporal_transformer(
            tokens
        )

        representation = tokens[:, 0]

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

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            predictions = logits.argmax(
                dim=1
            )

            total_correct += (
                predictions == labels
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

            total_samples += batch_size

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

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    train_dataset = RGBOnlyDataset(
        INDEX_ROOT
        / f"train_{args.fraction:03d}.json",
        transform
    )

    val_dataset = RGBOnlyDataset(
        INDEX_ROOT
        / f"val_{args.fraction:03d}.json",
        transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = RGBTemporalTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)

    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    print()
    print("=" * 80)
    print("RGB-only ViT Temporal Transformer")
    print(
        "Observation:",
        f"{args.fraction}%"
    )
    print(
        "Train samples:",
        len(train_dataset)
    )
    print(
        "Val samples:",
        len(val_dataset)
    )
    print("=" * 80)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
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
        / f"rgb_only_{args.fraction:03d}.pt"
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
        model.vision_encoder.eval()

        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0
            )

            optimizer.step()

            batch_size = labels.size(0)

            train_loss += (
                loss.item()
                * batch_size
            )

            train_correct += (
                logits.argmax(
                    dim=1
                )
                == labels
            ).sum().item()

            train_samples += batch_size

        train_loss /= train_samples

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

            best_accuracy = val_accuracy
            best_top3 = val_top3
            best_epoch = epoch

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

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
        / f"rgb_only_{args.fraction:03d}.csv"
    )

    with open(
        result_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "fraction",
            "best_epoch",
            "top1_accuracy",
            "top3_accuracy"
        ])

        writer.writerow([
            args.fraction,
            best_epoch,
            best_accuracy,
            best_top3
        ])

    print()
    print("=" * 80)
    print("BEST RESULT")
    print("=" * 80)

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
        "Results:",
        result_path
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fraction",
        type=int,
        default=10
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80
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

    train(
        args
    )


if __name__ == "__main__":
    main()