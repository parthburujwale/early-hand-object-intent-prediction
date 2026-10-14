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

from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
)


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
    "outputs/results/visual_hand_transformer_masked"
)

CHECKPOINT_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/"
    "outputs/checkpoints/visual_hand_transformer_masked"
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
            f"Expected 128 hand values, "
            f"found {len(values)} in {path}"
        )

    # -----------------------------------------------------
    # Left hand
    # -----------------------------------------------------

    left_valid = values[0]

    left_hand = values[
        1:64
    ].reshape(
        21,
        3
    )

    # -----------------------------------------------------
    # Right hand
    # -----------------------------------------------------

    right_valid = values[64]

    right_hand = values[
        65:128
    ].reshape(
        21,
        3
    )

    # -----------------------------------------------------
    # Wrist-relative coordinates
    # -----------------------------------------------------

    left_wrist = (
        left_hand[0].copy()
    )

    right_wrist = (
        right_hand[0].copy()
    )

    left_hand = (
        left_hand
        - left_wrist
    )

    right_hand = (
        right_hand
        - right_wrist
    )

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


# =========================================================
# Dataset
# =========================================================

class VisualHandMaskedDataset(
    Dataset
):

    def __init__(
        self,
        index_path,
        transform
    ):

        with open(
            index_path,
            "r"
        ) as f:

            self.samples = json.load(
                f
            )

        self.transform = transform

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

        images = []
        hands = []

        # -------------------------------------------------
        # New masked index contains exactly 32 frame slots
        # -------------------------------------------------

        for frame in sample[
            "frames"
        ]:

            # ---------------------------------------------
            # RGB
            # ---------------------------------------------

            image = Image.open(
                frame["rgb256"]
            ).convert(
                "RGB"
            )

            image = self.transform(
                image
            )

            images.append(
                image
            )

            # ---------------------------------------------
            # 3D hand pose
            # ---------------------------------------------

            hand = load_hand_pose(
                frame["hand_pose"]
            )

            hands.append(
                hand
            )

        # [32, 3, 224, 224]
        images = torch.stack(
            images
        )

        # [32, 128]
        hands = torch.tensor(
            np.stack(
                hands
            ),
            dtype=torch.float32
        )

        # True = real frame
        # False = padded frame
        #
        # [32]
        valid_mask = torch.tensor(
            sample["valid_mask"],
            dtype=torch.bool
        )

        # H2O labels are 1...36.
        # PyTorch expects 0...35.
        label = torch.tensor(
            sample[
                "action_label"
            ] - 1,
            dtype=torch.long
        )

        return (
            images,
            hands,
            valid_mask,
            label
        )


# =========================================================
# Frozen pretrained ViT
# =========================================================

class FrozenViTEncoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        weights = (
            ViT_B_16_Weights
            .IMAGENET1K_V1
        )

        self.vit = vit_b_16(
            weights=weights
        )

        # Freeze all ImageNet ViT weights
        for parameter in (
            self.vit.parameters()
        ):

            parameter.requires_grad = False

        self.output_dim = 768

    def forward(
        self,
        x
    ):

        """
        x:
            [N, 3, 224, 224]

        output:
            [N, 768]
        """

        with torch.no_grad():

            x = (
                self.vit
                ._process_input(
                    x
                )
            )

            batch_size = (
                x.shape[0]
            )

            cls_token = (
                self.vit
                .class_token
                .expand(
                    batch_size,
                    -1,
                    -1
                )
            )

            x = torch.cat(
                [
                    cls_token,
                    x
                ],
                dim=1
            )

            x = (
                self.vit.encoder(
                    x
                )
            )

            x = x[:, 0]

        return x


# =========================================================
# Multimodal masked Transformer
# =========================================================

class VisualHandMaskedTransformer(
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

        self.max_frames = (
            max_frames
        )

        # -------------------------------------------------
        # Vision encoder
        # -------------------------------------------------

        self.vision_encoder = (
            FrozenViTEncoder()
        )

        self.visual_projection = (
            nn.Sequential(

                nn.Linear(
                    768,
                    d_model
                ),

                nn.LayerNorm(
                    d_model
                ),

                nn.GELU()
            )
        )

        # -------------------------------------------------
        # Hand encoder
        # -------------------------------------------------

        self.hand_projection = (
            nn.Sequential(

                nn.Linear(
                    128,
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

        # -------------------------------------------------
        # CLS token
        # -------------------------------------------------

        self.cls_token = (
            nn.Parameter(
                torch.randn(
                    1,
                    1,
                    d_model
                ) * 0.02
            )
        )

        # -------------------------------------------------
        # Modality embeddings
        #
        # 0 = RGB
        # 1 = hand
        # -------------------------------------------------

        self.modality_embedding = (
            nn.Parameter(
                torch.randn(
                    2,
                    d_model
                ) * 0.02
            )
        )

        # -------------------------------------------------
        # Temporal embeddings
        #
        # Maximum sequence length = 32
        # -------------------------------------------------

        self.time_embedding = (
            nn.Parameter(
                torch.randn(
                    max_frames,
                    d_model
                ) * 0.02
            )
        )

        # -------------------------------------------------
        # Fusion Transformer
        # -------------------------------------------------

        layer = (
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

        self.fusion_transformer = (
            nn.TransformerEncoder(

                layer,

                num_layers=num_layers
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
        images,
        hands,
        valid_mask
    ):

        """
        images:
            [B, T, 3, 224, 224]

        hands:
            [B, T, 128]

        valid_mask:
            [B, T]

            True  = real observation
            False = padded observation
        """

        (
            B,
            T,
            C,
            H,
            W
        ) = images.shape

        # =================================================
        # Visual features
        # =================================================

        images_flat = (
            images.reshape(
                B * T,
                C,
                H,
                W
            )
        )

        visual_features = (
            self.vision_encoder(
                images_flat
            )
        )

        visual_features = (
            visual_features.reshape(
                B,
                T,
                -1
            )
        )

        visual_tokens = (
            self.visual_projection(
                visual_features
            )
        )

        # =================================================
        # Hand features
        # =================================================

        hand_tokens = (
            self.hand_projection(
                hands
            )
        )

        # =================================================
        # Temporal embeddings
        # =================================================

        time_embedding = (
            self.time_embedding[
                :T
            ][
                None,
                :,
                :
            ]
        )

        # =================================================
        # Add modality + time embeddings
        # =================================================

        visual_tokens = (
            visual_tokens

            + time_embedding

            + self.modality_embedding[
                0
            ][
                None,
                None,
                :
            ]
        )

        hand_tokens = (
            hand_tokens

            + time_embedding

            + self.modality_embedding[
                1
            ][
                None,
                None,
                :
            ]
        )

        # =================================================
        # Interleave modalities
        #
        # Sequence becomes:
        #
        # CLS,
        # RGB_t1,
        # HAND_t1,
        # RGB_t2,
        # HAND_t2,
        # ...
        #
        # T=32
        #
        # 64 multimodal tokens + CLS
        # =================================================

        tokens = torch.stack(
            [
                visual_tokens,
                hand_tokens
            ],
            dim=2
        )

        # [B, T, 2, D]
        #
        # ->
        #
        # [B, 2T, D]

        tokens = tokens.reshape(
            B,
            T * 2,
            -1
        )

        # =================================================
        # CLS token
        # =================================================

        cls = (
            self.cls_token.expand(
                B,
                -1,
                -1
            )
        )

        tokens = torch.cat(
            [
                cls,
                tokens
            ],
            dim=1
        )

        # =================================================
        # Build multimodal padding mask
        #
        # valid_mask:
        #
        # [True, True, False ...]
        #
        # Since each timestep has two tokens:
        #
        # RGB_t
        # HAND_t
        #
        # duplicate each timestep mask.
        # =================================================

        multimodal_valid_mask = (
            valid_mask
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                2
            )
            .reshape(
                B,
                T * 2
            )
        )

        # PyTorch key padding mask convention:
        #
        # True = IGNORE
        # False = USE
        #
        # Our valid_mask has the opposite meaning.

        padding_mask = (
            ~multimodal_valid_mask
        )

        # CLS token must never be masked

        cls_padding = torch.zeros(
            B,
            1,
            dtype=torch.bool,
            device=valid_mask.device
        )

        padding_mask = torch.cat(
            [
                cls_padding,
                padding_mask
            ],
            dim=1
        )

        # =================================================
        # Fusion Transformer
        # =================================================

        tokens = (
            self.fusion_transformer(

                tokens,

                src_key_padding_mask=(
                    padding_mask
                )
            )
        )

        # =================================================
        # Classification from CLS token
        # =================================================

        representation = (
            tokens[:, 0]
        )

        representation = (
            self.norm(
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
            images,
            hands,
            valid_mask,
            labels
        ) in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            hands = hands.to(
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
                hands,
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

            prediction = (
                logits.argmax(
                    dim=1
                )
            )

            total_correct += (
                prediction
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

    weights = (
        ViT_B_16_Weights
        .IMAGENET1K_V1
    )

    transform = (
        weights.transforms()
    )

    # -----------------------------------------------------
    # Dataset paths
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------

    train_dataset = (
        VisualHandMaskedDataset(
            train_path,
            transform
        )
    )

    val_dataset = (
        VisualHandMaskedDataset(
            val_path,
            transform
        )
    )

    # -----------------------------------------------------
    # Dataloaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,

        batch_size=(
            args.batch_size
        ),

        shuffle=True,

        num_workers=(
            args.num_workers
        ),

        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=(
            args.batch_size
        ),

        shuffle=False,

        num_workers=(
            args.num_workers
        ),

        pin_memory=True
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = (
        VisualHandMaskedTransformer(

            d_model=args.d_model,

            nhead=args.nhead,

            num_layers=(
                args.num_layers
            ),

            dim_feedforward=(
                args.dim_feedforward
            ),

            dropout=args.dropout,

            max_frames=32

        ).to(
            device
        )
    )

    # -----------------------------------------------------
    # Trainable parameters
    # -----------------------------------------------------

    trainable_parameters = [

        p

        for p in model.parameters()

        if p.requires_grad
    ]

    total_parameters = sum(

        p.numel()

        for p in model.parameters()
    )

    trainable_count = sum(

        p.numel()

        for p in trainable_parameters
    )

    print()

    print("=" * 80)

    print(
        "MASKED RGB + 3D Hand Transformer"
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

    print(
        f"Trainable parameters: "
        f"{trainable_count:,}"
    )

    print("=" * 80)

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = (
        nn.CrossEntropyLoss()
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = (
        torch.optim.AdamW(

            trainable_parameters,

            lr=args.learning_rate,

            weight_decay=(
                args.weight_decay
            )
        )
    )

    # -----------------------------------------------------
    # Scheduler
    # -----------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.5,

            patience=5
        )
    )

    # -----------------------------------------------------
    # Output directories
    # -----------------------------------------------------

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        CHECKPOINT_ROOT
        / (
            f"visual_hand_masked_"
            f"{args.fraction:03d}_"
            f"seed{args.seed}.pt"
        )
    )

    # -----------------------------------------------------
    # Best model tracking
    # -----------------------------------------------------

    best_accuracy = 0.0
    best_top3 = 0.0
    best_epoch = 0

    patience_counter = 0

    # =====================================================
    # Epoch loop
    # =====================================================

    for epoch in range(
        1,
        args.epochs + 1
    ):

        model.train()

        # Frozen ViT should always remain
        # in evaluation mode.

        model.vision_encoder.eval()

        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        # =================================================
        # Training batches
        # =================================================

        for (
            images,
            hands,
            valid_mask,
            labels
        ) in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            hands = hands.to(
                device,
                non_blocking=True
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
                images,
                hands,
                valid_mask
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

            batch_size = (
                labels.size(0)
            )

            train_loss += (
                loss.item()
                * batch_size
            )

            prediction = (
                logits.argmax(
                    dim=1
                )
            )

            train_correct += (
                prediction
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

        # =================================================
        # Validation
        # =================================================

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

        # =================================================
        # Save best model
        # =================================================

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

        # =================================================
        # Print training progress
        # =================================================

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

        # =================================================
        # Early stopping
        # =================================================

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

    # =====================================================
    # Save result CSV
    # =====================================================

    result_path = (
        RESULTS_ROOT
        / (
            f"visual_hand_masked_"
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

    # =====================================================
    # Final output
    # =====================================================

    print()

    print("=" * 80)

    print(
        "BEST RESULT"
    )

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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    if (
        device.type
        == "cuda"
    ):

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