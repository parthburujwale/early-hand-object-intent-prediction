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
    "outputs/results/motion_cross_attention"
)

CHECKPOINT_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/"
    "outputs/checkpoints/motion_cross_attention"
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

    # Wrist-relative hand geometry
    left_wrist = left_hand[0].copy()
    right_wrist = right_hand[0].copy()

    left_hand = (
        left_hand - left_wrist
    )

    right_hand = (
        right_hand - right_wrist
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

    First valid frame has zero velocity.

    Invalid padded frames stay zero.
    """

    T = hand_sequence.shape[0]

    velocity = np.zeros_like(
        hand_sequence,
        dtype=np.float32
    )

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

class MotionVisualHandDataset(
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

        valid_mask_np = np.array(
            sample["valid_mask"],
            dtype=bool
        )

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
            # Hand pose
            # ---------------------------------------------

            hand = load_hand_pose(
                frame["hand_pose"]
            )

            hands.append(
                hand
            )

        images = torch.stack(
            images
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
            images,
            hand_positions,
            hand_velocity,
            valid_mask,
            label
        )


# =========================================================
# Frozen ViT
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

        for parameter in (
            self.vit.parameters()
        ):

            parameter.requires_grad = False

    def forward(
        self,
        x
    ):

        with torch.no_grad():

            x = self.vit._process_input(
                x
            )

            batch_size = (
                x.shape[0]
            )

            cls_token = (
                self.vit.class_token
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

            x = self.vit.encoder(
                x
            )

            x = x[:, 0]

        return x


# =========================================================
# Hand motion encoder
# =========================================================

class HandMotionEncoder(
    nn.Module
):

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.30,
        max_frames=32
    ):

        super().__init__()

        # Position + velocity
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

        self.transformer = (
            nn.TransformerEncoder(

                layer,

                num_layers=num_layers
            )
        )

    def forward(
        self,
        hand_positions,
        hand_velocity,
        valid_mask
    ):

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

        T = x.shape[1]

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

        padding_mask = (
            ~valid_mask
        )

        x = self.transformer(
            x,
            src_key_padding_mask=(
                padding_mask
            )
        )

        return x


# =========================================================
# Main model
# =========================================================

class MotionCrossAttentionTransformer(
    nn.Module
):

    def __init__(
        self,
        d_model=256,
        nhead=8,
        hand_layers=2,
        fusion_layers=2,
        dim_feedforward=512,
        dropout=0.30,
        num_classes=36,
        max_frames=32
    ):

        super().__init__()

        # -------------------------------------------------
        # RGB encoder
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

        self.visual_time_embedding = (
            nn.Parameter(
                torch.randn(
                    max_frames,
                    d_model
                ) * 0.02
            )
        )

        # -------------------------------------------------
        # Hand motion encoder
        # -------------------------------------------------

        self.hand_encoder = (
            HandMotionEncoder(

                d_model=d_model,

                nhead=nhead,

                num_layers=hand_layers,

                dim_feedforward=(
                    dim_feedforward
                ),

                dropout=dropout,

                max_frames=max_frames
            )
        )

        # -------------------------------------------------
        # Hand -> visual cross attention
        # -------------------------------------------------

        self.cross_attention = (
            nn.MultiheadAttention(

                embed_dim=d_model,

                num_heads=nhead,

                dropout=dropout,

                batch_first=True
            )
        )

        self.cross_norm_1 = (
            nn.LayerNorm(
                d_model
            )
        )

        self.cross_ffn = (
            nn.Sequential(

                nn.Linear(
                    d_model,
                    dim_feedforward
                ),

                nn.GELU(),

                nn.Dropout(
                    dropout
                ),

                nn.Linear(
                    dim_feedforward,
                    d_model
                )
            )
        )

        self.cross_norm_2 = (
            nn.LayerNorm(
                d_model
            )
        )

        # -------------------------------------------------
        # Final temporal fusion
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

        fusion_layer = (
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

                fusion_layer,

                num_layers=fusion_layers
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
        images,
        hand_positions,
        hand_velocity,
        valid_mask
    ):

        (
            B,
            T,
            C,
            H,
            W
        ) = images.shape

        # =================================================
        # RGB features
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

        visual_tokens = (
            visual_tokens
            + self.visual_time_embedding[
                :T
            ][
                None,
                :,
                :
            ]
        )

        # =================================================
        # Hand motion tokens
        # =================================================

        hand_tokens = (
            self.hand_encoder(

                hand_positions,

                hand_velocity,

                valid_mask
            )
        )

        # =================================================
        # Cross attention
        #
        # Query:
        #   hand motion
        #
        # Key/value:
        #   RGB scene
        #
        # Interpretation:
        #   "Given how the hand is moving,
        #    which visual evidence matters?"
        # =================================================

        visual_padding_mask = (
            ~valid_mask
        )

        cross_output, _ = (
            self.cross_attention(

                query=hand_tokens,

                key=visual_tokens,

                value=visual_tokens,

                key_padding_mask=(
                    visual_padding_mask
                ),

                need_weights=False
            )
        )

        fused_tokens = (
            self.cross_norm_1(
                hand_tokens
                + cross_output
            )
        )

        ffn_output = (
            self.cross_ffn(
                fused_tokens
            )
        )

        fused_tokens = (
            self.cross_norm_2(
                fused_tokens
                + ffn_output
            )
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

        sequence = torch.cat(
            [
                cls,
                fused_tokens
            ],
            dim=1
        )

        # CLS always valid
        cls_padding = torch.zeros(
            B,
            1,
            dtype=torch.bool,
            device=valid_mask.device
        )

        sequence_padding_mask = (
            torch.cat(
                [
                    cls_padding,
                    ~valid_mask
                ],
                dim=1
            )
        )

        # =================================================
        # Final temporal fusion
        # =================================================

        sequence = (
            self.fusion_transformer(

                sequence,

                src_key_padding_mask=(
                    sequence_padding_mask
                )
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
            images,
            hand_positions,
            hand_velocity,
            valid_mask,
            labels
        ) in loader:

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

                images,

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

    weights = (
        ViT_B_16_Weights
        .IMAGENET1K_V1
    )

    transform = (
        weights.transforms()
    )

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
        MotionVisualHandDataset(
            train_path,
            transform
        )
    )

    val_dataset = (
        MotionVisualHandDataset(
            val_path,
            transform
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
        MotionCrossAttentionTransformer(

            d_model=args.d_model,

            nhead=args.nhead,

            hand_layers=(
                args.hand_layers
            ),

            fusion_layers=(
                args.fusion_layers
            ),

            dim_feedforward=(
                args.dim_feedforward
            ),

            dropout=args.dropout

        ).to(
            device
        )
    )

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
        "MOTION-AWARE RGB + HAND CROSS-ATTENTION TRANSFORMER"
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

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = (
        torch.optim.AdamW(

            trainable_parameters,

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
            f"motion_cross_"
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
        model.vision_encoder.eval()

        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for (
            images,
            hand_positions,
            hand_velocity,
            valid_mask,
            labels
        ) in train_loader:

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
            f"motion_cross_"
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
        "--hand_layers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--fusion_layers",
        type=int,
        default=2
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