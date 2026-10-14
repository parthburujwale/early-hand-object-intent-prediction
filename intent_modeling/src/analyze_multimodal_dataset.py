from pathlib import Path
import numpy as np


DATA_ROOT = Path(
    "/ocean/projects/bio260013p/pburujwale/"
    "intent_prediction/intent_modeling/outputs/multimodal"
)


def inspect(split, fraction):
    path = DATA_ROOT / f"{split}_{fraction:03d}.npz"

    data = np.load(
        path,
        allow_pickle=True
    )

    print()
    print("=" * 70)
    print(f"{split.upper()} {fraction}%")
    print("=" * 70)

    for name in [
        "X_hand",
        "X_object",
        "X_relative",
    ]:
        X = data[name]

        print()
        print(name)
        print("Shape:", X.shape)
        print("Min:", float(X.min()))
        print("Max:", float(X.max()))
        print("Mean:", float(X.mean()))
        print("Std:", float(X.std()))
        print(
            "NaN:",
            int(np.isnan(X).sum())
        )
        print(
            "Inf:",
            int(np.isinf(X).sum())
        )

    print()
    print(
        "Unique labels:",
        len(np.unique(data["y"]))
    )

    print(
        "Unique objects:",
        np.unique(data["object_id"])
    )


if __name__ == "__main__":
    inspect("train", 10)
    inspect("train", 100)

    inspect("val", 10)
    inspect("val", 100)