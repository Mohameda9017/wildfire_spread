# scripts/run_preprocessing.py

from pathlib import Path

import tensorflow as tf
import numpy as np

from src.preprocess import build_dataset

TRAIN_PATTERN = "data/raw/next_day_wildfire_spread/next_day_wildfire_spread_train*.tfrecord"

def main() -> None:
    dataset = build_dataset(
        file_pattern=TRAIN_PATTERN,
        batch_size=4,
        clip_and_normalize=False,
        clip_and_rescale=False,
        shuffle=False,
    )

    inputs, labels = next(iter(dataset))

    print("Inputs shape:", inputs.shape)   # expected: (4, 64, 64, 12)
    print("Labels shape:", labels.shape)   # expected: (4, 64, 64, 1)
    print("Inputs dtype:", inputs.dtype)
    print("Labels dtype:", labels.dtype)

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    fire_sizes = [labels[i].numpy().sum() for i in range(inputs.shape[0])]
    best_idx = int(np.argmax(fire_sizes))

    sample_x = inputs[best_idx].numpy()
    sample_y = labels[best_idx].numpy()

    np.savez(
        "data/processed/sample_1.npz",
        x=sample_x,
        y=sample_y,
    )

    print("Saved sample to data/processed/sample_1.npz")

    print("Preprocessing pipeline ran successfully.")

if __name__ == "__main__":
    main()