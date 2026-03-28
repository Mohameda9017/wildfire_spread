```python
from __future__ import annotations

import os
import sys
from pathlib import Path
import json

# Setup GPU libraries for pip-installed CUDA/cuDNN before ANY other imports
try:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.append(project_root)
    import src.utils.gpu
    src.utils.gpu.setup_gpu_libraries()
except ImportError:
    pass

import tensorflow as tf

from src.preprocess import build_dataset
from src.models.cnn_baseline import build_simple_cnn
from src.training.losses import (
    weighted_masked_binary_crossentropy,
    masked_binary_accuracy,
)
from src.training.metrics import (
    masked_precision,
    masked_recall,
    masked_f1,
    masked_iou,
)


TRAIN_PATTERN = "data/raw/next_day_wildfire_spread/next_day_wildfire_spread_train*.tfrecord"
VAL_PATTERN = "data/raw/next_day_wildfire_spread/next_day_wildfire_spread_eval*.tfrecord"
TEST_PATTERN = "data/raw/next_day_wildfire_spread/next_day_wildfire_spread_test*.tfrecord"


def cnn_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Wrapper loss for the CNN so Keras can save/load the model
    without running into functools.partial serialization issues.
    """
    return weighted_masked_binary_crossentropy(
        y_true,
        y_pred,
        pos_weight=150.0,
        neg_weight=1.0,
    )


def compile_model(model: tf.keras.Model) -> None:
    """
    Compiles the model with the shared optimizer, loss, and metrics.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=cnn_loss,
        metrics=[
            masked_binary_accuracy,
            masked_precision,
            masked_recall,
            masked_f1,
            masked_iou,
        ],
    )


def main() -> None:
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    train_ds = build_dataset(
        file_pattern=TRAIN_PATTERN,
        batch_size=8,
        clip_and_normalize=True,
        clip_and_rescale=False,
        shuffle=True,
        repeat=False,
    )

    val_ds = build_dataset(
        file_pattern=VAL_PATTERN,
        batch_size=8,
        clip_and_normalize=True,
        clip_and_rescale=False,
        shuffle=False,
        repeat=False,
    )

    test_ds = build_dataset(
        file_pattern=TEST_PATTERN,
        batch_size=8,
        clip_and_normalize=True,
        clip_and_rescale=False,
        shuffle=False,
        repeat=False,
    )

    inputs, labels = next(iter(train_ds))
    print("Train inputs shape:", inputs.shape)
    print("Train labels shape:", labels.shape)

    model = build_simple_cnn(input_shape=(64, 64, 12))
    model.summary()

    preds = model(inputs)
    print("labels min/max:", tf.reduce_min(labels).numpy(), tf.reduce_max(labels).numpy())
    print("preds min/max/mean:", tf.reduce_min(preds).numpy(), tf.reduce_max(preds).numpy(), tf.reduce_mean(preds).numpy())

    valid_mask = tf.cast(tf.not_equal(labels, -1.0), tf.float32)
    print("valid fraction:", tf.reduce_mean(valid_mask).numpy())

    labels_clean = tf.where(tf.equal(labels, -1.0), 0.0, labels)
    print("true positive fraction:", tf.reduce_mean(labels_clean).numpy())

    pred_bin_05 = tf.cast(preds >= 0.5, tf.float32)
    pred_bin_01 = tf.cast(preds >= 0.1, tf.float32)

    print("predicted positive fraction @0.5:", tf.reduce_mean(pred_bin_05).numpy())
    print("predicted positive fraction @0.1:", tf.reduce_mean(pred_bin_01).numpy())

    tp_05 = tf.reduce_sum(pred_bin_05 * labels_clean * valid_mask)
    tp_01 = tf.reduce_sum(pred_bin_01 * labels_clean * valid_mask)

    print("TP @0.5:", tp_05.numpy())
    print("TP @0.1:", tp_01.numpy())

    compile_model(model)
    print("Model compiled successfully.")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_masked_f1",
        mode = "max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="models/simple_cnn_best.keras",
        monitor="val_masked_f1",
        mode = "max",
        save_best_only=True,
        verbose=1,
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        filename="logs/simple_cnn_metrics.csv",
        append=False,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[early_stopping, model_checkpoint, csv_logger],
        verbose=2,
    )

    model.save("models/simple_cnn_final.keras")
    print("Final model saved to models/simple_cnn_final.keras")

    with open("logs/simple_cnn_history.json", "w") as f:
        json.dump(history.history, f, indent=2)

    print("Training history saved to logs/simple_cnn_history.json")

    # Debug one validation batch after training
    print("\nInspecting one validation batch...")
    x_batch, y_batch = next(iter(val_ds))
    pred = model.predict(x_batch, verbose=0)

    y_np = y_batch.numpy()
    valid_mask_np = (y_np != -1.0)

    pred_bin_05 = (pred >= 0.5).astype("float32")
    pred_bin_01 = (pred >= 0.1).astype("float32")

    print("pred min:", pred.min())
    print("pred max:", pred.max())
    print("pred mean:", pred.mean())
    print("predicted fire pixels @0.5:", (pred_bin_05 * valid_mask_np).sum())
    print("predicted fire pixels @0.1:", (pred_bin_01 * valid_mask_np).sum())
    print("true fire pixels:", ((y_np == 1.0) & valid_mask_np).sum())

    best_model = tf.keras.models.load_model(
        "models/simple_cnn_best.keras",
        compile=False,
    )
    compile_model(best_model)

    print("\nEvaluating best saved model on test set...")
    test_results = best_model.evaluate(test_ds, verbose=2)

    for name, value in zip(best_model.metrics_names, test_results):
        print(f"Test {name}: {value:.6f}")


if __name__ == "__main__":
    main()