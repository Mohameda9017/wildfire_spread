from __future__ import annotations

from pathlib import Path
import json

import tensorflow as tf

from src.preprocess import build_dataset
from src.models.logistic_baseline import build_logistic_baseline
from src.training.losses import masked_binary_crossentropy, masked_binary_accuracy
from src.training.metrics import (masked_precision, masked_recall, masked_f1, masked_iou)


TRAIN_PATTERN = "data/raw/next_day_wildfire_spread/next_day_wildfire_spread_train*.tfrecord"
VAL_PATTERN = "data/raw/next_day_wildfire_spread/next_day_wildfire_spread_eval*.tfrecord"
TEST_PATTERN = "data/raw/next_day_wildfire_spread/next_day_wildfire_spread_test*.tfrecord"


def main() -> None:
    # Create output folders
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Build datasets
    train_ds = build_dataset(
        file_pattern=TRAIN_PATTERN,
        batch_size=16,
        clip_and_normalize=False,
        clip_and_rescale=False,
        shuffle=True,
        repeat=True,
    )

    val_ds = build_dataset(
        file_pattern=VAL_PATTERN,
        batch_size=16,
        clip_and_normalize=False,
        clip_and_rescale=False,
        shuffle=False,
        repeat=False,
    )

    test_ds = build_dataset(
        file_pattern=TEST_PATTERN,
        batch_size=16,
        clip_and_normalize=False,
        clip_and_rescale=False,
        shuffle=False,
        repeat=False,
    )

    inputs, labels = next(iter(train_ds))
    print("Train inputs shape:", inputs.shape)
    print("Train labels shape:", labels.shape)

    # build model
    model = build_logistic_baseline(input_shape=(64, 64, 12))
    model.summary()

    preds = model(inputs)
    print("Predictions shape:", preds.shape)
    print("Prediction min:", tf.reduce_min(preds).numpy())
    print("Prediction max:", tf.reduce_max(preds).numpy())

    loss = masked_binary_crossentropy(labels, preds)
    acc = masked_binary_accuracy(labels, preds)

    print("Initial masked BCE:", float(loss.numpy()))
    print("Initial masked accuracy:", float(acc.numpy()))

    # complie the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=masked_binary_crossentropy,
        metrics=[masked_binary_accuracy, masked_precision, masked_recall, masked_f1, masked_iou],
    )

    print("Model compiled successfully.")

    # Callbacks are functions that are called during the training process at certain points, such as at the end of each epoch.
    # They allow us to perform actions like early stopping, saving the best model, and logging metrics during training.

    # stop training if validation loss does not improve for 3 consecutive epochs, and restore the best weights found during training
    # this prevents overfitting and ensures we keep the best model found during training.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    # save the model weights whenever there is an improvement in validation loss. 
    # This allows us to keep the best model found during training, even if we later restore weights from early stopping.
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="models/logistic_baseline_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # logs epoch-by-epoch metrics into a csv file 
    csv_logger = tf.keras.callbacks.CSVLogger(
        filename="logs/logistic_baseline_metrics.csv",
        append=False,
    )

    # training the model
    history = model.fit(
        train_ds,
        validation_data=val_ds, # validation data is used to evaluate the model at the end of each epoch, such as overfitting
        epochs=15,
        steps_per_epoch=937,
        callbacks=[early_stopping, model_checkpoint, csv_logger],
        verbose=1,
    )

    # Save final model
    model.save("models/logistic_baseline_final.keras")
    print("Final model saved to models/logistic_baseline_final.keras")

    # Save training history
    with open("logs/logistic_baseline_history.json", "w") as f:
        json.dump(history.history, f, indent=2)

    print("Training history saved to logs/logistic_baseline_history.json")

    # Load best saved model for final test evaluation
    best_model = tf.keras.models.load_model(
        "models/logistic_baseline_best.keras",
        custom_objects={
        "masked_binary_crossentropy": masked_binary_crossentropy,
        "masked_binary_accuracy": masked_binary_accuracy,
        "masked_precision": masked_precision,
        "masked_recall": masked_recall,
        "masked_f1": masked_f1,
        "masked_iou": masked_iou,
        },
    )

    print("\nEvaluating best saved model on test set...")
    test_results = best_model.evaluate(test_ds, verbose=1)

    for name, value in zip(best_model.metrics_names, test_results):
        print(f"Test {name}: {value:.6f}")


if __name__ == "__main__":
    main()