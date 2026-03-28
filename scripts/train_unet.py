import os
import sys
from pathlib import Path

# Setup GPU libraries for pip-installed CUDA/cuDNN before ANY other imports
try:
    project_root = str(Path(__file__).resolve().parents[1])
    sys.path.append(project_root)
    import src.utils.gpu
except ImportError:
    pass

# Ensure the project root is in sys.path (already done above but keeping for clarity)
if project_root not in sys.path:
    sys.path.append(project_root)

# Set the framework before importing segmentation_models
os.environ['SM_FRAMEWORK'] = 'tf.keras'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import json
import numpy as np
import tensorflow as tf

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPUs")
    except RuntimeError as e:
        print(f"Memory growth error: {e}")
import segmentation_models as sm

from src.preprocess import build_dataset
from src.training.losses import (
    weighted_masked_binary_crossentropy,
    masked_binary_accuracy,
)
from src.training.metrics import (
    masked_precision,
    masked_recall,
    masked_f1,
    masked_iou,
    masked_ap,
)

# Configuration
TRAIN_PATTERN = "data/raw/next_day_wildfire_spread/next_day_wildfire_spread_train*.tfrecord"
VAL_PATTERN = "data/raw/next_day_wildfire_spread/next_day_wildfire_spread_eval*.tfrecord"
TEST_PATTERN = "data/raw/next_day_wildfire_spread/next_day_wildfire_spread_test*.tfrecord"

def unet_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Wrapper loss for the U-Net so Keras can save/load the model.
    """
    return weighted_masked_binary_crossentropy(
        y_true,
        y_pred,
        pos_weight=150.0,
        neg_weight=1.0,
    )

def build_unet_12ch(backbone_name='resnet18', input_shape=(64, 64, 12)):
    """
    Builds a U-Net model with 12-channel input and pretrained ImageNet weights.
    The first layer weights are tiled/averaged to handle 12 channels instead of 3.
    """
    print(f"Building U-Net with backbone: {backbone_name} and {input_shape[2]} channels...")
    
    # 1. Instantiate 3-channel model with ImageNet weights to get pretrained parameters
    model_3ch = sm.Unet(
        backbone_name=backbone_name,
        encoder_weights='imagenet',
        input_shape=(input_shape[0], input_shape[1], 3),
        classes=1,
        activation='sigmoid'
    )
    
    # 2. Instantiate 12-channel model architecture without weights
    model_12ch = sm.Unet(
        backbone_name=backbone_name,
        encoder_weights=None,
        input_shape=input_shape,
        classes=1,
        activation='sigmoid'
    )
    
    # 3. Synchronize weights
    print("Adapting first layer weights and copying others...")
    for i, (l12, l3) in enumerate(zip(model_12ch.layers, model_3ch.layers)):
        weights_3ch = l3.get_weights()
        if not weights_3ch:
            continue
            
        weights_12ch = l12.get_weights()
        
        # Check if this is a layer where channels mismatch
        if weights_3ch[0].shape != weights_12ch[0].shape:
            print(f"Matched channel mismatch at layer {i}: {l12.name}")
            new_weights = []
            for w3, w12 in zip(weights_3ch, weights_12ch):
                if w3.shape == w12.shape:
                    new_weights.append(w3)
                    continue
                
                print(f"  Adapting weights: {w3.shape} -> {w12.shape}")
                if len(w3.shape) == 4: # Conv2D weights: (H, W, C_in, C_out)
                    w_new = np.zeros(w12.shape, dtype=w3.dtype)
                    in_ch_old = w3.shape[2]
                    in_ch_new = w12.shape[2]
                    for c in range(in_ch_new):
                        w_new[:, :, c, :] = w3[:, :, c % in_ch_old, :]
                    new_weights.append(w_new)
                elif len(w3.shape) == 1: # BN weights or biases: (C,)
                    w_new = np.zeros(w12.shape, dtype=w3.dtype)
                    in_ch_old = w3.shape[0]
                    in_ch_new = w12.shape[0]
                    for c in range(in_ch_new):
                        w_new[c] = w3[c % in_ch_old]
                    new_weights.append(w_new)
                else:
                    print(f"  WARNING: Unexpected weight shape mismatch for layer {l12.name}: {w3.shape}")
                    new_weights.append(w12) # Fallback to random
            
            l12.set_weights(new_weights)
        else:
            # For all other layers, simply copy the weights
            l12.set_weights(weights_3ch)
            
    return model_12ch

def compile_model(model: tf.keras.Model) -> None:
    """
    Compiles the model with the same optimizer and metrics as the CNN script.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=unet_loss,
        metrics=[
            masked_binary_accuracy,
            masked_precision,
            masked_recall,
            masked_f1,
            masked_iou,
            masked_ap,
        ],
    )

def main() -> None:
    # Setup absolute paths
    model_dir = Path(project_root) / "models"
    log_dir = Path(project_root) / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_dir / "unet_resnet18_best.keras"
    final_model_path = model_dir / "unet_resnet18_final.keras"
    csv_log_path = log_dir / "unet_resnet18_metrics.csv"
    history_json_path = log_dir / "unet_resnet18_history.json"

    # Build datasets - using same params as CNN for consistency
    train_ds = build_dataset(
        file_pattern=TRAIN_PATTERN,
        batch_size=8,
        clip_and_normalize=True,
        clip_and_rescale=False,
        shuffle=True,
        repeat=True,
    )

    val_ds = build_dataset(
        file_pattern=VAL_PATTERN,
        batch_size=8,
        clip_and_normalize=True,
        clip_and_rescale=False,
        shuffle=False,
        repeat=True,
    )

    test_ds = build_dataset(
        file_pattern=TEST_PATTERN,
        batch_size=8,
        clip_and_normalize=True,
        clip_and_rescale=False,
        shuffle=False,
        repeat=False,
    )

    initial_epoch = 0
    if csv_log_path.exists():
        try:
            with open(csv_log_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].split(',')
                    initial_epoch = int(last_line[0]) + 1
                    print(f"\n[INFO] Found previous logs. Resuming from epoch {initial_epoch}")
        except Exception as e:
            print(f"[WARNING] Could not parse CSV for initial_epoch: {e}")

    # Build and compile model
    if best_model_path.exists():
        print("\n" + "="*50)
        print(f"[CHECKPOINT] Found existing best model at: {best_model_path}")
        print("[CHECKPOINT] Loading weights into model...")
        sys.stdout.flush() # Ensure it prints before the long load
        
        # We load with compile=False to avoid issues with custom objects during load
        model = tf.keras.models.load_model(str(best_model_path), compile=False)
        
        print("[CHECKPOINT] Model LOADED SUCCESSFULLY.")
        print("="*50 + "\n")
        sys.stdout.flush()
    else:
        print("\n" + "="*50)
        print(f"[CHECKPOINT] No previous checkpoint found at {best_model_path}")
        print("[CHECKPOINT] Starting with a FRESH model.")
        print("="*50 + "\n")
        sys.stdout.flush()
        model = build_unet_12ch(backbone_name='resnet18', input_shape=(64, 64, 12))
        
    compile_model(model)
    print("Model ready and compiled.")
    sys.stdout.flush()
    model.summary()

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_masked_ap",
        mode="max",
        patience=100,
        restore_best_weights=True,
        verbose=1,
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_masked_ap",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=str(csv_log_path),
        append=True, # Set to True to keep previous logs when resuming
    )

    # Training
    # Step counts for progress bar (derived from 14,979 train / 1,877 val examples)
    steps_per_epoch = 14979 // 8
    validation_steps = 1877 // 8

    # Training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=10000,
        initial_epoch=initial_epoch,
        callbacks=[early_stopping, model_checkpoint, csv_logger],
        verbose=2, # Use 2 for cleaner "streamlined" output like the CNN model
    )

    # Save final model
    model.save(str(final_model_path))
    print(f"Final model saved to {final_model_path}")

    with open(str(history_json_path), "w") as f:
        json.dump(history.history, f, indent=2)

    print("\nEvaluating best saved model on test set...")
    best_model = tf.keras.models.load_model(
        str(best_model_path),
        compile=False,
    )
    compile_model(best_model)
    test_results = best_model.evaluate(test_ds, verbose=2)

    for name, value in zip(best_model.metrics_names, test_results):
        print(f"Test {name}: {value:.6f}")

if __name__ == "__main__":
    main()
