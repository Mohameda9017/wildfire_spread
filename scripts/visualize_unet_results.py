import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Setup project root and imports
try:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.append(project_root)
    import src.utils.gpu
except ImportError:
    project_root = os.getcwd()

from src.preprocess import build_dataset
from src.training.losses import masked_binary_accuracy
from src.training.metrics import (
    masked_precision,
    masked_recall,
    masked_f1,
    masked_iou,
    masked_ap,
)

# Configuration
TEST_PATTERN = "data/raw/next_day_wildfire_spread/next_day_wildfire_spread_test*.tfrecord"
MODEL_PATH = Path(project_root) / "models" / "unet_resnet18_best.keras"
OUTPUT_DIR = Path(project_root) / "logs" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_best_model(model_path):
    print(f"Loading model from {model_path}...")
    # Load with compile=False to avoid custom object issues during load
    model = tf.keras.models.load_model(str(model_path), compile=False)
    return model

def visualize_samples(model, dataset, num_samples=5):
    print(f"Visualizing {num_samples} samples...")
    
    # Get a batch of data
    for x_batch, y_batch in dataset.take(1):
        preds = model.predict(x_batch)
        
        for i in range(min(num_samples, x_batch.shape[0])):
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            # Input features (index 11 is PrevFireMask, 0 is elevation, 8 is NDVI)
            prev_fire = x_batch[i, :, :, 11].numpy()
            elevation = x_batch[i, :, :, 0].numpy()
            ndvi = x_batch[i, :, :, 8].numpy()
            
            # Ground truth and prediction
            gt = y_batch[i, :, :, 0].numpy()
            pred_prob = preds[i, :, :, 0]
            pred_binary = (pred_prob >= 0.5).astype(float)
            
            # Mask out the -1 values in ground truth for better visualization
            gt_masked = np.ma.masked_where(gt == -1, gt)
            
            # Plotting
            im0 = axes[0].imshow(elevation, cmap='terrain')
            axes[0].set_title("Elevation")
            plt.colorbar(im0, ax=axes[0])
            
            im1 = axes[1].imshow(prev_fire, cmap='YlOrRd')
            axes[1].set_title("Previous Day Fire")
            plt.colorbar(im1, ax=axes[1])
            
            im2 = axes[2].imshow(gt_masked, cmap='YlOrRd', vmin=0, vmax=1)
            axes[2].set_title("Actual Next Day Fire")
            plt.colorbar(im2, ax=axes[2])
            
            im3 = axes[3].imshow(pred_binary, cmap='YlOrRd', vmin=0, vmax=1)
            axes[3].set_title("Predicted Fire (0.5 threshold)")
            plt.colorbar(im3, ax=axes[3])
            
            # Error map (binary pred vs GT)
            error = np.zeros_like(gt)
            # 1: Correct Fire (TP), 2: False Positive (FP), 3: False Negative (FN), 0: Correct Non-Fire (TN)
            error[(gt == 1) & (pred_binary == 1)] = 1 # TP
            error[(gt == 0) & (pred_binary == 1)] = 2 # FP
            error[(gt == 1) & (pred_binary == 0)] = 3 # FN
            error_masked = np.ma.masked_where(gt == -1, error)
            
            im4 = axes[4].imshow(error_masked, cmap='tab10', vmin=0, vmax=9)
            axes[4].set_title("Error Map (TP=1, FP=2, FN=3)")
            plt.colorbar(im4, ax=axes[4])
            
            for ax in axes:
                ax.axis('off')
            
            plt.tight_layout()
            output_path = OUTPUT_DIR / f"prediction_sample_{i}.png"
            plt.savefig(output_path)
            print(f"Saved visualization to {output_path}")
            plt.close()

def main():
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # Build dataset
    test_ds = build_dataset(
        file_pattern=TEST_PATTERN,
        batch_size=8,
        clip_and_normalize=True,
        clip_and_rescale=False,
        shuffle=True,
        repeat=False,
    )

    model = load_best_model(MODEL_PATH)
    visualize_samples(model, test_ds)

if __name__ == "__main__":
    main()
