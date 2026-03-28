from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

INPUT_FEATURES = [
    "elevation",
    "th",
    "vs",
    "tmmn",
    "tmmx",
    "sph",
    "pr",
    "pdsi",
    "NDVI",
    "population",
    "erc",
    "PrevFireMask",
]


def visualize_sample(
    sample_path: str = "data/processed/sample_1.npz",
    output_path: str = "data/sample_outputs/sample_1_visualization.png",
) -> None:
    data = np.load(sample_path)
    x = data["x"]   # (64, 64, 12)
    y = data["y"]   # (64, 64, 1)

    # 4 continuous channels + 2 masks
    selected_features = [
        "elevation",
        "vs",
        "pdsi",
        "NDVI",
        "PrevFireMask",
        "FireMask",
    ]

    titles = {
        "elevation": "Elevation",
        "vs": "Wind speed",
        "pdsi": "Drought index",
        "NDVI": "NDVI",
        "PrevFireMask": "Previous fire mask",
        "FireMask": "Next-day fire mask",
    }

    # Fire mask colors: -1 unlabeled, 0 no fire, 1 fire
    fire_cmap = colors.ListedColormap(["black", "lightgray", "orangered"])
    fire_bounds = [-1, -0.1, 0.001, 1]
    fire_norm = colors.BoundaryNorm(fire_bounds, fire_cmap.N)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for i, feature_name in enumerate(selected_features):
        ax = axes[i]

        if feature_name == "FireMask":
            im = ax.imshow(y[:, :, 0], cmap=fire_cmap, norm=fire_norm)
        else:
            channel_idx = INPUT_FEATURES.index(feature_name)
            channel = x[:, :, channel_idx]

            if feature_name == "PrevFireMask":
                im = ax.imshow(channel, cmap=fire_cmap, norm=fire_norm)
            else:
                im = ax.imshow(channel, cmap="viridis")
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)

        ax.set_title(titles[feature_name], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to {output_path}")