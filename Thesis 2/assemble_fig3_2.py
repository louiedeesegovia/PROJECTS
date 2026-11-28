import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

cadb_images_dir = Path.home() / "thesis/data/CADB_Dataset/images"
meta_path = Path.home() / "thesis/selected_cadb_metadata.json"
sal_dir = Path.home() / "thesis/fig3_2_saliency"
out_fig_png = Path.home() / "thesis/fig3_2_cadb_saliency_grid.png"
out_fig_pdf = Path.home() / "thesis/fig3_2_cadb_saliency_grid.pdf"

with open(meta_path, "r", encoding="utf-8") as f:
    records = json.load(f)

n_items = len(records)
cmap_name = "inferno"  # Consistent colormap across all panels (e.g., 'inferno', 'magma', or 'gray')

# Vertical mini-stack per item:
# Row 0: Original RGB photo (aspect ratio preserved)
# Row 1: Spectral-residual saliency map (224x224)
# Row 2: Bottom caption (Scene category + Ground-truth mean score)
fig = plt.figure(figsize=(2.4 * n_items + 0.8, 6.2), dpi=300)
gs = fig.add_gridspec(
    nrows=3,
    ncols=n_items,
    height_ratios=[1.0, 1.0, 0.24],
    hspace=0.14,
    wspace=0.12,
    left=0.02,
    right=0.95,
    top=0.92,
    bottom=0.04,
)

last_im = None
for col, rec in enumerate(records):
    img_name = rec["image"]
    stem = Path(img_name).stem
    category = rec["category"]
    mean_score = rec["mean_score"]

    # 1. Top: Original RGB image (aspect ratio preserved)
    ax_rgb = fig.add_subplot(gs[0, col])
    with Image.open(cadb_images_dir / img_name) as im:
        rgb = np.asarray(im.convert("RGB"))
    ax_rgb.imshow(rgb)
    ax_rgb.set_xticks([])
    ax_rgb.set_yticks([])
    ax_rgb.set_title(img_name, fontsize=9.5, fontweight="bold", pad=4)

    # 2. Middle: Spectral-residual saliency map
    ax_sal = fig.add_subplot(gs[1, col])
    sal_path = sal_dir / f"{stem}_saliency.npy"
    sal_map = np.load(sal_path)
    last_im = ax_sal.imshow(sal_map, cmap=cmap_name, vmin=0.0, vmax=1.0, aspect="equal")
    ax_sal.set_xticks([])
    ax_sal.set_yticks([])

    # 3. Bottom label: Scene category and ground-truth mean score
    ax_lbl = fig.add_subplot(gs[2, col])
    ax_lbl.axis("off")
    score_str = f"{mean_score:.2f}" if mean_score is not None else "N/A"
    label_text = f"{category.capitalize()} | Mean: {score_str}"
    ax_lbl.text(
        0.5,
        0.55,
        label_text,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="medium",
        transform=ax_lbl.transAxes,
    )

# Optional shared colorbar on the right for the saliency map
if last_im is not None:
    cbar_ax = fig.add_axes([0.96, 0.28, 0.012, 0.58])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Saliency", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

fig.suptitle(
    "CADB Sample Images, Spectral-Residual Saliency Maps, and Ground-Truth Composition Scores",
    fontsize=11.5,
    fontweight="bold",
    y=0.98,
)

fig.savefig(out_fig_png, dpi=300)
fig.savefig(out_fig_pdf)
print(f"Saved PNG: {out_fig_png}")
print(f"Saved PDF: {out_fig_pdf}")
