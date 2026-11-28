import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from find_fig4_7_candidates import extract_pred_map, gt_mean_from_cadb

cadb_images_dir = Path.home() / "thesis/data/CADB_Dataset/images"
resnet_preds_path = Path.home() / "thesis/results/resnet18_sched_s0/test_preds.json"
regnet_preds_path = (
    Path.home()
    / "thesis/Image-Composition-Assessment-Dataset-CADB/SAMPNet/runs_sched/regnet_x_3_2gf_p0_final_s0/test_preds.json"
)
out_png = Path.home() / "thesis/fig4_7_example_predictions_s0.png"
out_pdf = Path.home() / "thesis/fig4_7_example_predictions_s0.pdf"

groups = [
    ("Both close to ground truth", ["7769.jpg", "8470.jpg"]),
    ("RegNetX-3.2GF clearly better", ["8302.jpg", "6018.jpg"]),
    ("ResNet-18 clearly better", ["1603.jpg", "8068.jpg"]),
]

res_map = extract_pred_map(resnet_preds_path)
reg_map = extract_pred_map(regnet_preds_path)

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(11.5, 7.8),
    dpi=300,
    gridspec_kw={
        "hspace": 0.38,
        "wspace": 0.16,
        "top": 0.89,
        "bottom": 0.08,
        "left": 0.03,
        "right": 0.97,
    },
)

for col, (col_title, img_list) in enumerate(groups):
    for row, img_name in enumerate(img_list):
        ax = axes[row, col]
        with Image.open(cadb_images_dir / img_name) as im:
            rgb = np.asarray(im.convert("RGB"))
        ax.imshow(rgb)
        ax.set_xticks([])
        ax.set_yticks([])

        if row == 0:
            ax.set_title(col_title, fontsize=10.5, fontweight="bold", pad=10)

        gt_cadb = gt_mean_from_cadb(img_name)
        gt = gt_cadb if gt_cadb is not None else res_map[img_name][0]
        p_res = res_map[img_name][1]
        p_reg = reg_map[img_name][1]

        under_text = (
            f"{img_name}  |  GT: {gt:.2f}\n"
            f"ResNet-18: {p_res:.2f}  |  RegNetX-3.2GF: {p_reg:.2f}"
        )
        ax.text(
            0.5,
            -0.06,
            under_text,
            ha="center",
            va="top",
            fontsize=8.8,
            linespacing=1.35,
            transform=ax.transAxes,
        )

fig.savefig(out_png, dpi=300)
fig.savefig(out_pdf)
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
