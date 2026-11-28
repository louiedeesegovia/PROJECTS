import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

cadb_images_dir = Path.home() / "thesis/data/CADB_Dataset/images"
meta_path = Path.home() / "thesis/selected_cadb_metadata.json"
out_dir = Path.home() / "thesis/fig3_2_saliency"
out_dir.mkdir(parents=True, exist_ok=True)


def compute_spectral_residual_saliency(
    img_path: Path,
    avg_kernel_size: int = 3,
    gauss_sigma: float = 4.0,
    out_size: tuple[int, int] = (224, 224),
    clip_percentile: float = 95.0,
) -> np.ndarray:
    # 1. Grayscale and downsample to 1/6 of original size
    with Image.open(img_path) as im:
        gray = im.convert("L")
        w, h = gray.size
        down_w = max(1, w // 6)
        down_h = max(1, h // 6)
        small_gray = np.asarray(
            gray.resize((down_w, down_h), Image.Resampling.BILINEAR),
            dtype=np.float64,
        )

    # 2. 2D FFT -> log amplitude -> subtract local average -> inverse FFT -> squared magnitude
    fft_vals = np.fft.fft2(small_gray)
    fft_shifted = np.fft.fftshift(fft_vals)

    amp = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)

    eps = 1e-8
    log_amp = np.log(amp + eps)
    local_avg = uniform_filter(log_amp, size=avg_kernel_size, mode="reflect")
    spectral_residual = log_amp - local_avg

    recon_shifted = np.exp(spectral_residual + 1j * phase)
    recon_unshifted = np.fft.ifftshift(recon_shifted)
    spatial_complex = np.fft.ifft2(recon_unshifted)
    saliency_small = np.abs(spatial_complex) ** 2

    # 3. Smooth with Gaussian filter, resize to 224x224, clip at 95th percentile, normalize to [0, 1]
    if gauss_sigma > 0:
        saliency_small = gaussian_filter(saliency_small, sigma=gauss_sigma, mode="reflect")

    # Resize to (224, 224) using PIL (width=224, height=224)
    sal_resized_pil = Image.fromarray(saliency_small.astype(np.float32), mode="F").resize(
        out_size, Image.Resampling.BILINEAR
    )
    sal = np.asarray(sal_resized_pil, dtype=np.float32)

    p95 = np.percentile(sal, clip_percentile)
    sal_clipped = np.clip(sal, sal.min(), p95)

    denom = float(sal_clipped.max() - sal_clipped.min())
    if denom > 1e-8:
        sal_norm = (sal_clipped - sal_clipped.min()) / denom
    else:
        sal_norm = np.zeros_like(sal_clipped, dtype=np.float32)

    return np.clip(sal_norm, 0.0, 1.0)


def main():
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            items = [row["image"] for row in json.load(f)]
    else:
        items = ["12.jpg", "5511.jpg", "5624.jpg", "6421.jpg", "6341.jpg", "8768.jpg", "8824.jpg"]

    for img_name in items:
        img_path = cadb_images_dir / img_name
        sal_map = compute_spectral_residual_saliency(img_path)

        np.save(out_dir / f"{Path(img_name).stem}_saliency.npy", sal_map)
        # Also save a grayscale PNG preview for quick visual inspection
        png_uint8 = (sal_map * 255.0).round().astype(np.uint8)
        Image.fromarray(png_uint8, mode="L").save(out_dir / f"{Path(img_name).stem}_saliency.png")

        print(
            f"{img_name:10s} | shape={sal_map.shape} | "
            f"min={sal_map.min():.4f} | max={sal_map.max():.4f} | mean={sal_map.mean():.4f}"
        )


if __name__ == "__main__":
    main()
