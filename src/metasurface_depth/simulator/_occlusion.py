import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


def extend_edge_and_boundaries(
    gray: np.ndarray,
    depth: np.ndarray,
    n: int = 35,
    edge_threshold: float = 0.1,
    return_depth_nan_outside: bool = True,
    depth_threshold: float = 0.01,
    vis_name: str = "",
):
    """
    Expand along depth edges and image borders (outward), on an expanded canvas.

    Args:
      gray:  (H,W) grayscale, any numeric scale (kept as-is)
      depth: (H,W) depth map, float
      n:     pixels to extend along the normal (interior) / outward (borders)
      visualize: save figure

    Returns (all on expanded canvas of size (H+2n, W+2n); original sits at [n:n+H, n:n+W]):
      ext_mask  : (H2,W2) uint8   == 1 only where newly filled (interior band & outward border)
      ext_gray  : (H2,W2) float32 == gray values at filled pixels (0 elsewhere)
      ext_depth : (H2,W2) float32 == depth values at filled pixels (NaN or 0 elsewhere)
      ext is NaN outside newly filled pixels (else 0)
    """

    assert gray.shape == depth.shape, "gray and depth must have same shape"
    H, W = depth.shape

    # --- 1. normalize depth for processing ---
    d_min, d_max = depth.min(), depth.max()
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-9)

    # --- 2. compute gradients & edge mask ---
    grad_x = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    edges = (grad_mag > edge_threshold).astype(np.uint8)

    # --- 3. morphological smoothing of edge map ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    bg_depth = np.full((H, W), np.nan, dtype=np.float32)
    bg_gray = np.full((H, W), np.nan, dtype=np.float32)

    # --- 4. background (farther) depth for each edge pixel ---
    grad_vec = np.stack([grad_y, grad_x], axis=-1)
    norm = np.linalg.norm(grad_vec, axis=-1, keepdims=True) + 1e-9
    grad_dir = grad_vec / norm
    edge_coords = np.column_stack(np.where(edges > 0))

    step_size = 0.5

    for y, x in edge_coords:
        yy, xx = float(y), float(x)
        dir_y, dir_x = grad_dir[y, x]

        local_max_depth = -1
        local_gray = np.nan
        for _ in range(int(n / step_size)):
            y0, x0 = int(round(yy)), int(round(xx))
            y0 = np.clip(y0, 1, H - 2)
            x0 = np.clip(x0, 1, W - 2)
            patch_depth = depth_norm[y0 - 1 : y0 + 1, x0 - 1 : x0 + 1]
            patch_gray = gray[y0 - 1 : y0 + 1, x0 - 1 : x0 + 1]
            max_idx = np.unravel_index(np.argmax(patch_depth), patch_depth.shape)
            if patch_depth[max_idx] > local_max_depth:
                local_max_depth = patch_depth[max_idx]
                local_gray = patch_gray[max_idx]
            if edges[y0, x0] == 0:
                break
            yy += dir_y * step_size
            xx += dir_x * step_size

        bg_depth[y, x] = local_max_depth if local_max_depth >= 0 else depth_norm[y, x]
        bg_gray[y, x] = local_gray if not np.isnan(local_gray) else gray[y, x]

    # --- 5. compute dilation band ---
    dist = distance_transform_edt(1 - edges)
    dilated_mask = (dist <= n).astype(np.uint8)

    # --- 6. extend background values over mask ---
    valid_bg = ~np.isnan(bg_depth)
    depth_dilated = np.zeros_like(depth_norm)
    gray_dilated = np.zeros_like(gray)

    depth_dilated[valid_bg] = bg_depth[valid_bg]
    gray_dilated[valid_bg] = bg_gray[valid_bg]

    kernel_size = n * 2 + 1
    sigma = n / 2.0
    gaussian_1d = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = gaussian_1d @ gaussian_1d.T
    kernel /= kernel.sum()

    depth_sum = cv2.filter2D(depth_dilated, -1, kernel)
    gray_sum = cv2.filter2D(gray_dilated, -1, kernel)
    valid_mask = np.zeros_like(depth_norm)
    valid_mask[valid_bg] = 1
    count_sum = cv2.filter2D(valid_mask, -1, kernel)

    valid = count_sum > 1e-6
    dilated_depth = np.zeros_like(depth_norm)
    dilated_gray = np.zeros_like(gray)
    dilated_depth[valid] = depth_sum[valid] / count_sum[valid]
    dilated_gray[valid] = gray_sum[valid] / count_sum[valid]

    # 7. Keep only dilated pixels significantly bigger (deeper) than original depth
    ext_mask = valid & (dilated_depth > depth_norm * (1 + depth_threshold))


    ext_depth = np.zeros_like(dilated_depth, dtype=np.float32)
    ext_gray = np.zeros_like(dilated_gray, dtype=np.float32)
    ext_depth[ext_mask] = dilated_depth[ext_mask]
    ext_gray[ext_mask] = dilated_gray[ext_mask]

    if return_depth_nan_outside:
        ext_depth[~ext_mask] = np.nan

    # --- 8. visualization ---
    if vis_name != "":
        fig, axs = plt.subplots(2, 4, figsize=(22, 8))
        vmin, vmax = np.nanmin(depth_norm), np.nanmax(depth_norm)

        # --- Row 1: Depth-related visualizations ---
        axs[0, 0].imshow(depth_norm, cmap="Spectral", vmin=vmin, vmax=vmax)
        axs[0, 0].set_title("Original Depth")

        axs[0, 1].imshow(edges, cmap="gray")
        axs[0, 1].set_title("Detected Edges")

        axs[0, 2].imshow(ext_depth, cmap="Spectral", vmin=vmin, vmax=vmax)
        axs[0, 2].set_title("Extended Depth")

        # Overlay: extended depth on original depth
        overlay_depth = depth_norm.copy()
        overlay_depth[ext_mask] = ext_depth[ext_mask]
        axs[0, 3].imshow(overlay_depth, cmap="Spectral", vmin=vmin, vmax=vmax)
        axs[0, 3].set_title("Overlay: Extended on Depth")

        # --- Row 2: Gray-related visualizations ---
        axs[1, 0].imshow(gray, cmap="gray", vmin=0, vmax=1)
        axs[1, 0].set_title("Original Gray")

        axs[1, 1].imshow(ext_mask, cmap="gray", vmin=0, vmax=1)
        axs[1, 1].set_title("Dilate Shape")

        axs[1, 2].imshow(ext_gray, cmap="gray", vmin=0, vmax=1)
        axs[1, 2].set_title("Extended Gray")

        overlay_gray = gray.copy()
        overlay_gray[ext_mask] = ext_gray[ext_mask]
        axs[1, 3].imshow(overlay_gray, cmap="gray", vmin=0, vmax=1)
        axs[1, 3].set_title("Overlay: Extended on Gray")

        for ax in axs.flat:
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(2)
        plt.tight_layout()
        plt.savefig(vis_name, dpi=300)
        plt.close()

    ext_depth = ext_depth * (d_max - d_min) + d_min
    return (
        ext_mask.astype(np.uint8),
        ext_gray.astype(np.float32),
        ext_depth.astype(np.float32),
    )
