"""
visual.py
─────────
Visualises per-token saliency maps derived from a posterior tensor.

Expected inputs
───────────────
    posterior : torch.Tensor | np.ndarray  shape (1, T, H*W)
                One spatial Grad-CAM / attention map per generated token.
                Produced externally by calculate_prior() or compute_gradcam().
    tokens    : list[str] | list[int]
                T token strings (or ids) — one label per map.
                Pass the raw token-id list from generate(); integer ids are
                decoded via processor.tokenizer if a processor is given,
                otherwise rendered as "<id>" strings.
    image     : PIL.Image.Image
                Original image in RGB format.
                Resized to config.IMAGE_SIZE when that constant is defined.
    processor : AutoProcessor (optional)
                Used only to decode integer token ids into readable labels.

Public API
──────────
  Main entry point:
    visualize(posterior, tokens, image, processor=None, ...)
        Decode token labels (if needed), render all plots, optionally save.
        Returns a dict with keys: figures, paths, tokens (decoded strings).

  Low-level (single token / single map):
    denormalize_image(image_tensor)
        Reverse ImageNet normalisation → (3, H, W) float tensor in [0, 1].

    show_grad_mask(patch_scores, ...)
        Grey-scale heat-map of one flat patch-score vector.

    show_heatmap_overlay(patch_scores, image, token_label, ...)
        4-panel figure: Original | Heatmap | Jet overlay | Pixel-masked.

  High-level (full posterior):
    plot_token_maps(posterior, tokens, image, ...)
        Grid of saliency maps, one cell per token.

    plot_overlay(posterior, tokens, image, ...)
        Same grid, always blended over the original image.

    plot_cumulative(posterior, tokens, image, ...)
        Mean saliency across all tokens — single summary map.

    plot_token_sequence(posterior, tokens, image, ...)
        Animated GIF stepping through tokens one by one.

    save_all(posterior, tokens, image, out_dir, ...)
        Convenience: writes all visualisations to *out_dir*.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from transformers import AutoProcessor

import config

# ── ImageNet denormalisation ──────────────────────────────────────────────────

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalisation.

    Args:
        image_tensor : (3, H, W) float tensor on any device.

    Returns:
        (3, H, W) float tensor with values clipped to [0, 1].
    """
    mean = _IMAGENET_MEAN.to(image_tensor.device)
    std  = _IMAGENET_STD.to(image_tensor.device)
    return (image_tensor * std + mean).clamp(0.0, 1.0)


# ── internal helpers ──────────────────────────────────────────────────────────

def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _squeeze_batch(posterior: np.ndarray) -> np.ndarray:
    """(1, T, P) or (T, P)  →  (T, P)."""
    if posterior.ndim == 3:
        if posterior.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got {posterior.shape[0]}.")
        return posterior[0]
    if posterior.ndim == 2:
        return posterior
    raise ValueError(f"posterior must be 2-D or 3-D, got shape {posterior.shape}")


def _infer_grid(num_patches: int) -> tuple[int, int]:
    """Infer (H_p, W_p) from a perfect-square patch count."""
    h = int(math.isqrt(num_patches))
    if h * h != num_patches:
        raise ValueError(
            f"num_patches={num_patches} is not a perfect square. "
            "Pass patch_grid=(H_p, W_p) explicitly."
        )
    return h, h


def _image_size_wh() -> tuple[int, int] | None:
    """Read config.IMAGE_SIZE → (W, H) tuple, or None."""
    size = getattr(config, "IMAGE_SIZE", None)
    if size is None:
        return None
    return (size, size) if isinstance(size, int) else tuple(size)


def _spatial_map(
    flat: np.ndarray,
    patch_grid: tuple[int, int],
    target_wh: tuple[int, int] | None,   # (W, H) output pixels
) -> np.ndarray:
    """
    Reshape (P,) → 2-D, normalise to [0, 1], bilinear-upsample via torch.
    Returns float32 array in [0, 1], shape (H_out, W_out).
    """
    h_p, w_p = patch_grid
    cam = flat.reshape(h_p, w_p)

    lo, hi = cam.min(), cam.max()
    cam = (cam - lo) / (hi - lo + 1e-8)

    if target_wh is not None:
        W, H = target_wh
        t = torch.tensor(cam).unsqueeze(0).unsqueeze(0)   # (1, 1, H_p, W_p)
        t = TF.resize(t, size=[H, W],
                      interpolation=TF.InterpolationMode.BILINEAR,
                      antialias=False)
        cam = t.squeeze().numpy()

    return cam.astype(np.float32)


def _pil_to_hwc(image: Image.Image) -> np.ndarray:
    """Resize PIL image to config.IMAGE_SIZE, return float32 (H,W,3) in [0,1]."""
    size_wh = _image_size_wh()
    if size_wh is not None:
        image = image.resize(size_wh, Image.BILINEAR)
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _img_wh(arr: np.ndarray) -> tuple[int, int]:
    """(H, W, 3) array → (W, H) for use as resize target."""
    return arr.shape[1], arr.shape[0]


def _token_label(t) -> str:
    return t.decode() if isinstance(t, bytes) else str(t)


def _grid_dims(n: int) -> tuple[int, int]:
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


# ── token label helpers ──────────────────────────────────────────────────────

def _resolve_token_labels(
    tokens: Sequence,
    processor: AutoProcessor | None,
) -> list[str]:
    """
    Convert raw tokens to display strings.

    Accepts:
      - list[str]  → returned as-is
      - list[int] / 1-D int tensor → decoded via processor.tokenizer when
        a processor is given, otherwise rendered as "<id>" strings
    """
    # unwrap tensor
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.squeeze().tolist()
        if isinstance(tokens, int):
            tokens = [tokens]

    labels: list[str] = []
    for t in tokens:
        if isinstance(t, (int, np.integer)):
            if processor is not None:
                decoded = processor.tokenizer.decode([int(t)], skip_special_tokens=True)
                labels.append(decoded if decoded.strip() else f"<{t}>")
            else:
                labels.append(f"<{t}>")
        else:
            labels.append(t.decode() if isinstance(t, bytes) else str(t))
    return labels


# ── main entry point ──────────────────────────────────────────────────────────

def visualize(
    posterior,
    tokens: Sequence,
    image: Image.Image,
    processor: AutoProcessor | None = None,
    patch_grid: tuple[int, int] | None = None,
    out_dir: str | Path | None = None,
    show: bool = True,
    animated: bool = True,
) -> dict:
    """
    Decode token labels then render all saliency plots.

    Args:
        posterior  : (1, T, H*W) or (T, H*W) — precomputed posterior tensor.
                     Produced by calculate_prior() or compute_gradcam().
        tokens     : T token labels.  list[str] is used directly.
                     list[int] / 1-D int tensor is decoded via
                     ``processor.tokenizer`` (requires processor to be given).
        image      : PIL.Image in RGB format — the original input image.
        processor  : HuggingFace processor / tokenizer.  Required only when
                     *tokens* contains integer ids.
        patch_grid : (H_p, W_p) patch layout; inferred from P when None.
        out_dir    : if given, all plots are saved to this directory.
        show       : call plt.show() after each figure.
        animated   : include the per-token animated GIF (requires out_dir).

    Returns:
        dict with keys:
            "tokens"   : list[str] — resolved display labels
            "figures"  : dict[str, plt.Figure]
            "paths"    : dict[str, Path]  (empty when out_dir is None)
    """
    image = image.convert("RGB")

    # ── decode token ids → display strings ───────────────────────────────────
    tok_labels = _resolve_token_labels(tokens, processor)

    # ── align posterior length with token list ────────────────────────────────
    post = _squeeze_batch(_to_numpy(posterior))   # (T, P)
    T_post, P = post.shape
    T_tok      = len(tok_labels)

    if T_post != T_tok:
        # BLIP prepends prompt tokens that inflate the token sequence;
        # take the last T_post labels to align with the posterior
        if T_tok > T_post:
            tok_labels = tok_labels[-T_post:]
        else:
            # posterior longer than labels — truncate posterior
            post       = post[:T_tok]
            T_post     = T_tok

    # ── infer patch grid ──────────────────────────────────────────────────────
    pg = patch_grid or _infer_grid(P)

    # ── filter blank / special-token cells ───────────────────────────────────
    valid = [i for i, t in enumerate(tok_labels) if t.strip()]
    if not valid:
        valid = list(range(T_post))
    post_f = post[valid]                          # (T_valid, P)
    tok_f  = [tok_labels[i] for i in valid]

    print(f"Posterior : {(1, T_post, P)}  →  rendering {len(tok_f)} token(s)")
    print(f"Tokens    : {tok_f}")

    # ── render ────────────────────────────────────────────────────────────────
    figs: dict[str, plt.Figure] = {}

    figs["token_maps"] = plot_token_maps(
        post_f, tok_f, image=image, patch_grid=pg,
    )
    if show:
        plt.show()

    figs["overlay"] = plot_overlay(
        post_f, tok_f, image=image, patch_grid=pg,
    )
    if show:
        plt.show()

    figs["cumulative"] = plot_cumulative(
        post_f, tok_f, image=image, patch_grid=pg,
    )
    if show:
        plt.show()

    # ── save ──────────────────────────────────────────────────────────────────
    paths: dict[str, Path] = {}
    if out_dir is not None:
        paths = save_all(
            post_f, tok_f, image=image,
            out_dir=out_dir, patch_grid=pg, animated=animated,
        )

    return {"tokens": tok_f, "figures": figs, "paths": paths}


# ── low-level: single-token functions ────────────────────────────────────────

def show_grad_mask(
    patch_scores,
    patch_grid: tuple[int, int] | None = None,
    token_label: str = "",
    cmap: str = "gray",
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Grey-scale heat-map of a single flat patch-score vector.

    Args:
        patch_scores : (P,) array / tensor of saliency values.
        patch_grid   : (H_p, W_p); inferred if None.
        token_label  : optional axes title.
        cmap         : colormap, default ``"gray"``.
        ax           : existing Axes to draw into; new figure if None.
        save_path    : write PNG if given.

    Returns:
        matplotlib Figure.
    """
    scores = _to_numpy(patch_scores).ravel()
    pg     = patch_grid or _infer_grid(scores.shape[0])
    cam    = _spatial_map(scores, pg, _image_size_wh())

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    else:
        fig = ax.get_figure()

    ax.imshow(cam, cmap=cmap, vmin=0, vmax=1)
    ax.axis("off")
    if token_label:
        ax.set_title(token_label, fontsize=9)

    if own_fig:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def show_heatmap_overlay(
    patch_scores,
    image: Image.Image,
    token_label: str = "",
    patch_grid: tuple[int, int] | None = None,
    cmap: str = "jet",
    alpha: float = 0.7,
    figsize: tuple[float, float] = (14, 3.5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    4-panel saliency figure for a single token.

    Panels: Original | Heatmap (gray) | Jet overlay | Pixel-masked image.

    Args:
        patch_scores : (P,) saliency vector.
        image        : PIL.Image (RGB).
        token_label  : figure suptitle.
        patch_grid   : (H_p, W_p); inferred if None.
        cmap         : overlay colormap (default ``"jet"``).
        alpha        : overlay blend strength (default 0.7).
        figsize      : overall figure size in inches.
        save_path    : write PNG if given.

    Returns:
        matplotlib Figure.
    """
    scores  = _to_numpy(patch_scores).ravel()
    pg      = patch_grid or _infer_grid(scores.shape[0])

    img_arr = _pil_to_hwc(image)                               # (H, W, 3)
    cam     = _spatial_map(scores, pg, _img_wh(img_arr))       # (H, W)
    masked  = np.clip(np.expand_dims(cam, 2) * img_arr, 0, 1)  # (H, W, 3)

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    axes[0].imshow(img_arr);                          axes[0].set_title("Original Image")
    axes[1].imshow(cam, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Heatmap")
    axes[2].imshow(img_arr)
    axes[2].imshow(cam, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    axes[3].imshow(masked);                           axes[3].set_title("Combined")

    for ax in axes:
        ax.axis("off")

    if token_label:
        fig.suptitle(f'Token: "{token_label}"', fontsize=11, y=1.02)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


# ── high-level: full-posterior functions ─────────────────────────────────────

def plot_token_maps(
    posterior,
    tokens: Sequence,
    image: Image.Image | None = None,
    patch_grid: tuple[int, int] | None = None,
    cmap: str = "inferno",
    figsize_per_cell: float = 2.8,
    title: str = "Per-token saliency",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Grid of saliency heat-maps, one cell per token.

    Blends over *image* when provided, otherwise renders the raw heat-map.

    Args:
        posterior        : (1, T, H*W) or (T, H*W)
        tokens           : T token labels
        image            : PIL.Image; optional background
        patch_grid       : (H_p, W_p); inferred if None
        cmap             : matplotlib colormap
        figsize_per_cell : inches per grid cell
        title            : figure suptitle
        save_path        : write PNG if given

    Returns:
        matplotlib Figure
    """
    post = _squeeze_batch(_to_numpy(posterior))   # (T, P)
    T, P = post.shape
    pg   = patch_grid or _infer_grid(P)

    img_arr = _pil_to_hwc(image) if image is not None else None
    size_wh = _img_wh(img_arr)  if img_arr is not None else None

    nrows, ncols = _grid_dims(T)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_per_cell, nrows * figsize_per_cell),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=13, y=1.01)

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        ax.axis("off")
        if idx >= T:
            continue

        cam = _spatial_map(post[idx], pg, size_wh)

        if img_arr is not None:
            ax.imshow(img_arr)
            ax.imshow(cam, cmap=cmap, alpha=0.55, vmin=0, vmax=1)
        else:
            ax.imshow(cam, cmap=cmap, vmin=0, vmax=1)

        ax.set_title(_token_label(tokens[idx]), fontsize=8, pad=2)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_overlay(
    posterior,
    tokens: Sequence,
    image: Image.Image,
    patch_grid: tuple[int, int] | None = None,
    cmap: str = "jet",
    figsize_per_cell: float = 2.8,
    title: str = "Saliency overlay",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Grid of jet overlays blended over *image*, one cell per token.
    *image* is required.
    """
    return plot_token_maps(
        posterior=posterior, tokens=tokens, image=image,
        patch_grid=patch_grid, cmap=cmap,
        figsize_per_cell=figsize_per_cell, title=title, save_path=save_path,
    )


def plot_cumulative(
    posterior,
    tokens: Sequence,
    image: Image.Image | None = None,
    patch_grid: tuple[int, int] | None = None,
    cmap: str = "inferno",
    figsize: tuple[float, float] = (5.0, 5.0),
    title: str = "Cumulative saliency (mean over tokens)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Single summary map: mean saliency across all T tokens, with a colour-bar
    and the full decoded caption as a subtitle.

    Args:
        posterior : (1, T, H*W) or (T, H*W)
        tokens    : T labels joined into a caption subtitle
        image     : PIL.Image; optional background

    Returns:
        matplotlib Figure
    """
    post      = _squeeze_batch(_to_numpy(posterior))
    mean_post = post.mean(axis=0)                        # (P,)
    pg        = patch_grid or _infer_grid(mean_post.shape[0])

    img_arr = _pil_to_hwc(image) if image is not None else None
    size_wh = _img_wh(img_arr)  if img_arr is not None else None
    cam     = _spatial_map(mean_post, pg, size_wh)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if img_arr is not None:
        ax.imshow(img_arr)
        im = ax.imshow(cam, cmap=cmap, alpha=0.55, vmin=0, vmax=1)
    else:
        im = ax.imshow(cam, cmap=cmap, vmin=0, vmax=1)

    ax.axis("off")
    caption = " ".join(_token_label(t) for t in tokens)
    ax.set_title(f'{title}\n"{caption}"', fontsize=9)

    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="4%", pad=0.05)
    fig.colorbar(im, cax=cax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_token_sequence(
    posterior,
    tokens: Sequence,
    image: Image.Image,
    patch_grid: tuple[int, int] | None = None,
    cmap: str = "jet",
    alpha: float = 0.7,
    fps: int = 2,
    figsize: tuple[float, float] = (14, 3.5),
    save_path: str | Path = "token_sequence.gif",
) -> Path:
    """
    Animated GIF — one frame per token, each frame a 4-panel figure
    (Original | Heatmap | Overlay | Combined) via show_heatmap_overlay.

    Args:
        posterior : (1, T, H*W) or (T, H*W)
        tokens    : T labels
        image     : PIL.Image (required)
        fps       : frames per second in the output GIF
        save_path : output .gif path

    Returns:
        Path to the saved GIF.
    """
    post      = _squeeze_batch(_to_numpy(posterior))   # (T, P)
    T, P      = post.shape
    pg        = patch_grid or _infer_grid(P)
    save_path = Path(save_path)
    frames: list[Image.Image] = []

    for idx in range(T):
        label = f"[{idx + 1}/{T}] {_token_label(tokens[idx])}"
        fig   = show_heatmap_overlay(
            patch_scores=post[idx], image=image,
            token_label=label, patch_grid=pg,
            cmap=cmap, alpha=alpha, figsize=figsize,
        )
        fig.canvas.draw()
        buf   = fig.canvas.buffer_rgba()
        frame = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                 buf, "raw", "RGBA", 0, 1)
        frames.append(frame.convert("RGB"))
        plt.close(fig)

    duration_ms = int(1000 / fps)
    frames[0].save(
        save_path, format="GIF",
        append_images=frames[1:],
        save_all=True, duration=duration_ms, loop=0,
    )
    return save_path


# ── convenience wrapper ───────────────────────────────────────────────────────

def save_all(
    posterior,
    tokens: Sequence,
    image: Image.Image | None = None,
    out_dir: str | Path = "./visualisations",
    patch_grid: tuple[int, int] | None = None,
    animated: bool = True,
) -> dict[str, Path]:
    """
    Write all visualisations to *out_dir* and return a path dict.

    Outputs
    ───────
    token_maps.png      — grid of per-token maps (optionally over image)
    overlay.png         — jet overlay grid           (image required)
    cumulative.png      — mean saliency summary map
    token_sequence.gif  — 4-panel animated GIF       (image + animated=True)

    Returns:
        dict with keys "token_maps", "overlay", "cumulative", "animation".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    p = out_dir / "token_maps.png"
    plot_token_maps(posterior, tokens, image=image,
                    patch_grid=patch_grid, save_path=p)
    paths["token_maps"] = p
    plt.close("all")

    if image is not None:
        p = out_dir / "overlay.png"
        plot_overlay(posterior, tokens, image,
                     patch_grid=patch_grid, save_path=p)
        paths["overlay"] = p
        plt.close("all")

    p = out_dir / "cumulative.png"
    plot_cumulative(posterior, tokens, image=image,
                    patch_grid=patch_grid, save_path=p)
    paths["cumulative"] = p
    plt.close("all")

    if animated and image is not None:
        p = out_dir / "token_sequence.gif"
        plot_token_sequence(posterior, tokens, image,
                            patch_grid=patch_grid, save_path=p)
        paths["animation"] = p
        plt.close("all")

    print(f"Saved {len(paths)} visualisation(s) to {out_dir}/")
    for k, v in paths.items():
        print(f"  {k:14s} → {v}")

    return paths


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    T = 8
    P = 576   # 24×24 patches — BLIP @ 384 px

    posterior  = torch.rand(1, T, P)
    token_ids  = [1037, 3746, 1997, 1037, 2417, 22759, 1047, 102]  # int ids
    pil_image  = Image.fromarray(
        (np.random.rand(384, 384, 3) * 255).astype(np.uint8)
    )

    # --- with string tokens (no processor needed) ---
    str_tokens = ["a", "photograph", "of", "a", "red", "floral", "kurta", "."]
    result = visualize(posterior, str_tokens, pil_image, out_dir="./vis_test")
    print("Figures:", list(result["figures"]))

    # --- with integer token ids + processor ---
    # from utils.load import load_blip
    # _, processor = load_blip()
    # result = visualize(posterior, token_ids, pil_image, processor=processor)