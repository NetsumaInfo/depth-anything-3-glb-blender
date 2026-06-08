# Copyright (c) 2025
# Slim Gradio interface for Depth Anything 3: image -> GLB (mesh + point cloud) for Blender.

from __future__ import annotations

import os
import time
import traceback

import imageio.v2 as imageio
import numpy as np
import torch
import trimesh
from PIL import Image

# Reduce CUDA fragmentation on small (8 GB) GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr  # noqa: E402

from depth_anything_3.api import DepthAnything3  # noqa: E402
from depth_anything_3.utils.export.glb import (  # noqa: E402
    export_to_glb,
    _as_homogeneous44,
    _compute_alignment_transform_first_cam_glTF_center_by_points,
)
from depth_anything_3.utils.visualize import visualize_depth  # noqa: E402

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
MODEL_ID = os.environ.get("DA3_MODEL", "depth-anything/DA3-LARGE-1.1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

_MODEL: DepthAnything3 | None = None


def get_model() -> DepthAnything3:
    """Lazy-load the model once and keep it resident on the GPU."""
    global _MODEL
    if _MODEL is None:
        print(f"[da3] loading {MODEL_ID} on {DEVICE} ...")
        t0 = time.time()
        model = DepthAnything3.from_pretrained(MODEL_ID)
        model = model.to(DEVICE)
        model.eval()
        model.device = torch.device(DEVICE)
        _MODEL = model
        print(f"[da3] model ready in {time.time() - t0:.1f}s")
    return _MODEL


# --------------------------------------------------------------------------------------
# Mesh builder (textured triangle surface from a single depth map)
# --------------------------------------------------------------------------------------
def _backproject_grid_to_world(depth, K, ext_w2c):
    """Backproject every pixel to world coords. Returns (H*W, 3) float array (row-major)."""
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3).astype(np.float64)

    K_inv = np.linalg.inv(K)
    c2w = np.linalg.inv(_as_homogeneous44(ext_w2c).astype(np.float64))

    rays = (K_inv @ pix.T)  # (3, H*W)
    Xc = rays * depth.reshape(-1)[None, :]  # (3, H*W)
    Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
    Xw = (c2w @ Xc_h)[:3].T  # (H*W, 3)
    return Xw


def export_textured_mesh_glb(
    prediction,
    out_dir: str,
    conf_thresh_percentile: float = 40.0,
    edge_rel_thresh: float = 0.05,
    full_frame: bool = False,
    alpha_keep: np.ndarray | None = None,
) -> str:
    """Build one textured triangle mesh from the first view and export it as mesh.glb.

    Faces are culled when a vertex is low-confidence or when the relative depth jump
    across the quad exceeds ``edge_rel_thresh`` (kills stretched background webbing).
    With ``full_frame`` the whole image is kept as a continuous relief surface (no
    cropping); depth is clamped to its 1-99 percentile range to tame spikes.
    The scene is re-oriented to the glTF frame (Y up) so it imports upright in Blender.
    """
    os.makedirs(out_dir, exist_ok=True)

    depth = np.asarray(prediction.depth[0], dtype=np.float64)        # (H, W)
    conf = np.asarray(prediction.conf[0], dtype=np.float64)          # (H, W)
    K = np.asarray(prediction.intrinsics[0], dtype=np.float64)       # (3, 3)
    ext = np.asarray(prediction.extrinsics[0])                       # (4,4) or (3,4)
    img = np.asarray(prediction.processed_images[0], dtype=np.uint8)  # (H, W, 3)
    H, W = depth.shape

    # World points for every pixel, then glTF-align using the valid subset.
    finite = np.isfinite(depth) & (depth > 0)
    if full_frame:
        # Keep the entire frame: no confidence/discontinuity culling, just clamp
        # depth so a few wrong far/near pixels don't blow up the bounding box.
        conf_thresh_percentile = 0.0
        edge_rel_thresh = 1e9
        if finite.any():
            lo, hi = np.percentile(depth[finite], [1.0, 99.0])
            depth = np.clip(depth, lo, hi)
    Xw = _backproject_grid_to_world(depth, K, ext)                   # (H*W, 3)
    valid_world = Xw[finite.reshape(-1)]
    A = _compute_alignment_transform_first_cam_glTF_center_by_points(ext, valid_world)
    verts = trimesh.transform_points(Xw, A).astype(np.float32)       # (H*W, 3)

    # Per-vertex validity: finite depth + confidence above adaptive percentile.
    conf_thr = np.percentile(conf[finite], conf_thresh_percentile) if finite.any() else 0.0
    vert_ok = (finite & (conf >= conf_thr)).reshape(-1)

    # Alpha mask cuts real holes (applies even in full-frame: it's explicit intent).
    if alpha_keep is not None:
        vert_ok = vert_ok & alpha_keep.reshape(-1)

    # Build two triangles per quad; cull on validity + depth discontinuity.
    idx = np.arange(H * W).reshape(H, W)
    tl = idx[:-1, :-1].reshape(-1)
    tr = idx[:-1, 1:].reshape(-1)
    bl = idx[1:, :-1].reshape(-1)
    br = idx[1:, 1:].reshape(-1)

    d = depth.reshape(-1)
    quad = np.stack([d[tl], d[tr], d[bl], d[br]], axis=1)
    dmax = quad.max(axis=1)
    dmin = quad.min(axis=1)
    smooth = (dmax - dmin) <= (edge_rel_thresh * np.maximum(dmax, 1e-6))
    quad_ok = vert_ok[tl] & vert_ok[tr] & vert_ok[bl] & vert_ok[br] & smooth

    faces = np.concatenate(
        [
            np.stack([tl[quad_ok], bl[quad_ok], tr[quad_ok]], axis=1),
            np.stack([tr[quad_ok], bl[quad_ok], br[quad_ok]], axis=1),
        ],
        axis=0,
    )

    # UVs map vertices onto the source image. glTF samples with V origin at the top
    # and PIL row 0 is the top, so no V-flip (verified by round-trip test).
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    uv = np.stack([us.reshape(-1) / max(W - 1, 1),
                   vs.reshape(-1) / max(H - 1, 1)], axis=1).astype(np.float32)

    visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(img))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)

    # Drop vertices unused by any surviving face to slim the file.
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    out_path = os.path.join(out_dir, "mesh.glb")
    mesh.export(out_path)
    return out_path


def export_depth_maps(prediction, out_dir: str, alpha_keep=None):
    """Write depth maps: 16-bit grayscale (Blender height map), 8-bit B&W, colour preview.

    Depth normalised to its 1-99 percentile range, brighter = farther.
    Transparent pixels (alpha) are set to black.
    Returns (path_16bit, path_gray, path_color).
    """
    os.makedirs(out_dir, exist_ok=True)
    d = np.asarray(prediction.depth[0], dtype=np.float64)
    valid = np.isfinite(d) & (d > 0)
    if alpha_keep is not None:
        valid &= alpha_keep
    if valid.any():
        lo, hi = np.percentile(d[valid], [1.0, 99.0])
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if alpha_keep is not None:
        norm[~alpha_keep] = 0.0

    path16 = os.path.join(out_dir, "depth_16bit.png")
    imageio.imwrite(path16, (norm * 65535.0).astype(np.uint16))

    path_gray = os.path.join(out_dir, "depth_gray.png")
    imageio.imwrite(path_gray, (norm * 255.0).astype(np.uint8))

    color = visualize_depth(prediction.depth[0]).astype(np.uint8)  # (H,W,3)
    if alpha_keep is not None:
        color[~alpha_keep] = 0
    path_color = os.path.join(out_dir, "depth_color.png")
    imageio.imwrite(path_color, color)
    return path16, path_gray, path_color


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
def run(image, process_res, conf_thresh_percentile, edge_rel_thresh, num_max_points, full_frame):
    if image is None:
        raise gr.Error("Charge une image d'abord.")
    try:
        model = get_model()

        run_dir = os.path.join(OUTPUT_ROOT, time.strftime("%Y%m%d-%H%M%S"))
        pcd_dir = os.path.join(run_dir, "pointcloud")
        os.makedirs(pcd_dir, exist_ok=True)

        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        img = arr[..., :3].astype(np.uint8)
        alpha = arr[..., 3] if arr.shape[-1] == 4 else None

        t0 = time.time()
        prediction = model.inference(
            [img],
            process_res=int(process_res),
            export_dir=None,            # no built-in export; we drive both ourselves
        )
        infer_s = time.time() - t0

        # Alpha channel -> keep mask at depth resolution. Transparent pixels become real
        # holes: removed from the mesh, and culled from the cloud via negative confidence.
        alpha_keep = None
        if alpha is not None and (alpha < 250).any():
            Hd, Wd = prediction.depth.shape[1:]
            a = np.asarray(Image.fromarray(alpha).resize((Wd, Hd), Image.NEAREST))
            alpha_keep = a > 16
            prediction.conf[0][~alpha_keep] = -1.0

        # In full-frame mode keep every point in the cloud too (no percentile cull).
        pcd_pct = 0.0 if full_frame else float(conf_thresh_percentile)
        pcd_base = 0.0 if full_frame else 1.05

        # 1) Point cloud GLB (reuse repo exporter; no cameras, no depth_vis -> avoids cp bug)
        export_to_glb(
            prediction,
            pcd_dir,
            num_max_points=int(num_max_points),
            conf_thresh=pcd_base,
            conf_thresh_percentile=pcd_pct,
            show_cameras=False,
            export_depth_vis=False,
        )
        pcd_path = os.path.join(pcd_dir, "scene.glb")

        # 2) Textured mesh GLB (custom)
        mesh_path = export_textured_mesh_glb(
            prediction,
            run_dir,
            conf_thresh_percentile=float(conf_thresh_percentile),
            edge_rel_thresh=float(edge_rel_thresh),
            full_frame=bool(full_frame),
            alpha_keep=alpha_keep,
        )

        # 3) Depth maps (16-bit height map + 8-bit B&W + colour preview)
        depth16_path, depthgray_path, depthcolor_path = export_depth_maps(
            prediction, run_dir, alpha_keep
        )

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        status = (
            f"OK — inference {infer_s:.1f}s · res {prediction.depth.shape[2]}x"
            f"{prediction.depth.shape[1]}\nMesh: {mesh_path}\nPoints: {pcd_path}"
            f"\nDepth 16-bit: {depth16_path}"
        )
        return (mesh_path, pcd_path, depthgray_path,
                mesh_path, pcd_path, depth16_path, depthgray_path, depthcolor_path, status)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise gr.Error("VRAM saturée. Baisse 'Résolution' (ex 392) et réessaie.")
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Échec: {e}")


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="Depth Anything 3 → GLB (Blender)") as demo:
        gr.Markdown(
            "## Depth Anything 3 → GLB pour Blender\n"
            f"Modèle: `{MODEL_ID}` · device: `{DEVICE}`\n\n"
            "Image en entrée → mesh texturé **.glb** + nuage de points **.glb**."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(
                    label="Image d'entrée (alpha = trou)", type="numpy",
                    image_mode="RGBA", height=360,
                )
                process_res = gr.Slider(
                    256, 1008, value=504, step=28, label="Résolution (baisse si VRAM saturée)"
                )
                full_frame = gr.Checkbox(
                    value=True,
                    label="Cadre complet (relief, aucune découpe) — décoche pour découper le fond",
                )
                conf = gr.Slider(
                    0, 90, value=15, step=5,
                    label="Seuil confiance (percentile) — ignoré si Cadre complet",
                )
                edge = gr.Slider(
                    0.01, 0.30, value=0.15, step=0.01,
                    label="Sensibilité bords mesh — bas = coupe plus aux ruptures (ignoré si Cadre complet)",
                )
                npts = gr.Slider(
                    50_000, 2_000_000, value=1_000_000, step=50_000,
                    label="Max points (nuage)",
                )
                btn = gr.Button("Générer GLB", variant="primary")
            with gr.Column(scale=1):
                mesh_view = gr.Model3D(label="Mesh texturé (.glb)")
                pcd_view = gr.Model3D(label="Nuage de points (.glb)")
                depth_view = gr.Image(label="Depth map (preview N&B)", height=240)
                mesh_file = gr.File(label="Télécharger mesh.glb")
                pcd_file = gr.File(label="Télécharger pointcloud .glb")
                depth16_file = gr.File(label="Télécharger depth 16-bit (height map Blender)")
                depthgray_file = gr.File(label="Télécharger depth N&B 8-bit")
                depthcolor_file = gr.File(label="Télécharger depth colorisée")
                status = gr.Textbox(label="Statut", lines=4)

        btn.click(
            run,
            inputs=[image, process_res, conf, edge, npts, full_frame],
            outputs=[mesh_view, pcd_view, depth_view,
                     mesh_file, pcd_file, depth16_file, depthgray_file, depthcolor_file, status],
        )
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
