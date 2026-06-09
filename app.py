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
MODEL_ID = os.environ.get("DA3_MODEL", "depth-anything/DA3NESTED-GIANT-LARGE-1.1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Output choices the user can toggle in the UI.
OUT_MESH = "Mesh texturé (.glb)"
OUT_PCD = "Nuage de points (.glb)"
OUT_COMBINED = "Combiné mesh + nuage (.glb)"
OUT_PLANE = "Plan image + relief (.glb)"
OUT_DEPTH = "Depth maps (16-bit / N&B / couleur)"
ALL_OUTPUTS = [OUT_MESH, OUT_PCD, OUT_COMBINED, OUT_PLANE, OUT_DEPTH]
DEFAULT_OUTPUTS = [OUT_MESH, OUT_COMBINED, OUT_PLANE, OUT_DEPTH]

_MODEL: DepthAnything3 | None = None


def get_model() -> DepthAnything3:
    """Lazy-load the model once and keep it resident on the GPU.

    Weights stay fp32: the depth head runs with autocast disabled and is built for
    fp32 (sincos pos-embed, exp activation), so casting the net to bf16 corrupts the
    dtype boundary. On an 8 GB card the big NESTED/GIANT checkpoints overflow VRAM and
    spill into shared system RAM — slower, but correct and full quality. Drop the
    resolution or switch DA3_MODEL to a lighter checkpoint if you need more speed.
    """
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


def build_textured_mesh(
    prediction,
    conf_thresh_percentile: float = 40.0,
    edge_rel_thresh: float = 0.05,
    full_frame: bool = False,
    alpha_keep: np.ndarray | None = None,
):
    """Build one textured triangle mesh from the first view.

    Returns ``(mesh, extras)``. ``extras`` carries the full-grid aligned vertices,
    per-pixel colours, the keep mask, and the texture/size — everything the combined
    and plane+relief exporters need so they share the SAME alignment as the mesh and
    line up in world space.

    Faces are culled when a vertex is low-confidence or when the relative depth jump
    across the quad exceeds ``edge_rel_thresh`` (kills stretched background webbing).
    With ``full_frame`` the whole image is kept as a continuous relief surface (no
    cropping); depth is clamped to its 1-99 percentile range to tame spikes.
    The scene is re-oriented to the glTF frame (Y up) so it imports upright in Blender.
    """
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

    tex = Image.fromarray(img)
    visual = trimesh.visual.TextureVisuals(uv=uv, image=tex)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)

    # Drop vertices unused by any surviving face to slim the file.
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    extras = {
        "verts": verts,                                  # (H*W, 3) aligned, full grid
        "colors": img.reshape(-1, 3).astype(np.uint8),   # (H*W, 3)
        "cloud_mask": vert_ok,                           # (H*W,) bool
        "finite": finite.reshape(-1),                    # (H*W,) bool
        "tex": tex,
        "W": W,
        "H": H,
    }
    return mesh, extras


def export_textured_mesh_glb(
    prediction,
    out_dir: str,
    conf_thresh_percentile: float = 40.0,
    edge_rel_thresh: float = 0.05,
    full_frame: bool = False,
    alpha_keep: np.ndarray | None = None,
) -> str:
    """Build the textured relief mesh and write it as mesh.glb."""
    os.makedirs(out_dir, exist_ok=True)
    mesh, _ = build_textured_mesh(
        prediction,
        conf_thresh_percentile=conf_thresh_percentile,
        edge_rel_thresh=edge_rel_thresh,
        full_frame=full_frame,
        alpha_keep=alpha_keep,
    )
    out_path = os.path.join(out_dir, "mesh.glb")
    mesh.export(out_path)
    return out_path


def export_combined_glb(mesh, extras, out_dir: str, num_max_points: int) -> str:
    """One GLB scene with the textured mesh AND the point cloud (same world frame).

    Import once in Blender → relief surface + dense coloured points, both aligned.
    """
    os.makedirs(out_dir, exist_ok=True)
    scene = trimesh.Scene()
    scene.add_geometry(mesh.copy(), geom_name="mesh")

    pv = extras["verts"][extras["cloud_mask"]]
    pc = extras["colors"][extras["cloud_mask"]]
    if pv.shape[0] > num_max_points:
        sel = np.random.choice(pv.shape[0], int(num_max_points), replace=False)
        pv, pc = pv[sel], pc[sel]
    if pv.shape[0] > 0:
        scene.add_geometry(trimesh.points.PointCloud(vertices=pv, colors=pc), geom_name="points")

    out_path = os.path.join(out_dir, "combined.glb")
    scene.export(out_path)
    return out_path


def _build_image_plane(extras) -> trimesh.Trimesh:
    """A flat textured quad sitting just behind the relief, same image texture/UVs.

    Corners are taken from the four corner patches of the aligned grid so the plane
    spans exactly the relief's XY extent and shares its orientation.
    """
    verts = extras["verts"]
    finite = extras["finite"]
    H, W = extras["H"], extras["W"]
    tex = extras["tex"]

    g = verts.reshape(H, W, 3)
    m = finite.reshape(H, W)
    vv = verts[finite] if finite.any() else verts
    zmin = float(vv[:, 2].min())
    zspan = float(vv[:, 2].max() - zmin) or 1.0
    zplane = zmin - 0.02 * zspan  # nudge behind the relief, avoid z-fighting

    fh = max(H // 10, 1)
    fw = max(W // 10, 1)

    def patch(r0, r1, c0, c1):
        s = g[r0:r1, c0:c1].reshape(-1, 3)
        sm = m[r0:r1, c0:c1].reshape(-1)
        s = s[sm]
        return s.mean(axis=0) if len(s) else None

    tl = patch(0, fh, 0, fw)
    tr = patch(0, fh, W - fw, W)
    bl = patch(H - fh, H, 0, fw)
    br = patch(H - fh, H, W - fw, W)

    if any(p is None for p in (tl, tr, bl, br)):
        xmin, ymin = vv[:, 0].min(), vv[:, 1].min()
        xmax, ymax = vv[:, 0].max(), vv[:, 1].max()
        tl = np.array([xmin, ymax, 0.0])
        tr = np.array([xmax, ymax, 0.0])
        bl = np.array([xmin, ymin, 0.0])
        br = np.array([xmax, ymin, 0.0])

    corners = np.array([tl, tr, bl, br], dtype=np.float32)  # TL, TR, BL, BR
    corners[:, 2] = zplane
    faces = np.array([[0, 2, 1], [1, 2, 3]], dtype=np.int64)
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    visual = trimesh.visual.TextureVisuals(uv=uv, image=tex)
    return trimesh.Trimesh(vertices=corners, faces=faces, visual=visual, process=False)


def export_plane_relief_glb(mesh, extras, out_dir: str) -> str:
    """One GLB scene with a flat image plane behind the textured relief mesh."""
    os.makedirs(out_dir, exist_ok=True)
    scene = trimesh.Scene()
    scene.add_geometry(mesh.copy(), geom_name="relief")
    scene.add_geometry(_build_image_plane(extras), geom_name="plane")
    out_path = os.path.join(out_dir, "plane_relief.glb")
    scene.export(out_path)
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
def run(image, process_res, conf_thresh_percentile, edge_rel_thresh, num_max_points,
        full_frame, outputs):
    if image is None:
        raise gr.Error("Charge une image d'abord.")
    outputs = outputs or []
    if not outputs:
        raise gr.Error("Coche au moins une sortie.")
    try:
        model = get_model()

        run_dir = os.path.join(OUTPUT_ROOT, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)

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

        mesh_path = pcd_path = combined_path = plane_path = None
        depth16_path = depthgray_path = depthcolor_path = None

        # Shared mesh geometry: built once, reused by mesh / combined / plane outputs.
        mesh = extras = None
        need_mesh = any(o in outputs for o in (OUT_MESH, OUT_COMBINED, OUT_PLANE))
        if need_mesh:
            mesh, extras = build_textured_mesh(
                prediction,
                conf_thresh_percentile=float(conf_thresh_percentile),
                edge_rel_thresh=float(edge_rel_thresh),
                full_frame=bool(full_frame),
                alpha_keep=alpha_keep,
            )

        if OUT_MESH in outputs:
            mesh_path = os.path.join(run_dir, "mesh.glb")
            mesh.export(mesh_path)

        if OUT_COMBINED in outputs:
            combined_path = export_combined_glb(mesh, extras, run_dir, int(num_max_points))

        if OUT_PLANE in outputs:
            plane_path = export_plane_relief_glb(mesh, extras, run_dir)

        if OUT_PCD in outputs:
            # Standalone point cloud (repo exporter; no cameras / depth_vis -> avoids cp bug).
            pcd_dir = os.path.join(run_dir, "pointcloud")
            os.makedirs(pcd_dir, exist_ok=True)
            pcd_pct = 0.0 if full_frame else float(conf_thresh_percentile)
            pcd_base = 0.0 if full_frame else 1.05
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

        if OUT_DEPTH in outputs:
            depth16_path, depthgray_path, depthcolor_path = export_depth_maps(
                prediction, run_dir, alpha_keep
            )

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        made = []
        for label, p in [
            ("Mesh", mesh_path), ("Combiné", combined_path), ("Plan+relief", plane_path),
            ("Nuage", pcd_path), ("Depth 16-bit", depth16_path),
        ]:
            if p:
                made.append(f"{label}: {p}")
        status = (
            f"OK — inference {infer_s:.1f}s · res {prediction.depth.shape[2]}x"
            f"{prediction.depth.shape[1]}\n" + "\n".join(made)
        )
        return (
            mesh_path, pcd_path, combined_path, plane_path, depthgray_path,
            mesh_path, pcd_path, combined_path, plane_path,
            depth16_path, depthgray_path, depthcolor_path, status,
        )
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
            "Image en entrée → choisis tes sorties: mesh, nuage, **combiné** (mesh+nuage), "
            "**plan+relief**, depth maps."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(
                    label="Image d'entrée (alpha = trou)", type="numpy",
                    image_mode="RGBA", height=360,
                )
                out_select = gr.CheckboxGroup(
                    choices=ALL_OUTPUTS, value=DEFAULT_OUTPUTS,
                    label="Sorties à générer",
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
                    label="Max points (nuage / combiné)",
                )
                btn = gr.Button("Générer GLB", variant="primary")
            with gr.Column(scale=1):
                mesh_view = gr.Model3D(label="Mesh texturé (.glb)")
                pcd_view = gr.Model3D(label="Nuage de points (.glb)")
                combined_view = gr.Model3D(label="Combiné mesh + nuage (.glb)")
                plane_view = gr.Model3D(label="Plan image + relief (.glb)")
                depth_view = gr.Image(label="Depth map (preview N&B)", height=240)
                mesh_file = gr.File(label="Télécharger mesh.glb")
                pcd_file = gr.File(label="Télécharger pointcloud .glb")
                combined_file = gr.File(label="Télécharger combiné .glb")
                plane_file = gr.File(label="Télécharger plan+relief .glb")
                depth16_file = gr.File(label="Télécharger depth 16-bit (height map Blender)")
                depthgray_file = gr.File(label="Télécharger depth N&B 8-bit")
                depthcolor_file = gr.File(label="Télécharger depth colorisée")
                status = gr.Textbox(label="Statut", lines=5)

        btn.click(
            run,
            inputs=[image, process_res, conf, edge, npts, full_frame, out_select],
            outputs=[mesh_view, pcd_view, combined_view, plane_view, depth_view,
                     mesh_file, pcd_file, combined_file, plane_file,
                     depth16_file, depthgray_file, depthcolor_file, status],
        )
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
