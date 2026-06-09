# Regen high-res textured mesh GLB for smoke (front) + ball (back).
# Higher process_res (denser geometry) + full-res texture (crisp), white->transparent alpha.
import os, sys, time
import numpy as np, torch, trimesh
from PIL import Image

ROOT = r"S:\projet_app\Depth Anything 3"
sys.path.insert(0, ROOT)
from app import (get_model, _backproject_grid_to_world,
                 _compute_alignment_transform_first_cam_glTF_center_by_points)

PROC_RES = 1008          # geometry density (was 504)
TEX_MAX  = 2048          # texture long side (was ~504)

def make_alpha(rgb, white_thr=244):
    lum = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]
    return np.where(lum < white_thr, 255, 0).astype(np.uint8)

def export_highres(pred, tex_img, out_dir, alpha_keep):
    os.makedirs(out_dir, exist_ok=True)
    depth = np.asarray(pred.depth[0], dtype=np.float64)
    K   = np.asarray(pred.intrinsics[0], dtype=np.float64)
    ext = np.asarray(pred.extrinsics[0])
    H, W = depth.shape
    finite = np.isfinite(depth) & (depth > 0)
    if finite.any():
        lo, hi = np.percentile(depth[finite], [1.0, 99.0]); depth = np.clip(depth, lo, hi)
    Xw = _backproject_grid_to_world(depth, K, ext)
    A  = _compute_alignment_transform_first_cam_glTF_center_by_points(ext, Xw[finite.reshape(-1)])
    verts = trimesh.transform_points(Xw, A).astype(np.float32)
    vert_ok = finite.reshape(-1)
    if alpha_keep is not None:
        vert_ok = vert_ok & alpha_keep.reshape(-1)
    idx = np.arange(H*W).reshape(H, W)
    tl=idx[:-1,:-1].reshape(-1); tr=idx[:-1,1:].reshape(-1); bl=idx[1:,:-1].reshape(-1); br=idx[1:,1:].reshape(-1)
    quad_ok = vert_ok[tl]&vert_ok[tr]&vert_ok[bl]&vert_ok[br]
    faces = np.concatenate([np.stack([tl[quad_ok],bl[quad_ok],tr[quad_ok]],1),
                            np.stack([tr[quad_ok],bl[quad_ok],br[quad_ok]],1)], 0)
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    uv = np.stack([us.reshape(-1)/max(W-1,1), vs.reshape(-1)/max(H-1,1)], 1).astype(np.float32)
    visual = trimesh.visual.TextureVisuals(uv=uv, image=tex_img)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)
    mesh.update_faces(mesh.nondegenerate_faces()); mesh.remove_unreferenced_vertices()
    out = os.path.join(out_dir, "mesh.glb"); mesh.export(out)
    return out, len(mesh.vertices), len(mesh.faces)

jobs = [
    ("smoke", r"S:\rush\Projet\OPM MMV\rush 2\47_0001_Calque-2.png", os.path.join(ROOT,"outputs","highres_smoke")),
    ("ball",  r"S:\rush\Projet\OPM MMV\rush 2\47_0002_font.png",     os.path.join(ROOT,"outputs","highres_ball")),
]

model = get_model()
for name, src, outd in jobs:
    t0 = time.time()
    im = Image.open(src).convert("RGB")
    rgb = np.asarray(im, np.uint8)
    alpha = make_alpha(rgb)
    pred = model.inference([rgb], process_res=PROC_RES, export_dir=None)
    Hd, Wd = pred.depth.shape[1:]
    a = np.asarray(Image.fromarray(alpha).resize((Wd, Hd), Image.NEAREST))
    alpha_keep = a > 16
    pred.conf[0][~alpha_keep] = -1.0
    tex = im.copy(); tex.thumbnail((TEX_MAX, TEX_MAX), Image.LANCZOS)
    out, nv, nf = export_highres(pred, tex, outd, alpha_keep)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"[{name}] res {Wd}x{Hd} tex {tex.size} verts {nv} faces {nf} -> {out}  ({time.time()-t0:.1f}s)")
print("DONE")
