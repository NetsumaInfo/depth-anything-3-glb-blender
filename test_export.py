"""Headless end-to-end test: run the app pipeline on a sample image, check GLB outputs."""
import os
import numpy as np
from PIL import Image

import app

IMG = os.path.join("repo", "assets", "examples", "SOH", "000.png")


def main():
    img = np.asarray(Image.open(IMG).convert("RGB"))
    print("input:", img.shape)
    import trimesh
    for ff in (False, True):
        mesh_path, pcd_path, _, _, status = app.run(
            img, process_res=504, conf_thresh_percentile=15, edge_rel_thresh=0.15,
            num_max_points=1_000_000, full_frame=ff,
        )
        m = trimesh.load(mesh_path, force="mesh")
        msize = os.path.getsize(mesh_path) / 1024
        print(f"full_frame={ff}: mesh {msize:.0f}KB verts={len(m.vertices)} "
              f"faces={len(m.faces)} textured={m.visual.kind}")


if __name__ == "__main__":
    main()
