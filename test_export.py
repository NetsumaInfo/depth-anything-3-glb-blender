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
    outputs = app.ALL_OUTPUTS
    for ff in (False, True):
        ret = app.run(
            img, process_res=504, conf_thresh_percentile=15, edge_rel_thresh=0.15,
            num_max_points=1_000_000, full_frame=ff, outputs=outputs,
        )
        (mesh_view, pcd_view, combined_view, plane_view, _depth_view,
         mesh_path, pcd_path, combined_path, plane_path,
         _d16, _dgray, _dcolor, status) = ret

        m = trimesh.load(mesh_path, force="mesh")
        msize = os.path.getsize(mesh_path) / 1024
        print(f"full_frame={ff}: mesh {msize:.0f}KB verts={len(m.vertices)} "
              f"faces={len(m.faces)} textured={m.visual.kind}")

        for name, p in [("combined", combined_path), ("plane", plane_path),
                        ("pointcloud", pcd_path)]:
            scene = trimesh.load(p)
            ngeo = len(scene.geometry) if hasattr(scene, "geometry") else 1
            print(f"  {name}: {os.path.getsize(p)/1024:.0f}KB geometries={ngeo}")


if __name__ == "__main__":
    main()
