"""Smoke test: xac nhan Ultralytics nap duoc ca 4 view va nhan OBB doc dung.

Kiem tra: so anh, so box, dai gia tri toa do (phai trong [0,1]), shape batch.
Chay TRUOC moi training run.
"""
import sys

import numpy as np
import yaml
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import IterableSimpleNamespace

HYP = IterableSimpleNamespace(**dict(
    mask_ratio=4, overlap_mask=True, bgr=0.0,
    mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, copy_paste_mode="flip",
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, auto_augment=None, erasing=0.0,
))

CASES = [
    ("dronevehicle_rgb", "configs/datasets/dronevehicle_rgb.yaml", "val"),
    ("dronevehicle_ir", "configs/datasets/dronevehicle_ir.yaml", "val"),
    ("vedai_rgb", "configs/datasets/vedai_rgb.yaml", "val"),
    ("vedai_ir", "configs/datasets/vedai_ir.yaml", "val"),
]

ok = True
for name, cfg, split in CASES:
    d = yaml.safe_load(open(cfg))
    path = f"{d['path']}/{d[split]}"
    try:
        ds = YOLODataset(img_path=path, data=d, task="obb", imgsz=640,
                         augment=False, rect=False, hyp=HYP)
    except Exception as e:
        print(f"[{name:18s}] FAIL khoi tao: {type(e).__name__}: {e}")
        ok = False
        continue

    nb = sum(len(l["bboxes"]) for l in ds.labels)
    segs = np.concatenate([l["segments"] for l in ds.labels if len(l["segments"])]) \
        if any(len(l["segments"]) for l in ds.labels) else np.zeros((0, 4, 2))
    s = ds[0]
    img = s["img"]
    rng = (float(segs.min()), float(segs.max())) if len(segs) else (float("nan"),) * 2
    bad = int(((segs < -1e-6) | (segs > 1 + 1e-6)).any()) if len(segs) else 0
    ncls = len(np.unique(np.concatenate([l["cls"] for l in ds.labels if len(l["cls"])])))
    print(f"[{name:18s}] imgs={len(ds):5d} boxes={nb:7d} classes={ncls} "
          f"coord_range=({rng[0]:.3f},{rng[1]:.3f}) out_of_range={bad} "
          f"img={tuple(img.shape)} {img.dtype}")
    if bad or nb == 0:
        ok = False

print("\nSMOKE TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
