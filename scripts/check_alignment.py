"""Kiem chung ghep cap va do sai lech dong dang ky RGB-IR tren DroneVehicle.

Ten file trong XML cua dataset KHONG dung lam khoa ghep cap duoc (nhieu quy uoc
dat ten khac nhau, co file RGB mang ten ..._R). Vi vay ta kiem chung bang ANH:

1) Ghep cap  : NMI(RGB_i, IR_i) phai cao han han NMI(RGB_i, IR_j) voi j ngau nhien.
2) Dong dang ky: uoc luong dich chuyen (dx,dy) bang phase correlation tren anh
   gradient (bat bien voi chenh lech cuong do giua hai pho).

Xuat: results/alignment/{report.json, hist_shift.png, overlay_*.jpg}
"""
import argparse, glob, json, os, random

import cv2
import numpy as np


def nmi(a, b, bins=64):
    """Normalized mutual information giua hai anh xam."""
    h, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    p = h / h.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    hxy = -(p[nz] * np.log(p[nz])).sum()
    hx = -(px[px > 0] * np.log(px[px > 0])).sum()
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    return 0.0 if hxy == 0 else float((hx + hy) / hxy - 1.0)


def grad_mag(img):
    g = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    m = (m - m.mean()) / (m.std() + 1e-6)
    return m


def estimate_shift(rgb_g, ir_g):
    """Phase correlation tren anh gradient -> (dx, dy, response)."""
    a, b = grad_mag(rgb_g), grad_mag(ir_g)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a * win, b * win)
    return float(dx), float(dy), float(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dronevehicle")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-overlay", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    ids = sorted(os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(f"{a.root}/images/{a.split}/*.jpg"))
    sample = rng.sample(ids, min(a.n, len(ids)))
    outdir = "results/alignment"
    os.makedirs(outdir, exist_ok=True)

    recs = []
    for k, sid in enumerate(sample):
        rgb = cv2.imread(f"{a.root}/images/{a.split}/{sid}.jpg", cv2.IMREAD_GRAYSCALE)
        ir = cv2.imread(f"{a.root}/imagesr/{a.split}/{sid}.jpg", cv2.IMREAD_GRAYSCALE)
        if rgb is None or ir is None:
            continue
        other = rng.choice([i for i in sample if i != sid])
        ir_other = cv2.imread(f"{a.root}/imagesr/{a.split}/{other}.jpg", cv2.IMREAD_GRAYSCALE)
        dx, dy, resp = estimate_shift(rgb, ir)
        recs.append(dict(id=sid, dx=dx, dy=dy, resp=resp,
                         nmi_matched=nmi(rgb, ir), nmi_random=nmi(rgb, ir_other),
                         lum=float(rgb.mean())))

        if k < a.n_overlay:
            # overlay: bien cua RGB (do) chong len anh IR (xam) -> soi bang mat
            e = cv2.Canny(cv2.GaussianBlur(rgb, (3, 3), 0), 60, 140)
            vis = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
            vis[e > 0] = (0, 0, 255)
            cv2.imwrite(f"{outdir}/overlay_{sid}.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

    d = {k: np.array([r[k] for r in recs]) for k in ("dx", "dy", "resp", "nmi_matched", "nmi_random")}
    mag = np.hypot(d["dx"], d["dy"])
    q = lambda x, p: float(np.percentile(x, p))
    report = {
        "split": a.split, "n_sampled": len(recs),
        "pairing": {
            "nmi_matched_mean": float(d["nmi_matched"].mean()),
            "nmi_random_mean": float(d["nmi_random"].mean()),
            "n_matched_gt_random": int((d["nmi_matched"] > d["nmi_random"]).sum()),
            "frac_matched_gt_random": float((d["nmi_matched"] > d["nmi_random"]).mean()),
        },
        "shift_px": {
            "dx_median": q(d["dx"], 50), "dy_median": q(d["dy"], 50),
            "dx_p05": q(d["dx"], 5), "dx_p95": q(d["dx"], 95),
            "dy_p05": q(d["dy"], 5), "dy_p95": q(d["dy"], 95),
            "mag_median": q(mag, 50), "mag_p75": q(mag, 75),
            "mag_p90": q(mag, 90), "mag_p95": q(mag, 95), "mag_max": float(mag.max()),
            "frac_gt_2px": float((mag > 2).mean()), "frac_gt_5px": float((mag > 5).mean()),
            "frac_gt_10px": float((mag > 10).mean()),
        },
        "phasecorr_response": {"median": q(d["resp"], 50), "p10": q(d["resp"], 10)},
    }
    json.dump({"report": report, "records": recs}, open(f"{outdir}/report.json", "w"), indent=1)

    print(json.dumps(report, indent=1))
    print(f"\n-> {a.n_overlay} anh overlay tai {outdir}/overlay_*.jpg (bien RGB mau do tren nen IR)")


if __name__ == "__main__":
    main()
