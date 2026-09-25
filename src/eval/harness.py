"""Chạy suy luận, kết xuất dự đoán và tính chỉ số phân tầng.

Thiết kế
--------
Hậu xử lý toạ độ OBB (letterbox -> ảnh gốc, regularize góc, NMS xoay) là chỗ rất
dễ sai một cách âm thầm. Vì vậy KHÔNG tự viết lại: ta thừa kế `OBBValidator` của
Ultralytics và chỉ móc thêm một bước ghi lại dự đoán đã đưa về toạ độ gốc. Nhờ đó
số liệu của harness và số liệu Ultralytics báo cáo đến từ cùng một đường code.

Model được **dựng lại từ provenance.json** rồi nạp `state_dict`, thay vì unpickle
thẳng checkpoint. Lý do: checkpoint của cấu hình two-stream tham chiếu các lớp
tuỳ chỉnh của dự án; dựng lại từ config khiến việc đánh giá không phụ thuộc vào
chi tiết pickle và tự nó là một phép kiểm tra rằng config mô tả đủ kiến trúc.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics.models.yolo.obb import OBBValidator
from ultralytics.utils import ops
from ultralytics.utils import YAML
from ultralytics.utils.ops import xywhr2xyxyxyxy

from src import patches
from src.eval import metrics as M
from src.models.build import build_arch
from src.models.fusion_ops import ModalityGate


class DumpingOBBValidator(OBBValidator):
    """OBBValidator có ghi lại dự đoán và GT theo từng ảnh (toạ độ gốc).

    Nếu truyền `gates` (các `ModalityGate` đã bật `record`), ghi thêm tỉ trọng
    của nhánh B ở từng mức hợp nhất cho từng ảnh. Đọc ngay trong `update_metrics`
    là đúng lúc: forward của batch này vừa chạy xong, `last_share` còn là của nó.
    """

    def __init__(self, *args, gates=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dump: dict[str, dict] = {}
        self.gates = list(gates or [])
        self.gate_share: dict[str, list[float]] = {}

    @staticmethod
    def scale_gt(pbatch: dict) -> dict:
        """Đưa GT từ toạ độ letterbox về toạ độ ảnh gốc — đối xứng với `scale_preds`."""
        gb = pbatch["bboxes"]
        if not len(gb):
            return pbatch
        return {**pbatch, "bboxes": ops.scale_boxes(
            pbatch["imgsz"], gb.clone(), pbatch["ori_shape"],
            ratio_pad=pbatch["ratio_pad"], xywh=True)}

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self.scale_preds(self._prepare_pred(pred), pbatch)
            stem = Path(pbatch["im_file"]).stem
            pb = predn["bboxes"].detach().cpu()
            # `_prepare_batch` chỉ đưa GT về toạ độ LETTERBOX (nhân imgsz), trong khi
            # `scale_preds` đã đưa dự đoán về toạ độ ẢNH GỐC. Phải scale GT y hệt,
            # nếu không hai bên lệch nhau đúng bằng padding của letterbox (64 px với
            # ảnh 640x512) và IoU sập về ~0.
            gb = self.scale_gt(pbatch)["bboxes"].detach().cpu()
            self.dump[stem] = {
                "pred_poly": (xywhr2xyxyxyxy(pb).reshape(-1, 8).numpy().astype(np.float32)
                              if len(pb) else np.zeros((0, 8), np.float32)),
                "conf": predn["conf"].detach().cpu().numpy().astype(np.float32),
                "pred_cls": predn["cls"].detach().cpu().numpy().astype(np.int32),
                "gt_poly": (xywhr2xyxyxyxy(gb).reshape(-1, 8).numpy().astype(np.float32)
                            if len(gb) else np.zeros((0, 8), np.float32)),
                "gt_cls": pbatch["cls"].detach().cpu().numpy().astype(np.int32).ravel(),
            }
            if self.gates:
                self.gate_share[stem] = [float(g.last_share[si]) for g in self.gates]


def rebuild_model(run_dir: str | Path, weights: str | Path | None = None):
    """Dựng lại kiến trúc từ provenance rồi nạp trọng số đã train."""
    run_dir = Path(run_dir)
    prov = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    exp = prov["config"]["experiment"]
    data = YAML.load(str(run_dir / "data.yaml"))
    nc = len(data["names"])
    ch = data["channels"]

    model, _ = build_arch(exp.get("arch_spec") or {"arch": "single"}, nc=nc, ch=ch)
    if weights is None:
        cand = run_dir / "weights" / "best.pt"
        if not cand.exists():                       # phong khi Ultralytics doi save_dir
            res = run_dir / "result.json"
            if res.exists():
                cand = Path(json.loads(res.read_text(encoding="utf-8"))["best_weights"])
        weights = cand
    w = Path(weights)
    ckpt = torch.load(w, map_location="cpu", weights_only=False)
    sd = ckpt["model"].float().state_dict() if hasattr(ckpt.get("model"), "state_dict") else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"thieu {len(missing)} tensor khi nap weight, vi du: {missing[:3]}")
    model.args = ckpt["model"].args if hasattr(ckpt.get("model"), "args") else None
    return model, exp, data


def run_inference(run_dir: str | Path, split: str = "test", imgsz: int = 640,
                  batch: int = 8, device: str = "0", out_name: str = "preds_test.npz",
                  data_override: str | Path | None = None) -> Path:
    """Chạy suy luận trên một split và lưu dự đoán ra .npz."""
    patches.apply_all()
    run_dir = Path(run_dir)
    model, exp, data = rebuild_model(run_dir)
    gates = [m for m in model.modules() if isinstance(m, ModalityGate)]
    for g in gates:
        g.record = True

    data_yaml = Path(data_override) if data_override else run_dir / "data.yaml"
    args = dict(model=None, data=str(data_yaml), split=split, imgsz=imgsz, batch=batch,
                device=device, task="obb", mode="val", conf=0.001, iou=0.7, max_det=300,
                plots=False, save_json=False, verbose=False, rect=False)
    v = DumpingOBBValidator(args=args, gates=gates)
    v(model=model.eval())

    out = run_dir / out_name
    save_dump(v.dump, out)
    if v.gate_share:
        save_gate_share(v.gate_share, run_dir / out_name.replace("preds_", "gates_")
                        .replace(".npz", ".csv"))
    return out


def gate_levels(n: int) -> list[str]:
    """Tên cột cho từng mức hợp nhất. `fused_levels` được sắp tăng dần, nên với
    YOLO11 (3 mức) thứ tự là P3, P4, P5."""
    return [f"P{3 + i}" for i in range(n)] if n == 3 else [f"g{i}" for i in range(n)]


def save_gate_share(share: dict[str, list[float]], path: str | Path) -> None:
    """Ghi tỉ trọng nhánh B (IR ở F2c) theo ảnh: id, P3, P4, P5 — giá trị trong [0, 1]."""
    ids = sorted(share)
    levels = gate_levels(len(share[ids[0]]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", *levels])
        for i in ids:
            w.writerow([i, *(f"{x:.6f}" for x in share[i])])


def summarize_gate_share(path: str | Path, strata: dict) -> dict:
    """Trung bình tỉ trọng nhánh B theo từng nhóm phân tầng (vd decile chiếu sáng).

    Returns:
        {dim: {group: {"n_images": int, "P3": mean, "P4": mean, "P5": mean}}}
    """
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    levels = [k for k in rows[0] if k != "id"] if rows else []
    by_id = {r["id"]: [float(r[k]) for k in levels] for r in rows}
    out: dict[str, dict] = {}
    for dim, groups in strata.items():
        out[dim] = {}
        for gname, ids in groups.items():
            vals = np.array([by_id[i] for i in ids if i in by_id], dtype=np.float64)
            if not len(vals):
                continue
            out[dim][gname] = {"n_images": len(vals),
                               **{k: float(v) for k, v in zip(levels, vals.mean(0))}}
    return out


def save_dump(dump: dict, path: str | Path) -> None:
    """Lưu dump nhiều ảnh vào một .npz phẳng (nén, tra cứu nhanh)."""
    ids = sorted(dump)
    pack = {"ids": np.array(ids, dtype=object)}
    for key in ("pred_poly", "conf", "pred_cls", "gt_poly", "gt_cls"):
        arrs = [np.asarray(dump[i][key]) for i in ids]
        lens = np.array([len(a) for a in arrs], dtype=np.int64)
        flat = np.concatenate(arrs) if len(arrs) and lens.sum() else np.zeros(
            (0,) + (arrs[0].shape[1:] if arrs and arrs[0].ndim > 1 else ()), dtype=np.float32)
        pack[f"{key}__data"] = flat
        pack[f"{key}__len"] = lens
    np.savez_compressed(str(path), **pack)


def load_dump(path: str | Path) -> dict:
    """Đọc lại dump đã lưu.

    `NpzFile` KHÔNG cache: mỗi lần `z[key]` là một lần giải nén lại toàn bộ mảng.
    Vì vậy phải nạp mỗi mảng đúng một lần rồi mới cắt lát trong vòng lặp — nếu
    truy cập `z[...]` bên trong vòng lặp thì với 8980 ảnh sẽ là 44.900 lần giải
    nén mảng 40 MiB, đủ để phân mảnh vùng nhớ và ném MemoryError.
    """
    keys = ("pred_poly", "conf", "pred_cls", "gt_poly", "gt_cls")
    with np.load(str(path), allow_pickle=True) as z:
        ids = [str(i) for i in z["ids"]]
        data = {k: z[f"{k}__data"] for k in keys}
        offs = {k: np.concatenate([[0], np.cumsum(z[f"{k}__len"])]) for k in keys}
    return {i: {k: data[k][offs[k][n]:offs[k][n + 1]] for k in keys}
            for n, i in enumerate(ids)}


def match_all(dump: dict, iou_thrs=M.IOU_THRS) -> dict:
    """Khớp dự đoán với GT cho từng ảnh -> đầu vào của metrics/bootstrap."""
    per_image = {}
    for sid, r in dump.items():
        order = np.argsort(-r["conf"]) if len(r["conf"]) else np.zeros(0, dtype=int)
        pp, pc, cf = r["pred_poly"][order], r["pred_cls"][order], r["conf"][order]
        tp = M.match_image(pp, pc, r["gt_poly"], r["gt_cls"], iou_thrs)
        per_image[sid] = {"tp": tp, "conf": cf, "pred_cls": pc, "gt_cls": r["gt_cls"],
                          "pred_poly": pp, "gt_poly": r["gt_poly"]}
    return per_image


def evaluate_strata(per_image: dict, strata: dict, nc: int, iou_thrs=M.IOU_THRS) -> dict:
    """Tính mAP cho từng nhóm trong từng chiều phân tầng."""
    from src.eval.bootstrap import BootstrapData

    out: dict[str, dict] = {}
    for dim, groups in strata.items():
        out[dim] = {}
        for gname, ids in groups.items():
            ids = [i for i in ids if i in per_image]
            if not ids:
                continue
            tp, conf, pcls, gcls = [], [], [], []
            for i in ids:
                r = per_image[i]
                if len(r["conf"]):
                    tp.append(r["tp"])
                    conf.append(r["conf"])
                    pcls.append(r["pred_cls"])
                if len(r["gt_cls"]):
                    gcls.append(r["gt_cls"])
            n_thr = len(iou_thrs)
            res = M.evaluate(
                np.concatenate(tp) if tp else np.zeros((0, n_thr), bool),
                np.concatenate(conf) if conf else np.zeros(0),
                np.concatenate(pcls) if pcls else np.zeros(0, int),
                np.concatenate(gcls) if gcls else np.zeros(0, int),
                nc=nc, iou_thrs=iou_thrs,
            )
            res["n_images"] = len(ids)
            res["n_gt"] = int(sum(len(per_image[i]["gt_cls"]) for i in ids))
            out[dim][gname] = res
    return out
