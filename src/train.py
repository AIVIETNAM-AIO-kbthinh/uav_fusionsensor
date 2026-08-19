"""Trainer cho toàn bộ ma trận thí nghiệm.

Mọi cấu hình (S1, S2, C1, C2, F1, F2a, F2b) đi qua ĐÚNG một đường code này; khác
nhau chỉ ở file config. Đó là điều kiện cần để so sánh có ý nghĩa.

Ghi chú về chuẩn hoá đầu vào (lệch có chủ đích so với plan mục 6.4)
------------------------------------------------------------------
Plan yêu cầu "normalization riêng cho từng modality (mean/std tính riêng)".
Mặc định ở đây là `normalize: none`, tức chỉ chia 255 như Ultralytics. Lý do:

1. Mối lo thật sự của plan là **dùng CHUNG thống kê giữa hai modality** ("dùng
   chung là hỏng ngay"). Với phép chia 255, không có thống kê nào được chia sẻ:
   mỗi kênh chịu đúng cùng một phép co tuyến tính, và BatchNorm ở lớp conv đầu
   tiên tự học thống kê riêng cho từng kênh từ dữ liệu. Mối lo đã được xử lý.
2. Ta nạp trọng số pretrained (xem src/models/build.py) — pretrained YOLO kỳ vọng
   đầu vào trong [0,1]. Chuẩn hoá về zero-mean/unit-var sẽ làm lệch phân phối mà
   stem pretrained mong đợi, gây hại nhiều hơn lợi.

Tuỳ chọn `normalize: per_modality` vẫn được cài đặt đầy đủ để kiểm chứng lại
quyết định này bằng thực nghiệm nếu cần. Dù chọn cách nào, nó áp dụng như nhau
cho MỌI cấu hình nên tính công bằng không đổi.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.utils import YAML

from src import patches
from src.data.paired_dataset import modality_channels
from src.models.build import build_arch, load_pretrained
from src.utils.seed import dump_provenance, set_seed


class FusionOBBTrainer(OBBTrainer):
    """OBBTrainer biết dựng kiến trúc two-stream và nạp pretrained cho nó."""

    arch_spec: dict | None = None
    pretrained_weights: str | None = None
    normalize: str = "none"
    modality_stats: dict | None = None
    pretrained_stats: dict | None = None

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model, built_cfg = build_arch(
            self.arch_spec or {"arch": "single"},
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose,
        )
        if self.pretrained_weights:
            self.pretrained_stats = load_pretrained(model, self.pretrained_weights, built_cfg,
                                                    verbose=verbose)
        return model

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        if self.normalize == "per_modality" and self.modality_stats:
            img = batch["img"]
            mean = torch.tensor(self.modality_stats["mean"], device=img.device).view(1, -1, 1, 1)
            std = torch.tensor(self.modality_stats["std"], device=img.device).view(1, -1, 1, 1)
            batch["img"] = (img - mean) / std
        return batch


# --------------------------------------------------------------------------- #

def make_data_yaml(dataset_yaml: str | Path, modalities: list[str], out_path: str | Path,
                   ir_shift: tuple[int, int] | None = None) -> Path:
    """Sinh data yaml dẫn xuất có `channels` + `modalities`.

    Ghi ra đĩa thay vì nhét vào bộ nhớ để mỗi run có một artefact tái lập được,
    và để trainer lẫn validator chắc chắn dùng cùng một cấu hình modality.
    """
    d = YAML.load(str(dataset_yaml))
    d["channels"] = modality_channels(modalities)
    d["modalities"] = list(modalities)
    if ir_shift and any(ir_shift):
        d["ir_shift"] = list(ir_shift)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(str(out_path), d)
    return out_path


def run_experiment(exp: dict, dataset_yaml: str | Path, seed: int, project: str = "runs",
                   name: str | None = None, overrides: dict | None = None) -> dict:
    """Chạy một ô của ma trận thí nghiệm.

    Args:
        exp: config thí nghiệm đã nạp (configs/experiments/*.yaml).
        dataset_yaml: configs/datasets/*.yaml.
        seed: seed cho run này.
        project/name: nơi ghi kết quả.

    Returns:
        dict tóm tắt (metrics + đường dẫn weight).
    """
    patches.apply_all()
    set_seed(seed, deterministic=True)

    name = name or f"{exp['id']}_seed{seed}"
    run_dir = Path(project) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = make_data_yaml(dataset_yaml, exp["modalities"], run_dir / "data.yaml",
                               exp.get("ir_shift"))

    base = YAML.load("configs/base.yaml")
    # `args` chỉ được chứa khoá hợp lệ của Ultralytics — mọi thứ riêng của dự án
    # nằm trong `exp` và gán thẳng lên trainer.
    args = {**base, **(exp.get("train") or {}), **(overrides or {})}
    # ep Ultralytics ghi dung vao run_dir: neu khong, no tu chen them runs/<task>/
    args.update(data=str(data_yaml), seed=seed, project=project, name=name, exist_ok=True,
                save_dir=str(run_dir))

    cfg_record = {"experiment": exp, "dataset_yaml": str(dataset_yaml), "seed": seed, "args": args}
    dump_provenance(run_dir / "provenance.json", cfg_record)

    trainer = FusionOBBTrainer(overrides=args)
    trainer.arch_spec = exp.get("arch_spec") or {"arch": "single"}
    trainer.pretrained_weights = exp.get("pretrained_weights")
    trainer.normalize = exp.get("normalize", "none")
    trainer.modality_stats = exp.get("modality_stats")
    trainer.train()

    save_dir = Path(getattr(trainer, "save_dir", run_dir))
    metrics = getattr(trainer, "metrics", {}) or {}
    result = {
        "id": exp["id"],
        "seed": seed,
        "modalities": exp["modalities"],
        "run_dir": str(run_dir),
        "save_dir": str(save_dir),
        "best_weights": str(save_dir / "weights" / "best.pt"),
        "params": sum(p.numel() for p in trainer.model.parameters()),
        "pretrained": trainer.pretrained_stats,
        "val_metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False),
                                         encoding="utf-8")
    return result
