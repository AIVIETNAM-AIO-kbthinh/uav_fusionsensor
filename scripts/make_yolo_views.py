"""Tao "view" theo dung quy uoc thu muc cua Ultralytics cho tung modality.

Ultralytics suy ra duong dan nhan bang cach thay '/images/' -> '/labels/' trong
duong dan anh. Thu muc `imagesr/` cua ta khong khop quy uoc do, nen ta dung
HARD LINK (khong ton them dung luong tren cung o dia NTFS) de dung ra:

    data/dronevehicle/rgb/{images,labels}/{split}
    data/dronevehicle/ir/{images,labels}/{split}
    data/vedai/rgb/{images,labels}   +  data/vedai/ir/{images,labels}

Ca hai view dung CHUNG mot bo nhan (annotation IR cua DroneVehicle - GT thong
nhat theo quyet dinh o plan.md muc 3.1).
"""
import argparse, glob, os, shutil


def link(src, dst, mode):
    if os.path.exists(dst):
        return 0
    try:
        if mode == "hardlink":
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
        return 1
    except OSError:
        shutil.copy2(src, dst)
        return 1


def build(pairs, mode):
    n = 0
    for src_dir, dst_dir, ext in pairs:
        os.makedirs(dst_dir, exist_ok=True)
        for p in glob.glob(f"{src_dir}/*{ext}"):
            n += link(p, os.path.join(dst_dir, os.path.basename(p)), mode)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    a = ap.parse_args()

    total = 0
    dv = "data/dronevehicle"
    if os.path.isdir(dv):
        for split in ("train", "val", "test"):
            total += build([
                (f"{dv}/images/{split}",  f"{dv}/rgb/images/{split}", ".jpg"),
                (f"{dv}/labels/{split}",  f"{dv}/rgb/labels/{split}", ".txt"),
                (f"{dv}/imagesr/{split}", f"{dv}/ir/images/{split}",  ".jpg"),
                (f"{dv}/labels/{split}",  f"{dv}/ir/labels/{split}",  ".txt"),
            ], a.mode)
            print(f"[dronevehicle/{split}] done ({total} link)", flush=True)

    vd = "data/vedai"
    if os.path.isdir(vd):
        total += build([
            (f"{vd}/images",  f"{vd}/rgb/images", ".png"),
            (f"{vd}/labels",  f"{vd}/rgb/labels", ".txt"),
            (f"{vd}/imagesr", f"{vd}/ir/images",  ".png"),
            (f"{vd}/labels",  f"{vd}/ir/labels",  ".txt"),
        ], a.mode)
        # danh sach anh theo fold cho tung view
        for f in sorted(glob.glob(f"{vd}/folds/fold*_*.txt")):
            base = os.path.splitext(os.path.basename(f))[0]
            ids = [l.strip() for l in open(f) if l.strip()]
            for view in ("rgb", "ir"):
                out = f"{vd}/{view}/{base}.txt"
                with open(out, "w") as g:
                    g.write("\n".join(os.path.abspath(f"{vd}/{view}/images/{i}.png") for i in ids) + "\n")
        print(f"[vedai] done ({total} link)", flush=True)

    print(f"TOTAL {total} entries ({a.mode})")


if __name__ == "__main__":
    main()
