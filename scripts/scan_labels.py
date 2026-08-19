"""Quet toan bo XML de liet ke ten lop va kieu hinh hoc truoc khi chot mapping."""
import collections, glob, sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor


def scan(f):
    try:
        root = ET.parse(f).getroot()
    except Exception:
        return (("__PARSE_ERROR__", "__PARSE_ERROR__"), 1),
    c = collections.Counter()
    for o in root.findall("object"):
        g = ("polygon" if o.find("polygon") is not None
             else "bndbox" if o.find("bndbox") is not None else "NONE")
        c[((o.findtext("name") or "").strip(), g)] += 1
    return tuple(c.items())


def main():
    tot = collections.Counter()
    per_mod = {"rgb": collections.Counter(), "ir": collections.Counter()}
    for split in ["train", "val", "test"]:
        for lab, mod in [(f"{split}label", "rgb"), (f"{split}labelr", "ir")]:
            files = glob.glob(f"data/raw/{split}/{lab}/*.xml")
            with ProcessPoolExecutor(16) as ex:
                for items in ex.map(scan, files, chunksize=200):
                    for k, v in items:
                        tot[k] += v
                        per_mod[mod][k[0]] += v
    names, geom = collections.Counter(), collections.Counter()
    for (n, g), v in tot.items():
        names[n] += v
        geom[g] += v
    print("CLASS NAMES (all splits, both modalities):")
    for n, v in names.most_common():
        print(f"   {n!r:22s} total={v:8d}  rgb={per_mod['rgb'][n]:8d}  ir={per_mod['ir'][n]:8d}")
    print("\nGEOMETRY:", dict(geom))
    print("\nname x geometry pairs with NONE/odd geometry:")
    for (n, g), v in sorted(tot.items(), key=lambda kv: -kv[1]):
        if g != "polygon":
            print(f"   {n!r:22s} {g:8s} {v:8d}")


if __name__ == "__main__":
    main()
