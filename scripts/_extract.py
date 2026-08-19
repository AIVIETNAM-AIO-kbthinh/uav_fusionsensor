import zipfile, os, time, json
SRC=r"C:/Users/Admin/.cache/huggingface/hub/datasets--McCheng--DroneVehicle/snapshots/947bd13f002857eb79a3e5b4feb2c6aca453ade7"
DST=r"g:/uav_fusionsensor/data/raw"
bad_all={}
for z in ["val.zip","test.zip","train.zip"]:
    p=os.path.join(SRC,z); t0=time.time(); zf=zipfile.ZipFile(p)
    names=[n for n in zf.namelist() if not n.endswith('/')]; n=len(names); bad=[]; skipped=0
    print(f"[{z}] {n} files", flush=True)
    for i,nm in enumerate(names):
        out=os.path.join(DST,nm)
        if os.path.exists(out) and os.path.getsize(out)==zf.getinfo(nm).file_size:
            skipped+=1; continue
        try:
            zf.extract(nm, DST)
        except Exception as e:
            bad.append(nm)
            if os.path.exists(out): os.remove(out)
        if i%10000==0: print(f"   {i}/{n} bad={len(bad)} cached={skipped} ({time.time()-t0:.0f}s)", flush=True)
    bad_all[z]=bad
    print(f"[{z}] done {time.time()-t0:.0f}s | extracted={n-len(bad)-skipped} cached={skipped} bad={len(bad)}", flush=True)
json.dump(bad_all, open(r"g:/uav_fusionsensor/data/corrupt_files.json","w"), indent=1)
print("ALL DONE", flush=True)
