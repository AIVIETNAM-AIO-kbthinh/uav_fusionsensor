import zipfile, hashlib, os, time
SRC=r"C:/Users/Admin/.cache/huggingface/hub/datasets--McCheng--DroneVehicle/snapshots/947bd13f002857eb79a3e5b4feb2c6aca453ade7"
blobs={"train.zip":"d22eccae518728352b40bb758b383e64db2b1b38e3d8c5d14406724dc869614f",
       "test.zip":"94bf47e493e7a57fd5d900439f614f339e7b6c7181e4d781bcfebaeb963ffd8e",
       "val.zip":"043b7944ebb8ce076c1e5cfd37c33de6a59a9f62cf47c0f028387f703d4f5250"}
for z,expect in blobs.items():
    p=os.path.realpath(os.path.join(SRC,z)); t0=time.time()
    h=hashlib.sha256()
    with open(p,'rb') as f:
        while (b:=f.read(1<<22)): h.update(b)
    got=h.hexdigest()
    print(f"[{z}] sha256 {'OK  ' if got==expect else 'MISMATCH'} got={got[:16]} exp={expect[:16]} ({time.time()-t0:.0f}s)", flush=True)
    zf=zipfile.ZipFile(p); names=[n for n in zf.namelist() if not n.endswith('/')]
    bad=[]
    for i,n in enumerate(names):
        try:
            with zf.open(n) as fh:
                while fh.read(1<<20): pass
        except Exception as e:
            bad.append((n,type(e).__name__))
        if i%10000==0: print(f"   crc {i}/{len(names)} bad={len(bad)}", flush=True)
    print(f"[{z}] CRC scan: {len(bad)} bad of {len(names)}", flush=True)
    for n,e in bad[:40]: print("    BAD", n, e, flush=True)
    with open(rf"g:/uav_fusionsensor/data/bad_{z}.txt","w") as f:
        f.write("\n".join(n for n,_ in bad))
print("VERIFY DONE", flush=True)
