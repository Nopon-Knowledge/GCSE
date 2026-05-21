import pandas as pd, shutil
from pathlib import Path

root = Path("/storage/home/402005/datasets/imagenet-1k/imagenet-object-localization-challenge(1)/ILSVRC")
csv_path = root.parent / "LOC_val_solution.csv"   # Often alongside ILSVRC; update if your path differs
val_dir = root / "Data/CLS-LOC/val"
out_dir = root / "Data/CLS-LOC/val_by_wnid_csv"

out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)  # columns: ImageId, PredictionString
miss = 0
for _, r in df.iterrows():
    img_id = r["ImageId"]
    wnid = str(r["PredictionString"]).split()[0]  # First token is the wnid
    src = val_dir / f"{img_id}.JPEG"
    if not src.exists():
        miss += 1
        continue
    dst = out_dir / wnid
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst / src.name)

print("copied:", len(df) - miss, "miss:", miss, "classes:", len(list(out_dir.iterdir())))
