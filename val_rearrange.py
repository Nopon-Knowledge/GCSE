import os, shutil, xml.etree.ElementTree as ET
from pathlib import Path

root = Path("/storage/home/402005/datasets/imagenet-1k/imagenet-object-localization-challenge(1)/ILSVRC")
val_img_dir = root / "Data/CLS-LOC/val"
val_xml_dir = root / "Annotations/CLS-LOC/val"
out_dir = root / "Data/CLS-LOC/val_by_wnid"

out_dir.mkdir(parents=True, exist_ok=True)

bad = 0
for xml_path in val_xml_dir.glob("*.xml"):
    # XML filename usually matches the image basename (without extension).
    stem = xml_path.stem
    # Common image extension is JPEG (could be jpg/png; adjust as needed).
    img_path = val_img_dir / f"{stem}.JPEG"
    if not img_path.exists():
        # Fallback: try any extension.
        cand = list(val_img_dir.glob(stem + ".*"))
        if not cand:
            bad += 1
            continue
        img_path = cand[0]

    tree = ET.parse(xml_path)
    root_xml = tree.getroot()

    # Use the first object's name as the class (wnid, e.g., n01440764).
    obj = root_xml.find("object")
    if obj is None or obj.find("name") is None:
        bad += 1
        continue
    wnid = obj.find("name").text.strip()

    dst_dir = out_dir / wnid
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, dst_dir / img_path.name)

print("done. missing/bad:", bad)
print("output:", out_dir)
