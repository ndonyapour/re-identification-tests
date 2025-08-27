from pathlib import Path
import pandas as pd


input_dir = Path("/home/ubuntu/data/ADNI_dataset/brainaic_masks")
csv_path = Path("/home/ubuntu/data/ADNI_dataset/brainaic_csv/brainaic_masks.csv")

if not csv_path.exists():
    csv_path.parent.mkdir(parents=True, exist_ok=True)

with open(csv_path, "w") as f:
    f.write("pat_id,label\n")

    for mask_path in input_dir.glob("*.nii.gz"):
        suffixes: str = "".join(mask_path.suffixes)
        stem: str = mask_path.name[: -len(suffixes)]
        pat_id = stem
        label = str(0)
        f.write(f"{pat_id},{label}\n")

print(f"Processed {csv_path} successfully")



        
        










