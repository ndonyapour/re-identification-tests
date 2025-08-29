from pathlib import Path
import pandas as pd
import nibabel as nib
from reidentification_utils import get_subject_and_date


input_dir = Path("/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images")
csv_path = Path("/home/ubuntu/data/ADNI_dataset/BrainIAC_input_csv/brainiac_ADNI.csv")
csv_info_path = Path("/home/ubuntu/data/ADNI_dataset/BrainIAC_input_csv/brainiac_ADNI_info.csv")



def create_ADNI_BrainAIC_csv(input_dir: Path, csv_path: Path):
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


def create_ADNI_BrainAIC_info_csv(input_dir: Path, csv_path: Path):
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w") as f:
        f.write("filename,pat_id,date\n")
        for image_path in input_dir.glob("*.nii.gz"):
            image = nib.load(image_path)
            subject_id, date = get_subject_and_date(image, image_path)
            f.write(f"{image_path.name},{subject_id},{date}\n")
        
    print(f"Processed {csv_path} successfully")



if __name__ == "__main__":
    # create_ADNI_BrainAIC_csv(input_dir, csv_path)
    create_ADNI_BrainAIC_info_csv(input_dir, csv_info_path)

        
        










