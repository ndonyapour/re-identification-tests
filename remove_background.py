import os
import ants
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import run_3d_extraction

def create_otsu_mask(
    input_image_path: str, 
    output_mask_path: str, 
    n_classes: int = 4,
    mask_background: bool = True
) -> None:
    """
    Create an Otsu threshold mask using ANTsPy.
    
    Args:
        input_image_path: Path to input image
        output_mask_path: Path to save the output mask
        n_classes: Number of Otsu classes (default=4)
        mask_background: If True, creates binary mask where background=0 (default=True)
    """
    # Read image using ANTsPy
    img = ants.image_read(input_image_path)
    
    # Apply Otsu thresholding (output will be 0 to n_classes)
    mask = ants.threshold_image(img, 'Otsu', n_classes)
    
    # Class 0: Background/air
    # Class 1: CSF/some soft tissue
    # Class 2: Gray matter
    # Class 3: White matter
    # Class 4: Skull/high intensity
    if mask_background:
        # Keep only the brain tissue classes (2 and 3 for brain tissue)
        mask = ants.threshold_image(
            mask,
            low_thresh=0.5,  # Keep classes 1, 2, and 3 (brain tissue)
            high_thresh=n_classes,
            inval=1,
            outval=0
        )
    
    # Save the mask
    ants.image_write(mask, output_mask_path)
    
    return mask
def create_otsu_mask_for_all_images(input_dir: str, out_dir: str) -> None:
    """
    Create Otsu masks for all images in the input directory.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f"Creating Otsu masks for all images in {input_dir} and saving to {out_dir}")
    for file in tqdm(os.listdir(input_dir)):
        if file.endswith(".nii"):
            input_file = os.path.join(input_dir, file)
            output_file = os.path.join(out_dir, file.replace(".nii", "_whole_brain_mask.nii.gz"))
            create_otsu_mask(input_file, output_file)

# Example usage
if __name__ == "__main__":
    # input_file = "input_image.nii.gz"
    # output_file = "otsu_mask.nii.gz"
    
    input_dir = "/home/ubuntu/data/ADNI_dataset/t1_mpr"
    out_dir = "/home/ubuntu/data/ADNI_dataset/t1_mpr_whole_brain_masks"
    create_otsu_mask_for_all_images(input_dir, out_dir)