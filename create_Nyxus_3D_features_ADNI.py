from utils import run_3d_extraction
import time

if __name__ == "__main__":
    input_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images"
    mask_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/masks"
    out_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features"

    time_start = time.time()
    run_3d_extraction(input_dir, mask_dir, out_dir, mask_pattern="_mask")
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")