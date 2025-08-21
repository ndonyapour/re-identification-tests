import nyxus
import os
from pathlib import Path
import numpy as np
from PIL import Image
import time

def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def process_image(image_file: str, input_dir: str, out_dir: str, nyx: nyxus.Nyxus) -> dict:
    """Process a single image and extract features.
    
    Args:
        image_file: Name of the image file
        input_dir: Directory containing the image
        nyx: Nyxus instance for feature extraction
        
    Returns:
        dict: Dictionary containing image path and extracted features
    """
    image_path = os.path.join(input_dir, image_file)
    image = pil_loader(image_path)
    image = np.array(image)
    image_gray = image.mean(axis=2)  # Average across RGB channels
    mask = np.ones(image_gray.shape)
    
    print(f"Processing {image_file}")
    print(f"Image shape: {image_gray.shape}")
    print(f"Mask shape: {mask.shape}")

    latent_features = nyx.featurize(image_gray, mask)
    latent_features.to_csv(os.path.join(out_dir, image_file.replace('.jpg', '.csv').replace('.jpeg', '.csv')), index=False)


def benchmark_2d_extraction(input_dir: str, out_dir: str, feature_types: list=["*WHOLESLIDE*"], n_tests=10) -> None:
    """Benchmark 2D extraction.
    
    Args:
        input_dir: Path for input intensity images
        out_dir: Output directory for results
        output_path: Path to save extracted features
    """

    # Initialize Nyxus
    nyx = nyxus.Nyxus(feature_types,  n_feature_calc_threads=5)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg'))]

    # Determine number of processes (use 75% of available CPUs)
    random_indices = np.random.choice(len(image_files), size=n_tests, replace=False)
    total_time = 0
    for idx in random_indices:
        time_start = time.time()
        image_file = image_files[idx]
        process_image(image_file, input_dir, out_dir, nyx)
        time_end = time.time()
        total_time += time_end - time_start
    print(f"Total time taken: {total_time} seconds")
    print(f"Average time per image: {total_time / n_tests} seconds")


if __name__ == "__main__":
    images_input_dir = "./GRAPE/CFPs"
    output_path = "./output"
    benchmark_2d_extraction(images_input_dir, output_path, feature_types=["*ALL*"], n_tests=10)
