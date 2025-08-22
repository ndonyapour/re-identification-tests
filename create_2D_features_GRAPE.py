import nyxus
import os
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def save_features_pkl(image_dir:str, features_dir: str, out_path: str) -> None:
    """Save features to a pickle file.
    
    Args:
        features_dir: Directory containing the features
        out_dir: Output directory for results
    """
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg'))]
    features = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        feature_path = os.path.join(features_dir, image_file.replace('.jpg', '.csv').replace('.jpeg', '.csv'))
        # Read CSV without using first column as index
        latent_features = pd.read_csv(feature_path, 
                     engine='python',  # More robust parsing
                     encoding='utf-8', sep=',', usecols=lambda x: x != 'Unnamed: 0')  
        # latent_features = df.drop(columns=['Unnamed: 0'])
        # Or if you want to explicitly drop the index column if it was saved:
        # latent_features = pd.read_csv(feature_path).drop('Unnamed: 0', axis=1, errors='ignore')
        numeric_cols = latent_features.select_dtypes(include=[np.number]).columns
        feature_array = latent_features[numeric_cols].iloc[0].values
       
        features.append(feature_array)

    # Standardize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    features_dict = [] 
    for image_file, feature in zip(image_files, features):
        features_dict.append({'path': image_file, 'features': feature})
    
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    with open(out_path, 'wb') as f:
        pkl.dump(features_dict, f)


#  nyx: nyxus.Nyxus=None
def process_image(image_file: str, input_dir: str, out_dir: str, feature_types: list=["*WHOLESLIDE*"]) -> dict:
    """Process a single image and extract features.
    
    Args:
        image_file: Name of the image file
        input_dir: Directory containing the image
        nyx: Nyxus instance for feature extraction
        
    Returns:
        dict: Dictionary containing image path and extracted features
    """
    # import pdb; pdb.set_trace()
    nyx = nyxus.Nyxus(feature_types,  n_feature_calc_threads=5)
    image_path = os.path.join(input_dir, image_file)
    image = pil_loader(image_path)
    image = np.array(image)
    image_gray = image.mean(axis=2)  # Average across RGB channels
    mask = np.ones(image_gray.shape)
    #import pdb; pdb.set_trace()
    #print("nyx.feature_types", nyx.features)
    print(f"Processing {image_file}")
    print(f"Image shape: {image_gray.shape}")
    print(f"Mask shape: {mask.shape}")

    latent_features = nyx.featurize(image_gray, mask)
    print(latent_features.shape, "latent_features.shape before saving")
    latent_features.to_csv(os.path.join(out_dir, image_file.replace('.jpg', '.csv').replace('.jpeg', '.csv')), index=False)
    print(f"Saved to {os.path.join(out_dir, image_file.replace('.jpg', '.csv').replace('.jpeg', '.csv'))}")


def run_2d_extraction(input_dir: str, out_dir: str, feature_types: list=["*WHOLESLIDE*"]) -> None:
    """Extract 2D features using parallel processing.
    
    Args:
        input_dir: Path for input intensity images
        out_dir: Output directory for results
        output_path: Path to save extracted features
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize Nyxus
    # nyx = nyxus.Nyxus(feature_types,  n_feature_calc_threads=5)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg'))]
    
    # Determine number of processes (use 75% of available CPUs)
    num_processes = max(1, int(cpu_count() * 0.75))
    print(f"Using {num_processes} processes")

    # Create partial function with fixed arguments
    process_func = partial(process_image, input_dir=input_dir, out_dir=out_dir, feature_types=feature_types)
    
    # Process images in parallel
    with Pool(processes=num_processes) as pool:
        latent_dict = list(tqdm(
            pool.imap(process_func, image_files),
            total=len(image_files),
            desc="Extracting features"
        ))

def calculate_l2_distance(features_dir: str) -> dict:
    """Calculate L2 distance between two images.
    
    Args:
        features_dir: Directory containing the features
        output_path: Path to save the results
    """
    features_files = [f for f in os.listdir(features_dir) if f.endswith(('.csv'))]
    features_list = []
    for feature_file in features_files:
        feature_path = os.path.join(features_dir, feature_file)
        # Read CSV without using first column as index
        latent_features = pd.read_csv(feature_path, usecols=lambda x: x != 'Unnamed: 0')
        # Or if you want to explicitly drop the index column if it was saved:
        # latent_features = pd.read_csv(feature_path).drop('Unnamed: 0', axis=1, errors='ignore')
        numeric_cols = latent_features.select_dtypes(include=[np.number]).columns
        feature_array = latent_features[numeric_cols].iloc[0].values
        features_list.append(feature_array)
    features = np.array(features_list)
    nn = NearestNeighbors(n_neighbors=11, metric="euclidean")  # 1 extra for self
    nn.fit(features)
    dists, idxs = nn.kneighbors(features) 
    # idxs[i] lists neighbors for query i (self at 0)

def benchmark_2d_extraction(input_dir: str, out_dir: str, feature_types: list=["*WHOLESLIDE*"], n_tests=10) -> None:
    """Benchmark 2D extraction.
    
    Args:
        input_dir: Path for input intensity images
        out_dir: Output directory for results
        output_path: Path to save extracted features
    """

    # Initialize Nyxus
    #nyx = nyxus.Nyxus(feature_types, n_feature_calc_threads=5)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg'))]

    # Determine number of processes (use 75% of available CPUs)
    random_indices = np.random.choice(len(image_files), size=n_tests, replace=False)
    total_time = 0
    for idx in random_indices:
        time_start = time.time()
        image_file = image_files[idx]
        process_image(image_file, input_dir, out_dir, feature_types)
        time_end = time.time()
        total_time += time_end - time_start
    print(f"Total time taken: {total_time} seconds")
    print(f"Average time per image: {total_time / n_tests} seconds")



if __name__ == "__main__":
    images_input_dir = "./GRAPE/CFPs"
    features_out_dir = "./GRAPE_Nyxus_ALL/CSV_files"
    output_path = "./GRAPE_Nyxus_ALL/PKL_file/ALL_2D_features.pkl"
    start_time = time.time()
    run_2d_extraction(images_input_dir, features_out_dir, feature_types=["*ALL*"])

    save_features_pkl(images_input_dir, features_out_dir, output_path)

    end_time = time.time()
    print(f"Time taken: {(end_time - start_time)/60} minutes")
    print(f"Average time per image: {(end_time - start_time)/60 / 631} minutes")

