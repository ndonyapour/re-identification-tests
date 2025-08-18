from nyxus import Nyxus
import os
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import numpy as np
import pandas as pd


from PIL import Image

def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
    

def run_2d_extraction(input_dir: str, out_dir: str) -> None:
    """Test 2D feature extraction.
    
    Args:
        input_files: Path pattern for input intensity images
        seg_files: Path pattern for segmentation masks
        out_dir: Output directory for results
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    #input_dir = Path(input_dir)
    nyx = Nyxus (["*ALL*"])
    #nyx.set_params(features=["ALL"], neighbor_distance=5, pixels_per_micron=1.0)
    features = nyx.featurize_directory(input_dir)
    # for input_file in tqdm(input_dir.iterdir()):
    #     if str(input_file).endswith(".jpg") or str(input_file).endswith(".jpeg"):
    #         outpath = Path(out_dir)
    #         print(f"Running 2D feature extraction for {input_file.name}...")
    #         nyxus = Nyxus()
    #         nyx = Nyxus(features=["ALL"])
    #         nyx_params = {
    #             "neighbor_distance": 5,
    #             "pixels_per_micron": 1.0,
    #         }
    #         nyx.set_params(**nyx_params)
    #         int_slice = int_slice.astype(np.float32)
    #         mk_slice = mk_slice.astype(np.uint8)
    #         feats_df = nyx.featurize(int_slice, mk_slice)
    #         feats_df.to_csv(outpath / f"{input_file.stem}.csv", index=False)
    
    #         print(f"2D results saved to: {outpath}/{input_file.stem}.csv")


def save_features(image_dir, feature_dir, output_path):
    """Save the features"""
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.jpeg')]
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        feature_path = os.path.join(feature_dir, image_file.replace('.jpg', '.csv').replace('.jpeg', '.csv'))
        latent_features = pd.read_csv(feature_path)
        latent_dict.append({'path': image_path, 'features': latent_features})

    latent_features = np.concatenate(latent_features)
    latent_dict: list = [{'path': path, 'features': feats} for path, feats in zip(data.paths, latent_features)]

    # save features

    with open(output_path, 'wb') as f_out:
        pkl.dump(latent_dict, f_out)

if __name__ == "__main__":
    input_dir = "./GRAPE/CFPs"
    out_dir = "./features/GRAPE_Nyxus_2D_features"
    output_path = "./features/GRAPE_Nyxus_2D_features.pkl"
    run_2d_extraction(input_dir, out_dir)
    save_features(input_dir, out_dir, output_path)