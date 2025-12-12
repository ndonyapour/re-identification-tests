import os
import re
import nibabel as nib
import numpy as np
from itertools import combinations
from typing import List, Dict
import pandas as pd
from pathlib import Path
from reidentification_utils import find_closest_neighbors_ChestXray14, print_results, compute_precision_recall

def create_NIH_ChestXray14_csv(original_csv_path: Path, new_csv_path: Path):
    if not original_csv_path.exists():
        raise FileNotFoundError(f"Original CSV file not found: {original_csv_path}")
    df = pd.read_csv(original_csv_path)
    df = df[['Image Index','Patient ID']]
    df.columns = ['filename','pat_id']
    df.to_csv(new_csv_path, index=False)
    print(f"Processed {new_csv_path} successfully")


if __name__ == "__main__":
    features_dir = "/home/ubuntu/data/Chest_Xray_Images/ChestX-ray14_cxr_embeddings/"

    original_info_csv = "/home/ubuntu/data/Chest_Xray_Images/Data_Entry_2017_v2020.csv"
    info_csv = "/home/ubuntu/data/Chest_Xray_Images/NIH_ChestXray14_info.csv"
    output_dir = "./NIH_ChestX-ray14_CXR_reidentification_analysis"

    results_lists = []

    # create_NIH_ChestXray14_csv(Path(original_info_csv), Path(info_csv))
    for standardize in [False, True]:
        for features_name in ["contrastive_img_emb", "all_contrastive_img_emb"]:
            results = find_closest_neighbors_ChestXray14(
                features_dir=features_dir,
                info_csv=info_csv,
                features_name=features_name,
                n_neighbors=100,
                standardize=standardize,
                exclude_same_date=False,
                distance_threshold=-1.0,
                min_images_per_patient=2,
                output_dir=output_dir
            )

            # Convert results dictionary to list format for printing
            results = [
                results['features_name'],
                results['standardized'],
                results['n_features'],
                f"{results['r_at_1_img']}%",
                f"{results['r_at_10_img']}%",
                f"{results['image_ap']:.1%}",
                f"{results['r_at_1_patient']}%",
                f"{results['r_at_10_patient']}%",
                f"{results['patient_ap']:.1%}"
                ]

            results_lists.append(results)

    headers = [
        "Features", 
        "Standardized", 
        "Number of Features", 
        "R@1 (image)", 
        "R@10 (image)", 
        "AP (image)",
        "R@1 (patient)", 
        "R@10 (patient)", 
        "AP (patient)"
    ]

    print_results(results_lists, headers)