import os
import re
import nibabel as nib
import numpy as np
from itertools import combinations
from typing import List, Dict
import pandas as pd
from reidentification_utils import find_closest_neighbors_Nyxus, print_results, compute_precision_recall

if __name__ == "__main__":
    image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
    info_csv = "/home/ubuntu/data/ADNI_dataset/BrainIAC_input_csv/brainiac_ADNI_info.csv"
    output_dir = "./Nyxus_reidentification_analysis"

    results_lists = []

    for standardize in [False, True]:
        for features_group in ["All" , "Texture", "Shape"]:
            results = find_closest_neighbors_Nyxus(
                features_dir=features_dir,
                image_dir=image_dir,
                info_csv=info_csv,
                features_group=features_group,
                n_neighbors=100,
                standardize=standardize,
                exclude_same_date=True,
                distance_threshold=-1.0,
                output_dir=output_dir
            )

            # Convert results dictionary to list format for printing
            results_list = [
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
            import pdb; pdb.set_trace()
            results_lists.append(results_list)

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
