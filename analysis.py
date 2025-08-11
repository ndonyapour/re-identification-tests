import os
import re
import nibabel as nib
import numpy as np
from itertools import combinations
from typing import List, Dict
import pandas as pd
from reid_utils import create_imgage_data, find_closest_neighbors, print_results

if __name__ == "__main__":
    input_dir = "/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_n4_corrected"
    features_dir = "/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_n4_background_removed_features"
    
    input_dir = "/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_n4_corrected"
    output_csv = "image_metadata.csv"

    results = []
    create_imgage_data(input_dir, output_csv)
    for features_group in ["All", "Shape", "Texture"]:
        for standardize in [False, True]:
            result = find_closest_neighbors(features_dir, output_csv, standardize=standardize, features_group=features_group)
            results.append(result)
            # headers = ["Features", "Standardized", "R@1 (image)", "R@10 (image)", "R@1 (patient)", "R@10 (patient)"]
            # print_results(results, headers)
    
    headers = ["Features", "Standardized", "Number of Features", "R@1 (image)", "R@10 (image)", "R@1 (patient)", "R@10 (patient)"]
    print_results(results, headers)

    # # Calculate L2 distances
    # l2_distances = calculate_pairwise_metrics(input_dir, features_dir)
    
    # # Find 10 closest matches for each file
    # closest_matches = find_closest_matches(l2_distances, n_matches=10)
    
    # # Print closest matches report
    # print_closest_matches(input_dir,closest_matches)
    
    # # Save results to a JSON file
    # import json
    # with open('closest_matches.json', 'w') as f:
    #     # Convert to serializable format
    #     serializable_dict = {
    #         k: [(match, float(dist)) for match, dist in v] 
    #         for k, v in closest_matches.items()
    #     }
    #     json.dump(serializable_dict, f, indent=4)