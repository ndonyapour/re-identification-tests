import os
import re
import nibabel as nib
import numpy as np
from itertools import combinations
from typing import List, Dict
import pandas as pd
from reidentification_utils import find_closest_neighbors_BrainAIC, print_results

if __name__ == "__main__":
    features_dir = "/home/ubuntu/data/ADNI_dataset/brainaic_features/features.csv"
    output_dir = "/home/ubuntu/data/ADNI_dataset/brainaic_features"

    results = find_closest_neighbors_BrainAIC(features_dir, 
                                            n_neighbors=10, 
                                            standardize=False, 
                                            exclude_same_date=True, 
                                            distance_threshold=-1.0, 
                                            output_dir=output_dir)
    headers = ["Features", "Standardized", "Number of Features", "R@1 (image)", "R@10 (image)", "AP (image)", 
            "R@1 (patient)", "R@10 (patient)", "AP (patient)"]
    print_results([results], headers)
