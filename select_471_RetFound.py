import os
import pandas as pd
import numpy as np
import pickle as pkl

def select_471_RetFound(features_pkl_path: str, output_pkl_path: str) -> dict:
    """Select 471 RetFound images from the features directory.
    
    Args:
        features_dir: Directory containing the features
        output_path: Path to save the results
    """
    with open(features_pkl_path, 'rb') as f:
        features_dict = pkl.load(f)

    clipped_features = []
    for feature in features_dict:
        clipped_feature = feature["features"][:471]
        clipped_features.append({"path": feature["path"], "features": clipped_feature})

    if not os.path.exists(os.path.dirname(output_pkl_path)):
        os.makedirs(os.path.dirname(output_pkl_path))

    with open(output_pkl_path, 'wb') as f:
        pkl.dump(clipped_features, f)


if __name__ == "__main__":
    features_pkl_path = "/home/ubuntu/projects/re_identification/my_features/GRAPE/features_0.pkl"
    output_pkl_path = "/home/ubuntu/projects/re_identification/my_features/GRAPE_471/features_0_471.pkl"
    select_471_RetFound(features_pkl_path, output_pkl_path)











