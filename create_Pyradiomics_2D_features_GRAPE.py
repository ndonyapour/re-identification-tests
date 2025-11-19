import os
from traceback import print_tb
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings
from utils_pyradiomics import PyRadiomicsExtractor

if __name__ == "__main__":
    images_input_dir = "/home/ubuntu/data/GRAPE/CFPs"
    features_out_dir = "GRAPE_Pyradiomics_WHOLESLIDE"
    os.makedirs(features_out_dir, exist_ok=True)

    # Initialize extractor
    extractor_2d = PyRadiomicsExtractor(
        dimension='2D', 
        bin_width=5, 
        voxel_array_shift=0, 
        normalize=False, 
        normalize_scale=100, 
        interpolator='sitkBSpline', 
        resample_pixel_spacing=None
    )

    # Get the list of PyRadiomics features corresponding to Nyxus wholeslide
    wholeslide_features = extractor_2d.get_pyradiomics_wholeslide_features()
    print(f"PyRadiomics can calculate {len(wholeslide_features)} features corresponding to Nyxus wholeslide")

    # Enable only those specific features (MUCH FASTER!)
    extractor_2d.enable_specific_features(wholeslide_features)
    print(f"Enabled {len(wholeslide_features)} features for extraction")

    # Get image files
    image_files = [f for f in os.listdir(images_input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images")

    image_path = os.path.join(images_input_dir, image_files[0])
    csv_path = os.path.join(features_out_dir, image_files[0].replace('.jpg', '_features.csv').replace('.jpeg', '_features.csv').replace('.png', '_features.csv'))

    features = extractor_2d.extract_features_from_image_only(
        image_path=image_path,
        csv_path=csv_path
    )
    import pdb; pdb.set_trace()

    # Extract features for each image
    # for image_file in image_files:
    #     image_path = os.path.join(images_input_dir, image_file)
    #     csv_filename = image_file.replace('.jpg', '_features.csv').replace('.jpeg', '_features.csv').replace('.png', '_features.csv')
    #     csv_path = os.path.join(features_out_dir, csv_filename)
        
    #     try:
    #         features = extractor_2d.extract_features_from_image_only(
    #             image_path=image_path,
    #             csv_path=csv_path
    #         )
    #         print(f"Extracted features from {image_file}")
    #     except Exception as e:
    #         print(f"Error processing {image_file}: {str(e)}")
    #         continue

    # print("Feature extraction complete!")