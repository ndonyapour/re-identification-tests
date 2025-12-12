import os
import re
import nibabel as nib
import numpy as np
from itertools import combinations
from typing import List, Dict
import pandas as pd
from reidentification_utils import extract_Nyxus_features
image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
info_csv = "/home/ubuntu/data/ADNI_dataset/BrainIAC_input_csv/brainiac_ADNI_info.csv"




firstorder_features = extract_Nyxus_features(
    features_dir=features_dir,
    features_group="All",
)
