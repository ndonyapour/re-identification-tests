import os
import re
import nibabel as nib
from itertools import combinations
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from tabulate import tabulate


TEST_RESULTS_TEMPLATE = """
* Test Results:
** Number of images: {num_images}
** Number of unique patients: {num_patients}
** Number of unique dates: {num_dates}
** R@1 (image): {r_at_1_img:.1f}%
** R@10 (image): {r_at_10_img:.1f}%
** R@1 (patient): {r_at_1_patient:.1f}%
** R@10 (patient): {r_at_10_patient:.1f}%
"""

SHAPE_FEATURES = [
    "3AREA",
    "3AREA_2_VOLUME",
    "3COMPACTNESS1",
    "3COMPACTNESS2",
    "3MESH_VOLUME",
    "3SPHERICAL_DISPROPORTION",
    "3SPHERICITY",
    "3SURFACE_AREA",
    "3VOLUME_CONVEXHULL",
    "3VOXEL_VOLUME"
]
TEXTURE_FEATURES = [
    '3GLCM_ACOR', '3GLCM_ASM', '3GLCM_CLUPROM', '3GLCM_CLUSHADE', '3GLCM_CLUTEND', '3GLCM_CONTRAST', '3GLCM_CORRELATION', 
    '3GLCM_DIFAVE', '3GLCM_DIFENTRO', '3GLCM_DIFVAR', '3GLCM_DIS', '3GLCM_ENERGY', '3GLCM_ENTROPY', '3GLCM_HOM1', '3GLCM_HOM2', 
    '3GLCM_ID', '3GLCM_IDN', '3GLCM_IDM', '3GLCM_IDMN', '3GLCM_INFOMEAS1', '3GLCM_INFOMEAS2', '3GLCM_IV', '3GLCM_JAVE', '3GLCM_JE', 
    '3GLCM_JMAX', '3GLCM_JVAR', '3GLCM_SUMAVERAGE', '3GLCM_SUMENTROPY', '3GLCM_SUMVARIANCE', '3GLCM_VARIANCE', '3GLCM_ASM_AVE', 
    '3GLCM_ACOR_AVE', '3GLCM_CLUPROM_AVE', '3GLCM_CLUSHADE_AVE', '3GLCM_CLUTEND_AVE', '3GLCM_CONTRAST_AVE', '3GLCM_CORRELATION_AVE', 
    '3GLCM_DIFAVE_AVE', '3GLCM_DIFENTRO_AVE', '3GLCM_DIFVAR_AVE', '3GLCM_DIS_AVE', '3GLCM_ENERGY_AVE', '3GLCM_ENTROPY_AVE', 
    '3GLCM_HOM1_AVE', '3GLCM_ID_AVE', '3GLCM_IDN_AVE', '3GLCM_IDM_AVE', '3GLCM_IDMN_AVE', '3GLCM_IV_AVE', '3GLCM_JAVE_AVE', 
    '3GLCM_JE_AVE', '3GLCM_INFOMEAS1_AVE', '3GLCM_INFOMEAS2_AVE', '3GLCM_VARIANCE_AVE', '3GLCM_JMAX_AVE', '3GLCM_JVAR_AVE', 
    '3GLCM_SUMAVERAGE_AVE', '3GLCM_SUMENTROPY_AVE', '3GLCM_SUMVARIANCE_AVE', '3GLDM_SDE', '3GLDM_LDE', '3GLDM_GLN', '3GLDM_DN', 
    '3GLDM_DNN', '3GLDM_GLV', '3GLDM_DV', '3GLDM_DE', '3GLDM_LGLE', '3GLDM_HGLE', '3GLDM_SDLGLE', '3GLDM_SDHGLE', '3GLDM_LDLGLE', 
    '3GLDM_LDHGLE', '3GLDZM_SDE', '3GLDZM_LDE', '3GLDZM_LGLZE', '3GLDZM_HGLZE', '3GLDZM_SDLGLE', '3GLDZM_SDHGLE', '3GLDZM_LDLGLE', 
    '3GLDZM_LDHGLE', '3GLDZM_GLNU', '3GLDZM_GLNUN', '3GLDZM_ZDNU', '3GLDZM_ZDNUN', '3GLDZM_ZP', '3GLDZM_GLM', '3GLDZM_GLV', '3GLDZM_ZDM', 
    '3GLDZM_ZDV', '3GLDZM_ZDE', '3NGLDM_LDE', '3NGLDM_HDE', '3NGLDM_LGLCE', '3NGLDM_HGLCE', '3NGLDM_LDLGLE', '3NGLDM_LDHGLE', 
    '3NGLDM_HDLGLE', '3NGLDM_HDHGLE', '3NGLDM_GLNU', '3NGLDM_GLNUN', '3NGLDM_DCNU', '3NGLDM_DCNUN', '3NGLDM_DCP', '3NGLDM_GLM', 
    '3NGLDM_GLV', '3NGLDM_DCM', '3NGLDM_DCV', '3NGLDM_DCENT', '3NGLDM_DCENE', '3NGTDM_COARSENESS', '3NGTDM_CONTRAST', '3NGTDM_BUSYNESS', 
    '3NGTDM_COMPLEXITY', '3NGTDM_STRENGTH', '3GLSZM_SAE', '3GLSZM_LAE', '3GLSZM_GLN', '3GLSZM_GLNN', '3GLSZM_SZN', '3GLSZM_SZNN',
    '3GLSZM_ZP', '3GLSZM_GLV', '3GLSZM_ZV', '3GLSZM_ZE', '3GLSZM_LGLZE', '3GLSZM_HGLZE', '3GLSZM_SALGLE', '3GLSZM_SAHGLE', 
    '3GLSZM_LALGLE', '3GLSZM_LAHGLE', '3GLRLM_SRE', '3GLRLM_LRE', '3GLRLM_GLN', '3GLRLM_GLNN', '3GLRLM_RLN', '3GLRLM_RLNN', 
    '3GLRLM_RP', '3GLRLM_GLV', '3GLRLM_RV', '3GLRLM_RE', '3GLRLM_LGLRE', '3GLRLM_HGLRE', '3GLRLM_SRLGLE', '3GLRLM_SRHGLE', 
    '3GLRLM_LRLGLE', '3GLRLM_LRHGLE', '3GLRLM_SRE_AVE', '3GLRLM_LRE_AVE', '3GLRLM_GLN_AVE', '3GLRLM_GLNN_AVE', '3GLRLM_RLN_AVE', 
    '3GLRLM_RLNN_AVE', '3GLRLM_RP_AVE', '3GLRLM_GLV_AVE', '3GLRLM_RV_AVE', '3GLRLM_RE_AVE', '3GLRLM_LGLRE_AVE', '3GLRLM_HGLRE_AVE', 
    '3GLRLM_SRLGLE_AVE', '3GLRLM_SRHGLE_AVE', '3GLRLM_LRLGLE_AVE', '3GLRLM_LRHGLE_AVE']


def get_subject_and_date(nifti_img, filename: str = '') -> tuple[str, str]:
    """
    Get subject ID and scan date from a NIfTI file.
    
    Args:
        nifti_img: A NiBabel image object
        filename: Optional filename to extract information from
        
    Returns:
        tuple: (subject_id, scan_date) where either could be 'unknown' if not found
    """
    subject_id = 'unknown'
    scan_date = 'unknown'
    
    # Get header information
    header = nifti_img.header
    descrip = header.get('descrip', '').tobytes().decode('utf-8').lower()
    db_name = header.get('db_name', '').tobytes().decode('utf-8').lower()
    
    # Try to find subject ID
    subject_patterns = [
        r'(sub-\w+)',           # BIDS format
        r'(adni_\d+)',          # ADNI format
        r'(s\d{4})',            # ADNI S#### format
        r'(subject[-_]\w+)'     # General format
    ]
    
    # Look in both header fields and filename
    text_to_search = f"{descrip} {db_name} {filename}".lower()
    
    for pattern in subject_patterns:
        match = re.search(pattern, text_to_search)
        if match:
            subject_id = match.group(1)
            break
    
    # Try to find date
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'(\d{8})',              # YYYYMMDD
        r'(\d{2}/\d{2}/\d{4})'   # MM/DD/YYYY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text_to_search)
        if match:
            scan_date = match.group(1)
            # Convert YYYYMMDD to YYYY-MM-DD if needed
            if len(scan_date) == 8 and scan_date.isdigit():
                scan_date = f"{scan_date[:4]}-{scan_date[4:6]}-{scan_date[6:]}"
            break
    
    return subject_id, scan_date


def create_imgage_data(input_images_dir: str, output_csv_path: str) -> None:
    """
    Create a CSV file containing metadata for all NIfTI images.
    
    Args:
        input_images_dir: Directory containing NIfTI files
        output_csv_path: Path to save the output CSV file
    """
    input_images = [f for f in os.listdir(input_images_dir) 
                     if f.endswith('.nii.gz') or f.endswith('.nii')]
    
    # Initialize lists to store data
    data = {
        'file_name': [],
        'patient_id': [],
        'scan_date': []
    }
    
    # Process each image
    for image_file in input_images:
        # Load NIfTI file
        nifti_path = os.path.join(input_images_dir, image_file)
        nifti_img = nib.load(nifti_path)
        
        # Get subject ID and date
        subject_id, scan_date = get_subject_and_date(nifti_img, image_file)
        
        # Add to data dictionary
        data['file_name'].append(image_file)
        data['patient_id'].append(subject_id)
        data['scan_date'].append(scan_date)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by patient_id and scan_date
    df = df.sort_values(['patient_id', 'scan_date'])
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    
    print(f"Created metadata CSV file with {len(df)} entries")
    print(f"Number of unique patients: {df['patient_id'].nunique()}")



def find_closest_neighbors(features_dir: str, image_data_csv: str, n_neighbors: int = 10, 
                           standardize: bool = False, features_group: str = "All") -> None:
    """
    Find the closest neighbors for each image in the dataset.
    
    Args:
        features_dir: Directory containing feature files
        image_data_csv: Path to the image data CSV file
        n_neighbors: Number of neighbors to find
    """
    df = pd.read_csv(image_data_csv)

    

    # ---- Inputs you provide ----
    # E: (N, D) float32 array of embeddings (one row per image)
    # patient_id: (N,) array-like of patient IDs (str or int)
    # study_date:  (N,) array-like of YYYY-MM-DD strings (or date objects)
    features_list = []
    patient_id = np.array(df['patient_id'])
    study_date = np.array(df['scan_date']) 
    file_names = np.array(df['file_name'])
    for file_name in file_names:    
        file_path = os.path.join(features_dir, file_name.split(".")[0]+".csv")
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if features_group == "Shape":
            numeric_cols = [col for col in numeric_cols if col in SHAPE_FEATURES]
        elif features_group == "Texture":
            numeric_cols = [col for col in numeric_cols if col in TEXTURE_FEATURES]
        elif features_group == "All":
            pass
        else:
            raise ValueError(f"Invalid features group: {features_group}")
        
        features_list.append(df[numeric_cols].values[0, :])

    features = np.array(features_list)
    if standardize:
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        
        # Use a small epsilon instead of 1
        eps = 1e-8
        std = np.maximum(std, eps)
        features = (features - mean) / std


    # import pdb; pdb.set_trace()
    # ---- Fit NN index (L2 distance) ----
    nn = NearestNeighbors(n_neighbors=11, metric="euclidean")  # 1 extra for self
    nn.fit(features)
    dists, idxs = nn.kneighbors(features)  # idxs[i] lists neighbors for query i (self at 0)

    # ---- Helper: filter neighbors to enforce "no same patient on same day" & no self ----
    topk = n_neighbors
    top1_hits = 0
    top10_hits = 0

    # Track patient-level hits: whether any image for a patient gets a hit
    patients = np.unique(patient_id)
    patient_hit_top1 = defaultdict(lambda: False)
    patient_hit_top10 = defaultdict(lambda: False)

    for i in range(features.shape[0]):
        q_pid = patient_id[i]
        q_date = study_date[i]

        # drop self (idxs[i][0]) and any neighbor from same patient on same day
        neighbors = []
        for j in idxs[i][1:]:  # skip self
            if not (patient_id[j] == q_pid and study_date[j] == q_date):
                neighbors.append(j)
            if len(neighbors) >= topk:  # we only need up to top-10
                break

        # ---- R@1 ----
        if len(neighbors) >= 1 and patient_id[neighbors[0]] == q_pid:
            top1_hits += 1
            patient_hit_top1[q_pid] = True

        # ---- R@10 ----
        if any(patient_id[j] == q_pid for j in neighbors[:topk]):
            top10_hits += 1
            patient_hit_top10[q_pid] = True

    # ---- Image-level metrics ----
    r_at_1_img  = 100.0 * top1_hits  / features.shape[0]
    r_at_10_img = 100.0 * top10_hits / features.shape[0]

    # ---- Patient-level metrics ----
    r_at_1_patient  = 100.0 * sum(patient_hit_top1[p]  for p in patients) / len(patients)
    r_at_10_patient = 100.0 * sum(patient_hit_top10[p] for p in patients) / len(patients)

    print(f"\nAnalysis results:\n{'='*50}")
    print(f"Number of images: {features.shape[0]}")
    print(f"Number of unique patients: {len(set(patient_id))}")
    print(f"Number of unique dates: {len(set(study_date))}")

    # Create metrics as a single row
    
    values = [
        f"{features_group}",
        "Yes" if standardize else "No",
        f"{features.shape[1]}",
        f"{r_at_1_img:.1f}%",
        f"{r_at_10_img:.1f}%",
        f"{r_at_1_patient:.1f}%",
        f"{r_at_10_patient:.1f}%"
    ]
    
    # Create table with a single row
    return values
  
def print_results(results: List[List[str]], headers: List[str]) -> None:
    table = tabulate(results, headers=headers, tablefmt="fancy_grid")
    print(table)

