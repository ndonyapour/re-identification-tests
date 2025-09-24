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
from sklearn.preprocessing import StandardScaler
from pathlib import Path

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


def get_subject_and_date(filename: str) -> tuple[str, str]:
    """
    Get subject ID and scan date from a NIfTI file.
    
    Args:
        filename: Filename to extract information from (ADNI format expected)
        
    Returns:
        tuple: (subject_id, scan_date) where either could be 'unknown' if not found
    """
    subject_id = 'unknown'
    scan_date = 'unknown'
    
    # ADNI filename pattern: ADNI_035_S_0048_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080206161657886_S44467_I89626.nii
    # Subject ID is in format: XXX_S_XXXX (e.g., 035_S_0048)
    # Date is in format: YYYYMMDD (e.g., 20080206)
    
    # Extract ADNI subject ID pattern (XXX_S_XXXX)
    subject_pattern = r'ADNI_(\d{3}_S_\d{4})'
    subject_match = re.search(subject_pattern, filename)
    if subject_match:
        subject_id = subject_match.group(1)
    
    # Extract date pattern (YYYYMMDD) - look for 8 digits that represent a valid date
    date_pattern = r'_(\d{8})\d+'  # 8 digits followed by more digits (timestamp)
    date_match = re.search(date_pattern, filename)
    if date_match:
        date_str = date_match.group(1)
        # Validate it's a reasonable date (starts with 19xx or 20xx)
        if date_str.startswith(('19', '20')):
            # Convert YYYYMMDD to YYYY-MM-DD format
            scan_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    
    return subject_id, scan_date


def find_closest_neighbors(features: np.ndarray, 
                            ADNI_metadata: pd.DataFrame, 
                            n_neighbors: int = 10, 
                            exclude_same_date: bool = False, 
                            distance_threshold: float = -1.0,
                            features_name: str = "All",
                            standardize: bool = False,
                            output_dir: str = None) -> np.ndarray:
    """
    Find the closest neighbors for each image in the dataset.
    
    Args:
        features: The features to find neighbors for
        ADNI_metadata_path: Path to the ADNI metadata CSV file
        n_neighbors: Number of neighbors to find
        exclude_same_date: Whether to exclude matches from same date for same patient
        distance_threshold: Only keep matches with distance <= threshold (-1 to disable)
        output_dir: Directory to save matches.csv (if None, don't save)
    """
    patient_id = ADNI_metadata['patient_id'].values
    study_date = ADNI_metadata['scan_date'].values
    file_names = ADNI_metadata['file_name'].values
    
    if standardize:
        features = StandardScaler().fit_transform(features)
    
    nn = NearestNeighbors(n_neighbors=n_neighbors+1, metric="euclidean")  # +1 for self
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

     # Track matches
    matches = []
    
    # Track metrics
    top1_hits = 0
    top10_hits = 0
    patients = np.unique(patient_id)
    patient_hit_top1 = defaultdict(lambda: False)
    patient_hit_top10 = defaultdict(lambda: False)

    for i in range(len(features)):
        q_pid = patient_id[i]
        q_date = study_date[i]
        
        # Filter neighbors
        valid_neighbors = []
        valid_distances = []
        
        for j, (neighbor_idx, dist) in enumerate(zip(indices[i][1:], distances[i][1:])):  # skip self
            n_pid = patient_id[neighbor_idx]
            n_date = study_date[neighbor_idx]
            
            # Skip if same patient on same date when exclude_same_date is True
            if exclude_same_date and n_pid == q_pid and n_date == q_date:
                continue
                
            # Skip if distance above threshold
            if distance_threshold > -1 and dist > distance_threshold:
                continue
                
            valid_neighbors.append(neighbor_idx)
            valid_distances.append(dist)
            
            if len(valid_neighbors) >= n_neighbors:
                break
        
        if not valid_neighbors:
            continue
            
        # Record match
        matches.append({
            'query': file_names[i],
            'result': file_names[valid_neighbors[0]],
            'n_results': len(valid_neighbors),
            'top_distance': valid_distances[0]
        })
        
        # Update metrics
        if patient_id[valid_neighbors[0]] == q_pid:
            top1_hits += 1
            patient_hit_top1[q_pid] = True
            
        if any(patient_id[j] == q_pid for j in valid_neighbors[:10]):
            top10_hits += 1
            patient_hit_top10[q_pid] = True
    
    # Save matches if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        matches_df = pd.DataFrame(matches)
        filename = 'matches_diff_dates.csv' if exclude_same_date else 'matches.csv'
        matches_df.to_csv(os.path.join(output_dir, filename), index=False)
    
    # Calculate metrics
    n_total = len(features)
    n_patients = len(patients)
    
    r_at_1_img = 100.0 * top1_hits / n_total
    r_at_10_img = 100.0 * top10_hits / n_total
    r_at_1_patient = 100.0 * sum(patient_hit_top1[p] for p in patients) / n_patients
    r_at_10_patient = 100.0 * sum(patient_hit_top10[p] for p in patients) / n_patients
    
    print(f"\nAnalysis results:\n{'='*50}")
    print(f"Number of images: {n_total}")
    print(f"Number of unique patients: {n_patients}")
    print(f"Number of unique dates: {len(set(study_date))}")
    print(f"\nImage-level metrics:")
    print(f"R@1: {r_at_1_img:.1f}%")
    print(f"R@10: {r_at_10_img:.1f}%")
    print(f"\nPatient-level metrics:")
    print(f"R@1: {r_at_1_patient:.1f}%")
    print(f"R@10: {r_at_10_patient:.1f}%")
    
    # Return metrics for table
    return [
        features_name,
        "Yes" if standardize else "No",
        f"{features.shape[1]}",
        f"{r_at_1_img:.1f}%",
        f"{r_at_10_img:.1f}%",
        f"{r_at_1_patient:.1f}%",
        f"{r_at_10_patient:.1f}%"
    ]



def find_closest_neighbors_BrainAIC(features_csv_path: str, 
                                    info_csv_path: str,
                                    n_neighbors: int = 10, 
                                    standardize: bool = False,
                                    exclude_same_date: bool = False, 
                                    distance_threshold: float = -1.0,
                                    output_dir: str | None = None) -> None:
    """
    Find the closest neighbors for each image in the dataset.
    
    Args:
        features_csv_path: Path to the features CSV file
        ADNI_metadata_csv: Path to the ADNI metadata CSV file
        n_neighbors: Number of neighbors to find

        standardize: Whether to standardize features
        features_group: Which features to use ("All", "Shape", or "Texture")
        exclude_same_date: Whether to exclude matches from same date for same patient
        distance_threshold: Only keep matches with distance <= threshold (-1 to disable)
        output_dir: Directory to save matches.csv (if None, don't save)
    """

    # load features
    features_df = pd.read_csv(features_csv_path)

    metadata = {
        'file_name': [],
        'patient_id': [],
        'scan_date': []
    }
    features_list = []
    print(f'Loading features for {len(features_df)} images and creating metadata')
    for nifti_path in features_df['image_path']:

        nifti_img = nib.load(nifti_path)
        
        # Get subject ID and date
        subject_id, scan_date = get_subject_and_date(nifti_path)
        
        # Add to data dictionary
        metadata['file_name'].append(Path(nifti_path).name)
        metadata['patient_id'].append(subject_id)
        metadata['scan_date'].append(scan_date)
        features_cols = [f for f in features_df.columns if f.startswith('Feature_')]
        features_list.append(features_df[features_df['image_path'] == nifti_path][features_cols].values[0, :])

    features = np.array(features_list)

    ADNI_metadata = pd.DataFrame(metadata)
    rate_results = find_closest_neighbors(features, ADNI_metadata, n_neighbors, exclude_same_date, 
                                    distance_threshold, "BrainAIC", standardize, output_dir)
    ap_results = compute_precision_recall(os.path.join(output_dir, 'matches_diff_dates.csv'), info_csv_path, output_dir)

   # Combine both results into one dictionary
    combined_results = {
        # From rate_results list
        'features_name': rate_results[0],
        'standardized': rate_results[1],
        'n_features': rate_results[2],
        'r_at_1_img': float(rate_results[3].strip('%')),  # Convert "X.X%" to float
        'r_at_10_img': float(rate_results[4].strip('%')),
        'image_ap': ap_results['image_ap'],
        'r_at_1_patient': float(rate_results[5].strip('%')),
        'r_at_10_patient': float(rate_results[6].strip('%')),
        'patient_ap': ap_results['patient_ap'],
    }
    return combined_results


def find_closest_neighbors_Nyxus_topfea(features_dir: str, image_dir: str, info_csv: str, features_names: list[str],
                         n_neighbors: int = 10, standardize: bool = False, num_top: int = 10,
                         exclude_same_date: bool = False, distance_threshold: float = -1.0,
                         output_dir: str | None = None) -> None:
    """
    Find the closest neighbors for each image in the dataset.
    
    Args:
        features_dir: Directory containing feature files
        image_data_csv: Path to the image data CSV file
        n_neighbors: Number of neighbors to find
        standardize: Whether to standardize features
        features_group: Which features to use ("All", "Shape", or "Texture")
        exclude_same_date: Whether to exclude matches from same date for same patient
        distance_threshold: Only keep matches with distance <= threshold (-1 to disable)
        output_dir: Directory to save matches.csv (if None, don't save)
    """

    
    # load metadata
    metadata = {
        'file_name': [],
        'patient_id': [],
        'scan_date': []
    }
    features_list = []

    file_names = os.listdir(features_dir)
    
    for file_name in file_names:
        if file_name.endswith(".csv"):
            file_path = os.path.join(features_dir, file_name)
            image_path = os.path.join(image_dir, file_name.replace(".csv", ".nii.gz"))
            nifti_img = nib.load(image_path)
            subject_id, scan_date = get_subject_and_date(file_name.replace(".csv", ".nii.gz"))
            metadata['file_name'].append(file_name.replace(".csv", ".nii.gz"))
            metadata['patient_id'].append(subject_id)
            metadata['scan_date'].append(scan_date)

            # load features
            feat_df = pd.read_csv(file_path,
                                engine='python',  # More robust parsing
                                encoding='utf-8', sep=',', usecols=lambda x: x != 'Unnamed: 0')  
            
            # To remove specific columns, you can do:
            columns_to_remove = ['intensity_image', 'mask_image', 'ROI_label','Unnamed: 0', 'index', 'id']  # add any column names you want to remove
            numeric_cols = [col for col in feat_df.select_dtypes(include=[np.number]).columns 
                            if col not in columns_to_remove] 
            
            numeric_cols = [col for col in numeric_cols if col in features_names]
            features_list.append(feat_df[numeric_cols].values[0, :])
        
    metadata_df = pd.DataFrame(metadata)
    features = np.array(features_list)
    #add features to the metadata_df
    #import pdb; pdb.set_trace()
    rate_results = find_closest_neighbors(features, metadata_df, n_neighbors, exclude_same_date, 
                                    distance_threshold, "Nyxus " + str(num_top), standardize, output_dir)
    ap_results = compute_precision_recall(os.path.join(output_dir, 'matches_diff_dates.csv'), info_csv, output_dir)

    # Combine both results into one dictionary
    combined_results = {
        'features_name': num_top,
        'standardized': rate_results[1],
        'n_features': rate_results[2],
        'r_at_1_img': float(rate_results[3].strip('%')),  # Convert "X.X%" to float
        'r_at_10_img': float(rate_results[4].strip('%')),
        'image_ap': ap_results['image_ap'],
        'r_at_1_patient': float(rate_results[5].strip('%')),
        'r_at_10_patient': float(rate_results[6].strip('%')),
        'patient_ap': ap_results['patient_ap'],
    }
    return combined_results



def find_closest_neighbors_Nyxus(features_dir: str, image_dir: str, info_csv: str, n_neighbors: int = 10, 
                         standardize: bool = False, features_group: str = "All",
                         exclude_same_date: bool = False, distance_threshold: float = -1.0,
                         output_dir: str | None = None) -> None:
    """
    Find the closest neighbors for each image in the dataset.
    
    Args:
        features_dir: Directory containing feature files
        image_data_csv: Path to the image data CSV file
        n_neighbors: Number of neighbors to find
        standardize: Whether to standardize features
        features_group: Which features to use ("All", "Shape", or "Texture")
        exclude_same_date: Whether to exclude matches from same date for same patient
        distance_threshold: Only keep matches with distance <= threshold (-1 to disable)
        output_dir: Directory to save matches.csv (if None, don't save)
    """

    
    # load metadata
    metadata = {
        'file_name': [],
        'patient_id': [],
        'scan_date': []
    }
    features_list = []

    file_names = os.listdir(features_dir)
    
    for file_name in file_names:
        if file_name.endswith(".csv"):
            file_path = os.path.join(features_dir, file_name)
            image_path = os.path.join(image_dir, file_name.replace(".csv", ".nii.gz"))
            nifti_img = nib.load(image_path)
            subject_id, scan_date = get_subject_and_date(file_name.replace(".csv", ".nii.gz"))
            metadata['file_name'].append(file_name.replace(".csv", ".nii.gz"))
            metadata['patient_id'].append(subject_id)
            metadata['scan_date'].append(scan_date)

            # load features
            feat_df = pd.read_csv(file_path,
                                engine='python',  # More robust parsing
                                encoding='utf-8', sep=',', usecols=lambda x: x != 'Unnamed: 0')  
            
            # To remove specific columns, you can do:
            columns_to_remove = ['intensity_image', 'mask_image', 'ROI_label','Unnamed: 0', 'index', 'id']  # add any column names you want to remove
            numeric_cols = [col for col in feat_df.select_dtypes(include=[np.number]).columns 
                            if col not in columns_to_remove] 
            
            if features_group == "Shape":
                numeric_cols = [col for col in numeric_cols if col in SHAPE_FEATURES]
            elif features_group == "Texture":
                numeric_cols = [col for col in numeric_cols if col in TEXTURE_FEATURES]
            elif features_group == "All":
                pass
            else:
                raise ValueError(f"Invalid features group: {features_group}")
            features_list.append(feat_df[numeric_cols].values[0, :])
        
    metadata_df = pd.DataFrame(metadata)
    features = np.array(features_list)
    #add features to the metadata_df
    #import pdb; pdb.set_trace()
    rate_results = find_closest_neighbors(features, metadata_df, n_neighbors, exclude_same_date, 
                                    distance_threshold, "Nyxus " + features_group, standardize, output_dir)
    ap_results = compute_precision_recall(os.path.join(output_dir, 'matches_diff_dates.csv'), info_csv, output_dir)

    # Combine both results into one dictionary
    combined_results = {
        'features_name': rate_results[0],
        'standardized': rate_results[1],
        'n_features': rate_results[2],
        'r_at_1_img': float(rate_results[3].strip('%')),  # Convert "X.X%" to float
        'r_at_10_img': float(rate_results[4].strip('%')),
        'image_ap': ap_results['image_ap'],
        'r_at_1_patient': float(rate_results[5].strip('%')),
        'r_at_10_patient': float(rate_results[6].strip('%')),
        'patient_ap': ap_results['patient_ap'],
    }
    return combined_results
    

def print_results(results: List[List[str]], headers: List[str]) -> None:
    table = tabulate(results, headers=headers, tablefmt="fancy_grid")
    print(table)


def compute_precision_recall(matches_csv: str, info_csv: str, output_dir: str) -> dict:
    """
    Compute Precision/Recall metrics based on previous re-identification results.
    
    Args:
        matches_csv: Path to matches_diff_dates.csv with query/result pairs
        info_csv: Path to CSV with filename and patient ID mapping
        output_dir: Directory to save precision-recall plots
        
    Returns:
        dict: Dictionary containing AP and R@1 metrics for both image and patient levels
    """
    from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
    from matplotlib.pyplot import savefig
    import os
    import pandas as pd

    # Read patient info
    px_info = pd.read_csv(info_csv)
    if 'filename' not in px_info.columns or 'pat_id' not in px_info.columns:
        raise ValueError("Info CSV must contain 'filename' and 'pat_id' columns")

    # Read re-identification results
    results = pd.read_csv(matches_csv)
    print(f'Read results for {len(results):,d} images')

    # Merge images with patient ids
    results = pd.merge(results, px_info[['pat_id', 'filename']], 
                      left_on='query', right_on='filename')
    results = pd.merge(results, px_info[['pat_id', 'filename']], 
                      left_on='result', right_on='filename')
    results = results.rename(columns={
        'pat_id_x': 'query_px',
        'pat_id_y': 'result_px'
    })

    # Ground truth of whether the patient is correctly re-identified or not
    results['same_px'] = results['result_px'] == results['query_px']
    print(f'R@1 image level: {results.same_px.sum() / len(results):.2%}')

    # Change direction of score: it's a distance (higher = farther), we want higher = closer
   
    # d = results['top_distance'].to_numpy()
    # score = 1 - (d - d.min())/(d.max()-d.min()+1e-12)
    results['rescaled_score'] = results['top_distance'].max() - results['top_distance']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Image-level precision-recall curve
    _ = PrecisionRecallDisplay.from_predictions(
        results['same_px'], 
        results['rescaled_score'], 
        plot_chance_level=True
    )
    savefig(os.path.join(output_dir, 'precision_recall_diff_date.png'), dpi=300)
    
    image_ap = average_precision_score(results.same_px, results.rescaled_score)
    print(f'AP (image level): {image_ap:.2%}')

    # Group at patient level
    results_px = []
    for px_id, px_imgs in results.groupby('query_px'):
        curr_res = {'query_px': px_id}
        same_px = px_imgs[px_imgs.result_px == px_id]
        if len(same_px) == 0:
            same_px = px_imgs
        
        # Get most similar image
        matched = same_px.sort_values('rescaled_score', ascending=False)
        curr_res.update({
            'result_px': matched.iloc[0].result_px,
            'rescaled_score': matched.iloc[0].rescaled_score
        })
        results_px.append(curr_res)

    results_px = pd.DataFrame(results_px)
    results_px['same_px'] = results_px.result_px == results_px.query_px
    
    patient_r1 = results_px.same_px.sum() / len(results_px)
    print(f'R@1 patient level: {patient_r1:.2%} for {len(results_px):,d} patients')

    # Patient-level precision-recall curve
    _ = PrecisionRecallDisplay.from_predictions(
        results_px['same_px'],
        results_px['rescaled_score'],
        plot_chance_level=True
    )
    savefig(os.path.join(output_dir, 'precision_recall_diff_date_px_level.png'), dpi=300)

    patient_ap = average_precision_score(results_px.same_px, results_px.rescaled_score)
    print(f'AP (patient level): {patient_ap:.2%}')

    # Return metrics dictionary
    return {
        'image_ap': image_ap,
        'patient_ap': patient_ap
    }


