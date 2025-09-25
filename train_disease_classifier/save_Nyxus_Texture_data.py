
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#from reidentification_utils import get_subject_and_date, SHAPE_FEATURES, TEXTURE_FEATURES
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

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

# def save_Nyxus_Texture_data(features_dir: str, image_dir: str, info_csv: str,
#                                    features_group: str = "All", 
#                                    val_size: float = 0.2, classes: list[str] = ['CN', 'MCI', 'AD']):
#     """
#     Prepare features with proper 3-way split to ensure fair validation/test comparison.
#     """
#     # ... existing feature loading code stays the same ...
#     metadata = {
#         'patient_id': [],
#         'label': []
#     }
#     for features_group in TEXTURE_FEATURES:
#         metadata[features_group] = []

#     features_list = []
#     file_names = os.listdir(features_dir)
#     adni_info_df = pd.read_excel(info_csv, engine="openpyxl")

#     for file_name in file_names:
#         if file_name.endswith(".csv"):
#             file_path = os.path.join(features_dir, file_name)
#             image_path = os.path.join(image_dir, file_name.replace(".csv", ".nii.gz"))
            
#             # Extract subject ID from filename (not from NIfTI header)
#             subject_id, scan_date = get_subject_and_date(image_path)
            
#             label = adni_info_df[(adni_info_df['Subject'] == subject_id)]['Group'].values[0]
#             if label in classes:
#                 metadata['label'].append(label)
#                 metadata['patient_id'].append(subject_id)
    

#                 # load features
#                 feat_df = pd.read_csv(file_path,
#                                     engine='python',  # More robust parsing
#                                     encoding='utf-8', sep=',', usecols=lambda x: x != 'Unnamed: 0')  
                
#                 # To remove specific columns, you can do:
#                 columns_to_remove = ['intensity_image', 'mask_image', 'ROI_label','Unnamed: 0', 'index', 'id']
#                 numeric_cols = [col for col in feat_df.select_dtypes(include=[np.number]).columns 
#                                 if col not in columns_to_remove] 
        
#                 numeric_cols = [col for col in numeric_cols if col in TEXTURE_FEATURES]

#                 for feature in  TEXTURE_FEATURES:
#                     metadata[feature].append(feat_df[feature].values[0])

#     metadata_df = pd.DataFrame(metadata)
#     print(metadata_df.head())
#     metadata_df.to_csv("./data/Nyxus_Texture_metadata.csv", index=False)

def patient_aware_split(X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray, 
                                random_state, test_size: float = 0.2, val_size: float = 0.2, 
                                remove_constant: bool = True
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/validation/test ensuring no patient appears in multiple sets.
    This prevents the issue where validation and test sets have different difficulty levels.
    
    Args:
        X: Feature matrix
        y: Labels  
        patient_ids: Array of patient IDs corresponding to each sample
        test_size: Proportion of patients to include in test set
        val_size: Proportion of patients to include in validation set
        random_state: Random state for reproducibility
        remove_constant: Whether to remove constant features after scaling
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, patient_ids_train, patient_ids_val, patient_ids_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Get unique patients and their labels for stratified splitting
    unique_patients = np.unique(patient_ids)
    patient_labels = []
    for patient in unique_patients:
        # Get the most common label for this patient (for stratification only)
        patient_mask = patient_ids == patient
        patient_label_counts = np.bincount(y[patient_mask])
        most_common_label = np.argmax(patient_label_counts)
        patient_labels.append(most_common_label)
    
    # First split: separate test patients from train+val patients
    patients_trainval, patients_test = train_test_split(
        unique_patients, test_size=test_size, random_state=random_state, 
        stratify=patient_labels
    )
    
    # Get labels for remaining patients (train+val)
    trainval_labels = []
    for patient in patients_trainval:
        patient_mask = patient_ids == patient
        patient_label_counts = np.bincount(y[patient_mask])
        most_common_label = np.argmax(patient_label_counts)
        trainval_labels.append(most_common_label)
    
    # Second split: separate train and validation patients
    # Adjust val_size to account for the remaining patients
    adjusted_val_size = val_size / (1 - test_size)
    patients_train, patients_val = train_test_split(
        patients_trainval, test_size=adjusted_val_size, 
        random_state=random_state + 1,  # Different seed for second split
        stratify=trainval_labels
    )
    
    # Create masks for each split
    train_mask = np.isin(patient_ids, patients_train)
    val_mask = np.isin(patient_ids, patients_val) 
    test_mask = np.isin(patient_ids, patients_test)
    
    # Split the data - each image keeps its original label
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    
    # Get patient IDs for each image in each set
    patient_ids_train = patient_ids[train_mask]
    patient_ids_val = patient_ids[val_mask]
    patient_ids_test = patient_ids[test_mask]
    
    # Scale features using only training data
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Remove constant features after scaling
    if remove_constant:
        # Fit on training, transform all sets
        variance_selector = VarianceThreshold(threshold=0.0)
        X_train = variance_selector.fit_transform(X_train)
        X_val = variance_selector.transform(X_val)
        X_test = variance_selector.transform(X_test)
        
        selected_features_mask = variance_selector.get_support()
        n_removed = np.sum(~selected_features_mask)
        np.save(f"models/selected_features_mask_{random_state}.npy", selected_features_mask)
        print(f"Removed {n_removed} constant features")

    # Print split statistics
    print(f"Patient-level 3-way split:")
    print(f"  Train: {len(patients_train)} patients ({len(X_train)} images)")
    print(f"  Val:   {len(patients_val)} patients ({len(X_val)} images)")  
    print(f"  Test:  {len(patients_test)} patients ({len(X_test)} images)")
    
    # Check class distributions
    print(f"Class distributions:")
    for i, split_name in enumerate(['Train', 'Val', 'Test']):
        y_split = [y_train, y_val, y_test][i]
        class_counts = np.bincount(y_split)
        class_props = class_counts / len(y_split)
        print(f"  {split_name}: {class_counts} -> {class_props}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, patient_ids_train, patient_ids_val, patient_ids_test

if __name__ == "__main__":
        # Paths to your data
    # image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    # features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
    # info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"

    # save_Nyxus_Texture_data(features_dir, image_dir, info_csv, features_group="Texture")


    data_df = pd.read_csv("./data/Nyxus_Texture_metadata.csv")
    images_per_patient = data_df.groupby("patient_id").size()
    # Calculate statistics
    print("📊 Images per Patient Statistics:")
    print(f"Average images per patient: {images_per_patient.mean():.2f}")
    print(f"Minimum images per patient: {images_per_patient.min()}")
    print(f"Maximum images per patient: {images_per_patient.max()}")
    print(f"Median images per patient: {images_per_patient.median():.2f}")
    print(f"Standard deviation: {images_per_patient.std():.2f}")
    average_images = images_per_patient.mean()
    print(f"Average images per patient: {average_images}")
    X = data_df[TEXTURE_FEATURES].values
    y = data_df["label"].values
    patient_ids = data_df["patient_id"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)


    X_train, X_val, X_test, y_train, y_val, y_test, patient_ids_train, patient_ids_val, patient_ids_test = patient_aware_split(
        X, y, patient_ids, random_state=42, test_size=0.2, val_size=0.2
    )
 

    




