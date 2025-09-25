import os
import pickle
from tkinter import N
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


def patient_aware_split(X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray, 
                                random_state, test_size: float = 0.2, val_size: float = 0.2, 
                                remove_constant: bool = True, robust_scale: bool = False,
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

    # Get unique patients and their labels for stratified splitting
    unique_patients = np.unique(patient_ids)
    
    # First split: separate test patients from train+val patients
    patients_trainval, patients_test = train_test_split(
        unique_patients, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate train and validation patients
    # Adjust val_size to account for the remaining patients
    adjusted_val_size = val_size / (1 - test_size)
    patients_train, patients_val = train_test_split(
        patients_trainval, test_size=adjusted_val_size, 
        random_state=random_state + 1 # Different seed for second split
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
    
    if robust_scale:
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

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

    # Print split statistics
    print(f"Patient-level 3-way split:")
    print(f"  Train: {len(patients_train)} patients ({len(X_train)} images)")
    print(f"  Val:   {len(patients_val)} patients ({len(X_val)} images)")  
    print(f"  Test:  {len(patients_test)} patients ({len(X_test)} images)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, patient_ids_train, patient_ids_val, patient_ids_test

def prepare_data(metadata_csv, Nyxus_whole_slide_features_pkl, random_state=42, test_size=0.2, val_size=0.2):


    metadata_df = pd.read_excel(metadata_csv, header=[0,1])
    # # Flatten column names
    metadata_df.columns = ['_'.join([str(c1), str(c2)]) if 'Unnamed' not in str(c2) else str(c1)
                for c1, c2 in metadata_df.columns]

    metadata_df = metadata_df[metadata_df['OCT RNFL thickness_Mean'] != '/']

    print("metadata_df.shape", metadata_df.shape)

    with open(Nyxus_whole_slide_features_pkl, 'rb') as f:
        features_dict = pickle.load(f)

    data = {'features': [], 'RNFL_thickness_Mean': [], 'patient_id': []}
    for item in features_dict:
        #import pdb; pdb.set_trace()
        result_df = metadata_df[metadata_df['Corresponding CFP'] == item['path']]
        if len(result_df) != 0:
            data['features'].append(item['features'])
            data['RNFL_thickness_Mean'].append(result_df['OCT RNFL thickness_Mean'].values[0])
            data['patient_id'].append(result_df['Subject Number'].values[0])

    X = np.array(data['features'])
    y = np.array(data['RNFL_thickness_Mean']).astype(float)
    patient_ids = np.array(data['patient_id'])

    print(X.shape, y.shape, patient_ids.shape)

    X_train, X_val, X_test, y_train, y_val, y_test, patient_ids_train, patient_ids_val, patient_ids_test = patient_aware_split(
        X, y, patient_ids, random_state=random_state, test_size=test_size, val_size=val_size
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, patient_ids_train, patient_ids_val, patient_ids_test


def train_ridge_regressor(X_train, y_train):
    model = Ridge(alpha=0.1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    y_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print(f"Train MSE: {mse_train}, Train R2: {r2_train}")
    print(f"Test MSE: {mse_test}, Test R2: {r2_test}")


def search_best_random_forest_regressor(X_train, y_train, random_state=42):
    param_grid = {
        'n_estimators': [100, 200],              # Fewer trees
        'max_depth': [3, 5, 8],                  # MUCH shallower trees
        'min_samples_split': [20, 50, 100],      # MUCH higher - require more samples to split
        'min_samples_leaf': [10, 20, 30],        # MUCH higher - larger leaf nodes
        'max_features': ['sqrt'],                # Most restrictive feature sampling
        'bootstrap': [True],
        'max_samples': [0.7, 0.8],              # Use less data per tree
        'random_state': [random_state]
    }
    
    search = RandomizedSearchCV(
        RandomForestRegressor(n_jobs=-1),
        param_grid,
        n_iter=20,  # Fewer iterations
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1,
        random_state=random_state,
        cv=5,
        return_train_score=True  # Monitor train vs validation gap
    )
    
    search.fit(X_train, y_train)
    
    return search.best_params_

def select_features(X_train, y_train, X_test, k=100):
    """Very conservative feature selection"""
    print(f"🔧 Conservative feature selection: top {k} features...")
    
    # Remove constant features
    variance_selector = VarianceThreshold(threshold=0.05)  # Higher threshold
    X_train_var = variance_selector.fit_transform(X_train)
    X_test_var = variance_selector.transform(X_test)
    
    print(f"After variance threshold (0.05): {X_train_var.shape[1]} features")
    
    # Select fewer features
    if X_train_var.shape[1] > k:
        feature_selector = SelectKBest(score_func=f_regression, k=k)
        X_train_selected = feature_selector.fit_transform(X_train_var, y_train)
        X_test_selected = feature_selector.transform(X_test_var)
        print(f"After feature selection: {X_train_selected.shape[1]} features")
    else:
        X_train_selected = X_train_var
        X_test_selected = X_test_var
    
    return X_train_selected, X_test_selected

def train_rf_regressor(X_train, y_train, X_test, y_test, random_state=42):
    """Train Random Forest with aggressive anti-overfitting measures"""
    print("🛡️ Training ANTI-OVERFITTING Random Forest...")
    
    # Use very few features
    X_train_selected, X_test_selected = select_features(
        X_train, y_train, X_test, k=1000  # Very few features
    )
    
    # Get conservative parameters
    best_params = search_best_random_forest_regressor(X_train_selected, y_train, random_state)
    
    # Train final model
    model = RandomForestRegressor(**best_params, n_jobs=-1)
    model.fit(X_train_selected, y_train)
    
    # Evaluate on both train and test
    train_pred = model.predict(X_train_selected)
    test_pred = model.predict(X_test_selected)
    
    train_mse = mean_squared_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    overfitting_gap = train_r2 - test_r2
    
    print(f"\n📊 ANTI-OVERFITTING RESULTS:")
    print(f"Train MSE: {train_mse:.2f}, Train R²: {train_r2:.4f}")
    print(f"Test MSE: {test_mse:.2f}, Test R²: {test_r2:.4f}")
    print(f"Overfitting gap: {overfitting_gap:.4f}")
    
    if overfitting_gap < 0.2:
        print("✅ Good! Overfitting gap is acceptable (<0.2)")
    elif overfitting_gap < 0.4:
        print("⚠️ Moderate overfitting gap (0.2-0.4)")
    else:
        print("❌ Still overfitting! Gap > 0.4")
    
    return model, {
        'train_mse': train_mse, 'train_r2': train_r2,
        'test_mse': test_mse, 'test_r2': test_r2,
        'overfitting_gap': overfitting_gap
    }


def main(metadata_csv, features_pkl, random_state=42, test_size=0.2, val_size=0.2):
    print("🚀 Starting ANTI-OVERFITTING Random Forest Training...")
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, patient_ids_train, patient_ids_val, patient_ids_test = prepare_data(
        metadata_csv, features_pkl, random_state=random_state, test_size=test_size, val_size=val_size)
    
    # Combine train and validation
    X_train = np.concatenate((X_train, X_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)
    patient_ids_train = np.concatenate((patient_ids_train, patient_ids_val), axis=0)
    
    print("📊 Dataset Summary:")
    print(f"Train: {X_train.shape[0]} samples, Features: {X_train.shape[1]}")
    print(f"Test: {X_test.shape[0]} samples")
    
    # Compare regularization levels
    #reg_results = compare_regularization_levels(X_train, y_train, X_test, y_test, random_state)
    
    # Train final anti-overfitting model
    model, results = train_rf_regressor(X_train, y_train, X_test, y_test, random_state)

    # evaluate_model(model, X_train, X_test, y_train, y_test)
    
    return model, results


if __name__ == "__main__":
    metadata_csv = "/home/ubuntu/data/GRAPE/VF_and_clinical_information.xlsx"

    Nyxus_whole_slide_features_pkl = "/home/ubuntu/projects/re_identification/my_features/GRAPE_Nyxus_WHOLESLIDE/PKL_file/WHOLESLIDE_2D_features.pkl"
    Nyxus_all_features_pkl = "/home/ubuntu/projects/re_identification/my_features/GRAPE_Nyxus_ALL/PKL_file/ALL_2D_features.pkl"
    RetFound_features_pkl = "/home/ubuntu/projects/re_identification/my_features/GRAPE_RETFound/features_0.pkl"

    main(metadata_csv, Nyxus_whole_slide_features_pkl, random_state=42, test_size=0.2, val_size=0.2)


