import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from calssifier_utils import prepare_features_Nyxus, prepare_features_BrainAIC



def get_feature_names(csv_file_path: str):
    """
    Get feature names from a CSV file.
    """

    # load features
    feat_df = pd.read_csv(csv_file_path,
                        engine='python',  # More robust parsing
                        encoding='utf-8', sep=',', usecols=lambda x: x != 'Unnamed: 0')  

    # To remove specific columns, you can do:
    columns_to_remove = ['intensity_image', 'mask_image', 'ROI_label','Unnamed: 0', 'index', 'id']
    numeric_cols = [col for col in feat_df.select_dtypes(include=[np.number]).columns 
                    if col not in columns_to_remove] 
    return numeric_cols

def get_rf_feature_ranking(rf_model, X_test, y_test, feature_names=None, top_n=20):
    """
    Simple function to get Random Forest feature importance ranking.
    
    Args:
        rf_model: Trained Random Forest model (or pipeline with RF)
        X_test: Test features for permutation importance
        y_test: Test labels
        feature_names: List of feature names (optional)
        top_n: Number of top features to return
        
    Returns:
        pd.DataFrame: Feature rankings with importance scores
    """
    
    # Handle pipeline vs direct RF model
    if hasattr(rf_model, 'named_steps'):
        rf_estimator = rf_model.named_steps['rf']
    else:
        rf_estimator = rf_model
    
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_test.shape: {y_test.shape}")
    # Get built-in feature importance
    rf_importance = rf_estimator.feature_importances_
    
    # Get permutation importance (most reliable)
    from sklearn.inspection import permutation_importance
    print(f"{X_test.shape}")
    perm_result = permutation_importance(rf_model, X_test, y_test, n_repeats=5, random_state=42)
    perm_importance = perm_result.importances_mean
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(rf_importance))]
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'feature': feature_names[:len(rf_importance)],
        'rf_importance': rf_importance,
        'permutation_importance': perm_importance,
        'combined_score': (rf_importance + perm_importance) / 2  # Average of both methods
    })
    
    # Sort by combined score
    results_df = results_df.sort_values('combined_score', ascending=False)
    results_df['rank'] = range(1, len(results_df) + 1)
    
    # Print top features
    print(f"\n🏆 TOP {min(top_n, len(results_df))} MOST IMPORTANT FEATURES:")
    print("-" * 70)
    print(f"{'Rank':<4} {'Feature':<30} {'RF Score':<12} {'Perm Score':<12} {'Combined':<10}")
    print("-" * 70)
    
    for _, row in results_df.head(top_n).iterrows():
        print(f"{int(row['rank']):<4} {row['feature']:<30} {row['rf_importance']:<12.6f} "
              f"{row['permutation_importance']:<12.6f} {row['combined_score']:<10.6f}")
    
    return results_df


def drop_column_importance(model, X_train, X_test, y_train, y_test):
    baseline_score = model.score(X_test, y_test)
    importances = []
    
    for i in range(X_train.shape[1]):
        X_train_dropped = np.delete(X_train, i, axis=1)
        X_test_dropped = np.delete(X_test, i, axis=1)
        
        model.fit(X_train_dropped, y_train)
        dropped_score = model.score(X_test_dropped, y_test)
        
        importance = baseline_score - dropped_score
        importances.append(importance)
    
    return importances

if __name__ == "__main__":
        # Paths to your data
    image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    feature_csv_path = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/ADNI_137_S_1414_MR_MPR__GradWarp__N3__Scaled_Br_20081017083139155_S56047_I121712_0000.csv"

    print("Getting feature names...")
    feature_names = get_feature_names(feature_csv_path)
    features_group = "All"
    random_state = 42
    np.random.seed(random_state)

    print("Loading RF model...")
    rf_model = joblib.load("./models/rf_model_Nyxus_All_42.pkl")
    print("Preparing features...")
    X_train, X_val, X_test, y_train, y_val, y_test, class_names, patient_ids_train, patient_ids_val, patient_ids_test = prepare_features_Nyxus(
            features_dir, image_dir, info_csv, 
            features_group=features_group, 
            random_state=random_state, 
            test_size=0.2,
            classes=['CN', 'MCI', 'AD']
        )
    X_train = np.concatenate((X_train, X_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)
    patient_ids_train = np.concatenate((patient_ids_train, patient_ids_val), axis=0)
    print("Selecting features...")
    selected_features_mask = np.load(f"models/selected_features_mask_{random_state}.npy")
    feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features_mask[i]]
    print("Getting RF feature ranking...")
    df = get_rf_feature_ranking(rf_model, X_test, y_test, feature_names, top_n=20)
    print("Getting drop column importance...")
    drop_column_importance = drop_column_importance(rf_model, X_train, X_test, y_train, y_test)
    df['drop_column_importance'] = drop_column_importance
    df.to_csv(f"models/rf_feature_ranking_{features_group}_{random_state}.csv", index=False)