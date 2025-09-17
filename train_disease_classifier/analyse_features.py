import numpy as np

from calssifier_utils import prepare_features_Nyxus, prepare_features_BrainAIC

def analyze_data_quality(features_dir, image_dir, info_csv, features_group, test_size, classes):
    """
    Comprehensive data quality analysis to identify potential issues.
    """
    print("\n" + "="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    
    # 1. Basic Dataset Statistics
    print(f"\n1. DATASET OVERVIEW:")
    X_train, X_test, y_train, y_test, class_names, patient_ids_train, patient_ids_test = prepare_features_Nyxus(
        features_dir, image_dir, info_csv, 
        features_group=features_group,
        test_size=test_size,
        classes=classes
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Unique patients (train): {len(np.unique(patient_ids_train))}")
    print(f"   Unique patients (test): {len(np.unique(patient_ids_test))}")
    
    # 2. Class Distribution Analysis
    print(f"\n2. CLASS DISTRIBUTION:")
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    
    for i, class_name in enumerate(class_names):
        train_pct = train_counts[i] / len(y_train) * 100
        test_pct = test_counts[i] / len(y_test) * 100
        print(f"   {class_name}: Train={train_counts[i]} ({train_pct:.1f}%), Test={test_counts[i]} ({test_pct:.1f}%)")
    
    # Check for severe class imbalance
    min_class = np.min(train_counts)
    max_class = np.max(train_counts)
    imbalance_ratio = max_class / min_class
    print(f"   Imbalance ratio: {imbalance_ratio:.2f} (>5.0 is problematic)")
    
    # 3. Feature Quality Analysis
    print(f"\n3. FEATURE QUALITY:")
    
    # Check for constant features
    constant_features = np.sum(np.var(X_train, axis=0) == 0)
    print(f"   Constant features: {constant_features}")
    
    # Check for near-zero variance features
    low_variance_threshold = 0.01
    low_variance_features = np.sum(np.var(X_train, axis=0) < low_variance_threshold)
    print(f"   Low variance features (<{low_variance_threshold}): {low_variance_features}")
    
    # Check for missing values
    missing_values = np.sum(np.isnan(X_train))
    print(f"   Missing values: {missing_values}")
    
    # Check for infinite values
    infinite_values = np.sum(np.isinf(X_train))
    print(f"   Infinite values: {infinite_values}")
    
    # Feature scaling analysis
    feature_means = np.mean(X_train, axis=0)
    feature_stds = np.std(X_train, axis=0)
    
    print(f"   Feature range - Min: {np.min(X_train):.3f}, Max: {np.max(X_train):.3f}")
    print(f"   Mean feature std: {np.mean(feature_stds):.3f}")
    print(f"   Features with std > 1000: {np.sum(feature_stds > 1000)}")
    print(f"   Features with std < 0.001: {np.sum(feature_stds < 0.001)}")
    
    # 4. Correlation Analysis
    print(f"\n4. FEATURE CORRELATION:")
    correlation_matrix = np.corrcoef(X_train.T)
    
    # Find highly correlated features
    high_corr_threshold = 0.95
    high_corr_pairs = 0
    for i in range(len(correlation_matrix)):
        for j in range(i+1, len(correlation_matrix)):
            if abs(correlation_matrix[i, j]) > high_corr_threshold:
                high_corr_pairs += 1
    
    print(f"   Highly correlated feature pairs (>{high_corr_threshold}): {high_corr_pairs}")
    
    # 5. Separability Analysis
    print(f"\n5. CLASS SEPARABILITY:")
    
    # Simple separability test using class means
    class_means = []
    class_stds = []
    
    for i, class_name in enumerate(class_names):
        class_mask = y_train == i
        if np.any(class_mask):
            class_data = X_train[class_mask]
            class_means.append(np.mean(class_data, axis=0))
            class_stds.append(np.std(class_data, axis=0))
    
    # Calculate separability score (simplified)
    if len(class_means) >= 2:
        mean_distances = []
        for i in range(len(class_means)):
            for j in range(i+1, len(class_means)):
                distance = np.linalg.norm(class_means[i] - class_means[j])
                mean_distances.append(distance)
        
        avg_separation = np.mean(mean_distances)
        avg_within_class_std = np.mean([np.mean(std) for std in class_stds])
        separability_ratio = avg_separation / avg_within_class_std if avg_within_class_std > 0 else 0
        
        print(f"   Average class separation: {avg_separation:.3f}")
        print(f"   Average within-class std: {avg_within_class_std:.3f}")
        print(f"   Separability ratio: {separability_ratio:.3f} (>2.0 is good)")
    
    # 6. Data Leakage Check
    print(f"\n6. DATA LEAKAGE CHECK:")
    
    # Check for patient overlap between train/test
    patient_overlap = set(patient_ids_train) & set(patient_ids_test)
    print(f"   Patient overlap between train/test: {len(patient_overlap)}")
    
    # Check for duplicate samples
    train_duplicates = len(X_train) - len(np.unique(X_train, axis=0))
    print(f"   Duplicate samples in training: {train_duplicates}")
    
    # 7. Feature Importance Preview (Random Forest)
    print(f"\n7. QUICK FEATURE IMPORTANCE CHECK:")
    
    # Quick RF to check if any features are useful
    from sklearn.ensemble import RandomForestClassifier
    quick_rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    quick_rf.fit(X_train, y_train)
    
    # Get top 5 most important features
    importances = quick_rf.feature_importances_
    top_indices = np.argsort(importances)[-5:][::-1]
    
    print(f"   Top 5 feature importances:")
    for i, idx in enumerate(top_indices):
        print(f"     Feature {idx}: {importances[idx]:.4f}")
    
    max_importance = np.max(importances)
    print(f"   Max feature importance: {max_importance:.4f} (>0.1 is good)")
    
    # Quick accuracy test
    quick_accuracy = quick_rf.score(X_test, y_test)
    print(f"   Quick RF accuracy: {quick_accuracy:.3f}")
    
    return {
        'imbalance_ratio': imbalance_ratio,
        'separability_ratio': separability_ratio,
        'max_importance': max_importance,
        'quick_accuracy': quick_accuracy,
        'constant_features': constant_features,
        'low_variance_features': low_variance_features
    }

def diagnose_data_problems(analysis_results):
    """
    Diagnose potential data problems based on analysis results.
    """
    print("\n" + "="*60)
    print("DATA PROBLEM DIAGNOSIS")
    print("="*60)
    
    problems = []
    suggestions = []
    
    # Check class imbalance
    if analysis_results['imbalance_ratio'] > 5.0:
        problems.append("❌ Severe class imbalance detected")
        suggestions.append("   → Use SMOTE, BorderlineSMOTE, or class_weight='balanced'")
    else:
        print("✅ Class distribution is reasonable")
    
    # Check feature quality
    if analysis_results['constant_features'] > 0:
        problems.append("❌ Constant features detected")
        suggestions.append("   → Remove constant features using VarianceThreshold")
    
    if analysis_results['low_variance_features'] > analysis_results.get('total_features', 100) * 0.5:
        problems.append("❌ Too many low-variance features")
        suggestions.append("   → Use feature selection (SelectKBest, RFECV)")
    
    # Check separability
    if analysis_results['separability_ratio'] < 1.0:
        problems.append("❌ Poor class separability - classes overlap significantly")
        suggestions.append("   → Try feature engineering, PCA, or different feature types")
        suggestions.append("   → Consider if this is inherently a difficult classification problem")
    elif analysis_results['separability_ratio'] < 2.0:
        problems.append("⚠️  Moderate class separability - challenging but doable")
        suggestions.append("   → Use ensemble methods and careful feature selection")
    else:
        print("✅ Good class separability")
    
    # Check feature importance
    if analysis_results['max_importance'] < 0.05:
        problems.append("❌ Very low feature importance - features may not be informative")
        suggestions.append("   → Try different feature extraction methods")
        suggestions.append("   → Consider domain-specific features")
        suggestions.append("   → Check if preprocessing is removing important information")
    elif analysis_results['max_importance'] < 0.1:
        problems.append("⚠️  Low feature importance - limited signal in features")
        suggestions.append("   → Combine with other feature types (clinical, genetic)")
    else:
        print("✅ Features show reasonable importance")
    
    # Check overall performance
    if analysis_results['quick_accuracy'] < 0.4:
        problems.append("❌ Very poor baseline accuracy - likely data quality issue")
        suggestions.append("   → Check data preprocessing pipeline")
        suggestions.append("   → Verify labels are correct")
        suggestions.append("   → Consider if task is too difficult with current features")
    elif analysis_results['quick_accuracy'] < 0.6:
        problems.append("⚠️  Low baseline accuracy - room for improvement")
        suggestions.append("   → Try ensemble methods and better hyperparameter tuning")
    else:
        print("✅ Reasonable baseline accuracy")
    
    # Print problems and suggestions
    if problems:
        print(f"\n🚨 IDENTIFIED PROBLEMS:")
        for problem in problems:
            print(f"  {problem}")
        
        print(f"\n💡 SUGGESTIONS:")
        for suggestion in suggestions:
            print(f"  {suggestion}")
    else:
        print("\n✅ No major data quality issues detected!")
    
    return len(problems) == 0

# Add this to your training function

if __name__ == "__main__":
    features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
    image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    features_group = "All"
    test_size = 0.2
    classes = ['CN', 'MCI', 'AD']
    analysis_results = analyze_data_quality(features_dir, image_dir, info_csv, features_group, test_size, classes)
    diagnose_data_problems(analysis_results)