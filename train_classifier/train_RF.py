import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from calssifier_utils import prepare_features_Nyxus, prepare_features_BrainAIC
from sklearn.model_selection import StratifiedGroupKFold

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix_RF.png'):
    """
    Plot confusion matrix for Random Forest results.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Random Forest - Confusion Matrix\n3-way Classification (CN/MCI/AD)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add accuracy information
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.3f}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def plot_feature_importance(rf_model, feature_names, top_n=20, save_path='feature_importance_RF.png'):
    """
    Plot top feature importances from Random Forest.
    """
    # Get feature importances
    importances = rf_model.feature_importances_
    
    # Create feature importance dataframe
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(12, 8))
    top_features = feature_imp_df.head(top_n)
    
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title(f'Top {top_n} Feature Importances - Random Forest\n3-way Classification (CN/MCI/AD)', fontsize=14)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to: {save_path}")
    
    return feature_imp_df

def search_best_rf_model(X_train, y_train, patient_ids_train, cv_folds=5):
    """
    Perform hyperparameter search for Random Forest with patient-aware cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        patient_ids_train: Patient IDs for group-aware splitting
        cv_folds: Number of cross-validation folds
        
    Returns:
        dict: Best hyperparameters
    """
    print("Starting Random Forest hyperparameter search...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of unique patients: {len(np.unique(patient_ids_train))}")
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.3],
        'bootstrap': [True],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Create Random Forest classifier
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        oob_score=True  # Out-of-bag score for additional validation
    )
    
    # Use GroupKFold for patient-aware cross-validation
    group_kfold = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=group_kfold,
        scoring='balanced_accuracy',
        verbose=2,
        n_jobs=1,  # Keep as 1 to avoid memory issues
        return_train_score=True
    )
    
    print("Fitting GridSearchCV...")
    grid_search.fit(X_train, y_train, groups=patient_ids_train)
    
    print("="*50)
    print("RANDOM FOREST HYPERPARAMETER SEARCH RESULTS")
    print("="*50)
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return grid_search.best_params_, grid_search

def search_best_rf_model_fast(X_train, y_train, patient_ids_train, cv_folds=3):
    """
    Fast hyperparameter search for Random Forest with reduced parameter grid.
    
    Args:
        X_train: Training features
        y_train: Training labels
        patient_ids_train: Patient IDs for group-aware splitting
        cv_folds: Number of cross-validation folds (reduced from 5 to 3)
        
    Returns:
        dict: Best hyperparameters
    """
    print("Starting FAST Random Forest hyperparameter search...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of unique patients: {len(np.unique(patient_ids_train))}")
    
    # MUCH SMALLER parameter grid - focus on most important parameters
    param_grid = {
        'n_estimators': [100, 200],  # Reduced from [100, 200, 300]
        'max_depth': [15, None],     # Reduced from [10, 15, 20, None]
        'min_samples_split': [2, 10], # Reduced from [2, 5, 10]
        'max_features': ['sqrt', 0.3], # Reduced from ['sqrt', 'log2', 0.3]
        'class_weight': ['balanced'],   # Only use balanced (most important)
        # Removed min_samples_leaf and bootstrap to reduce combinations
    }
    
    print(f"Parameter combinations to test: {np.prod([len(v) for v in param_grid.values()])}")
    
    # Create Random Forest classifier
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,  # Use all cores
        oob_score=True
    )
    
    # Use fewer CV folds
    group_kfold = GroupKFold(n_splits=cv_folds)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=group_kfold,
        scoring='balanced_accuracy',
        verbose=1,  # Reduced verbosity
        n_jobs=-1,  # Use parallel processing
        return_train_score=False  # Don't compute train scores to save time
    )
    
    print("Fitting GridSearchCV...")
    grid_search.fit(X_train, y_train, groups=patient_ids_train)
    
    print("="*50)
    print("FAST RANDOM FOREST SEARCH RESULTS")
    print("="*50)
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return grid_search.best_params_, grid_search

def search_best_rf_model_minimal(X_train, y_train, patient_ids_train):
    """
    Ultra-fast Random Forest with minimal hyperparameter search.
    """
    print("Starting MINIMAL Random Forest hyperparameter search...")
    
    # Test only a few key combinations
    param_grid = {
        'n_estimators': [100],
        'max_depth': [None],
        'max_features': ['sqrt'],
        'class_weight': ['balanced']
    }
    
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        oob_score=True,
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    # Use only 3-fold CV
    group_kfold = GroupKFold(n_splits=3)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=group_kfold,
        scoring='balanced_accuracy',
        verbose=0,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train, groups=patient_ids_train)
    
    print(f"Minimal search completed!")
    print(f"CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_, grid_search

def train_random_forest(X_train, y_train, patient_ids_train, best_params, feature_names=None):
    """
    Train final Random Forest model with best parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        patient_ids_train: Patient IDs
        best_params: Best hyperparameters from grid search
        feature_names: Names of features for importance analysis
        
    Returns:
        RandomForestClassifier: Trained model
    """
    print("Training final Random Forest model...")
    
    # Create model with best parameters
    rf_model = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1,
        oob_score=True
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    print("="*50)
    print("RANDOM FOREST TRAINING SUMMARY")
    print("="*50)
    print(f"Number of trees: {rf_model.n_estimators}")
    print(f"Out-of-bag score: {rf_model.oob_score_:.4f}")
    print(f"Training accuracy: {rf_model.score(X_train, y_train):.4f}")
    
    # Feature importance analysis
    if feature_names is not None:
        print(f"Top 10 most important features:")
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        for i in range(min(10, len(feature_names))):
            idx = indices[i]
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    return rf_model

def evaluate_random_forest(rf_model, X_test, y_test, patient_ids_test, class_names):
    """
    Evaluate Random Forest model on test set.
    
    Args:
        rf_model: Trained Random Forest model
        X_test: Test features
        y_test: Test labels
        patient_ids_test: Test patient IDs
        class_names: Class names for reporting
    """
    print("\n" + "="*60)
    print("RANDOM FOREST TEST SET EVALUATION")
    print("="*60)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    
    # Basic statistics
    print(f"Test set size: {len(X_test)} images from {len(np.unique(patient_ids_test))} patients")
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Probabilities shape: {y_pred_proba.shape}")
    
    # Classification metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Additional metrics
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"\nAdditional Metrics:")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    
    # Per-class analysis
    print(f"\nPer-class Analysis:")
    for i, class_name in enumerate(class_names):
        class_mask = y_test == i
        if np.any(class_mask):
            class_acc = np.mean(y_pred[class_mask] == y_test[class_mask])
            class_count = np.sum(class_mask)
            print(f"  {class_name}: {class_count} samples, accuracy = {class_acc:.4f}")
    
    return y_pred, y_pred_proba

def train_rf_nyxus():
    """
    Train Random Forest on Nyxus features.
    """
    print("Training Random Forest with Nyxus features...")
    
    # Paths to your data
    image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    
    # Prepare data
    print("Preparing and splitting data...")
    X_train, X_test, y_train, y_test, class_names, patient_ids_train, patient_ids_test = prepare_features_Nyxus(
        features_dir, image_dir, info_csv, 
        features_group="All",  # Use all features
        test_size=0.2,
        classes=['CN', 'MCI', 'AD']
    )
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Classes: {class_names}")
    
    # Get feature names (you might need to modify this based on your feature extraction)
    # For now, create generic feature names
    feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    
    # Hyperparameter search
    best_params, grid_search = search_best_rf_model(
        X_train, y_train, patient_ids_train, cv_folds=5
    )
    
    # Train final model
    rf_model = train_random_forest(
        X_train, y_train, patient_ids_train, best_params, feature_names
    )
    
    # Evaluate on test set
    y_pred, y_pred_proba = evaluate_random_forest(
        rf_model, X_test, y_test, patient_ids_test, class_names
    )
    
    # Create output directory
    os.makedirs('./plots', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    
    # Plot results
    plot_confusion_matrix(y_test, y_pred, class_names, './plots/confusion_matrix_RF_Nyxus.png')
    feature_imp_df = plot_feature_importance(
        rf_model, feature_names, top_n=20, save_path='./plots/feature_importance_RF_Nyxus.png'
    )
    
    # Save model and results
    joblib.dump(rf_model, './models/RF_Nyxus_model.pkl')
    feature_imp_df.to_csv('./models/feature_importance_RF_Nyxus.csv', index=False)
    
    print(f"\nModel saved to: ./models/RF_Nyxus_model.pkl")
    print(f"Feature importance saved to: ./models/feature_importance_RF_Nyxus.csv")
    
    return rf_model, best_params, feature_imp_df

def train_rf_brainiac():
    """
    Train Random Forest on BrainIAC features.
    """
    print("Training Random Forest with BrainIAC features...")
    
    # Paths to your data
    features_csv_path = "/home/ubuntu/data/ADNI_dataset/BrainIAC_features/features.csv"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    
    # Prepare data
    print("Preparing and splitting data...")
    X_train, X_test, y_train, y_test, class_names, patient_ids_train, patient_ids_test = prepare_features_BrainAIC(
        features_csv_path, info_csv, test_size=0.2
    )
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Classes: {class_names}")
    
    # Get feature names
    feature_names = [f"BrainIAC_Feature_{i}" for i in range(X_train.shape[1])]
    
    # Hyperparameter search
    best_params, grid_search = search_best_rf_model(
        X_train, y_train, patient_ids_train, cv_folds=5
    )
    
    # Train final model
    rf_model = train_random_forest(
        X_train, y_train, patient_ids_train, best_params, feature_names
    )
    
    # Evaluate on test set
    y_pred, y_pred_proba = evaluate_random_forest(
        rf_model, X_test, y_test, patient_ids_test, class_names
    )
    
    # Create output directory
    os.makedirs('./plots', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    
    # Plot results
    plot_confusion_matrix(y_test, y_pred, class_names, './plots/confusion_matrix_RF_BrainIAC.png')
    feature_imp_df = plot_feature_importance(
        rf_model, feature_names, top_n=20, save_path='./plots/feature_importance_RF_BrainIAC.png'
    )
    
    # Save model and results
    joblib.dump(rf_model, './models/RF_BrainIAC_model.pkl')
    feature_imp_df.to_csv('./models/feature_importance_RF_BrainIAC.csv', index=False)
    
    print(f"\nModel saved to: ./models/RF_BrainIAC_model.pkl")
    print(f"Feature importance saved to: ./models/feature_importance_RF_BrainIAC.csv")
    
    return rf_model, best_params, feature_imp_df

def train_rf_nyxus_fast():
    """
    Fast Random Forest training on Nyxus features.
    """
    print("Training Random Forest with Nyxus features (FAST VERSION)...")
    
    # Paths to your data
    image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    
    # Prepare data
    print("Preparing and splitting data...")
    X_train, X_test, y_train, y_test, class_names, patient_ids_train, patient_ids_test = prepare_features_Nyxus(
        features_dir, image_dir, info_csv, 
        features_group="All",
        test_size=0.2,
        classes=['CN', 'MCI', 'AD']
    )
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Classes: {class_names}")
    
    # Choose search method
    print("\nChoose search method:")
    print("1. Fast search (2^4 = 16 combinations)")
    print("2. Minimal search (1 combination)")
    print("3. Skip search, use default parameters")
    
    choice = input("Enter choice (1-3) or press Enter for option 3: ").strip()
    
    if choice == "1":
        best_params, grid_search = search_best_rf_model_fast(
            X_train, y_train, patient_ids_train, cv_folds=3
        )
    elif choice == "2":
        best_params, grid_search = search_best_rf_model_minimal(
            X_train, y_train, patient_ids_train
        )
    else:
        print("Using default parameters (fastest option)...")
        best_params = {
            'n_estimators': 100,
            'max_depth': None,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
    
    # Get feature names
    feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    
    # Train final model
    rf_model = train_random_forest(
        X_train, y_train, patient_ids_train, best_params, feature_names
    )
    
    # Evaluate on test set
    y_pred, y_pred_proba = evaluate_random_forest(
        rf_model, X_test, y_test, patient_ids_test, class_names
    )
    
    # Create output directory
    os.makedirs('./plots', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    
    # Plot results
    plot_confusion_matrix(y_test, y_pred, class_names, './plots/confusion_matrix_RF_Nyxus_fast.png')
    feature_imp_df = plot_feature_importance(
        rf_model, feature_names, top_n=20, save_path='./plots/feature_importance_RF_Nyxus_fast.png'
    )
    
    # Save model and results
    joblib.dump(rf_model, './models/RF_Nyxus_model_fast.pkl')
    feature_imp_df.to_csv('./models/feature_importance_RF_Nyxus_fast.csv', index=False)
    
    print(f"\nModel saved to: ./models/RF_Nyxus_model_fast.pkl")
    print(f"Feature importance saved to: ./models/feature_importance_RF_Nyxus_fast.csv")
    
    return rf_model, best_params, feature_imp_df

# Update the main function
if __name__ == "__main__":
    print("="*60)
    print("FAST RANDOM FOREST 3-WAY CLASSIFICATION (CN/MCI/AD)")
    print("="*60)
    
    # Train on Nyxus features with fast search
    rf_nyxus, params_nyxus, importance_nyxus = train_rf_nyxus_fast()
    
    print("\n" + "="*60)
    print("FAST RANDOM FOREST TRAINING COMPLETE!")
    print("="*60)
    print("Check the ./plots/ directory for visualizations")
    print("Check the ./models/ directory for saved models")
