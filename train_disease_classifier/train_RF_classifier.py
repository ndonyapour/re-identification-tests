import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Failed to initialize NumPy.*')

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
from calssifier_utils import prepare_features_Nyxus, prepare_features_BrainAIC, prepare_features_Pyradiomics
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from calssifier_utils import calculate_averaged_classification_report, print_averaged_classification_report
from sklearn.feature_selection import f_classif

#from xgboost import XGBClassifier

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

def search_best_rf_model(X_train, y_train, patient_ids_train, random_state: int = 42, cv_folds=3):
    """
    More aggressive anti-overfitting parameter grid
    """
    print("Starting AGGRESSIVE ANTI-OVERFITTING Random Forest search...")
    
    # MORE AGGRESSIVE anti-overfitting parameters
    param_grid = {
        'n_estimators': [100, 150],              # Keep moderate number
        'max_depth': [3, 5, 8],                  # REDUCE max depth further
        'min_samples_split': [20, 50, 100],      # INCREASE significantly
        'min_samples_leaf': [10, 20, 30],        # INCREASE significantly  
        'max_features': ['sqrt'],                # Only sqrt (most restrictive)
        'class_weight': ['balanced'],
    }
    
    # More aggressive regularization
    rf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
        # STRONGER regularization
        min_impurity_decrease=0.01,              # 10x higher than before
        ccp_alpha=0.05,                          # 5x higher pruning
        max_samples=0.7,                         # Use only 70% of samples
    )
    
    # Use 5 CV folds for better validation
    group_kfold = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Perform grid search with train score monitoring
    # grid_search = GridSearchCV(
    #     estimator=rf,
    #     param_grid=param_grid,
    #     cv=group_kfold,
    #     scoring='balanced_accuracy',
    #     verbose=1,
    #     n_jobs=-1,
    #     return_train_score=True  # Monitor overfitting
    # )

    grid_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=50,                              # Test 50 random combinations
        cv=group_kfold,
        scoring='balanced_accuracy',
        verbose=1,
        n_jobs=-1,
        return_train_score=True,                # Monitor train vs validation gap
        random_state=random_state
    )
    
    print("Fitting GridSearchCV...")
    grid_search.fit(X_train, y_train, groups=patient_ids_train)
    
    # Check for overfitting
    results_df = pd.DataFrame(grid_search.cv_results_)
    best_idx = grid_search.best_index_
    train_score = results_df.loc[best_idx, 'mean_train_score']
    val_score = results_df.loc[best_idx, 'mean_test_score']
    overfitting_gap = train_score - val_score
    
    print("="*50)
    print("ANTI-OVERFITTING RANDOM FOREST SEARCH RESULTS")
    print("="*50)
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Train score: {train_score:.4f}")
    print(f"Train-validation gap: {overfitting_gap:.4f}")
    
    if overfitting_gap > 0.1:
        print("⚠️  WARNING: Large train-validation gap detected! Model may be overfitting.")
    else:
        print("✅ Good train-validation gap. Model appears well-regularized.")
    
    print(f"Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return grid_search.best_params_, grid_search

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
        feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = feature_selector.fit_transform(X_train_var, y_train)
        X_test_selected = feature_selector.transform(X_test_var)
        print(f"After feature selection: {X_train_selected.shape[1]} features")
    else:
        X_train_selected = X_train_var
        X_test_selected = X_test_var
    
    return X_train_selected, X_test_selected

def train_random_forest(X_train, y_train, best_params, random_state):
    """
    Train final Random Forest model with anti-overfitting measures.
    
    Args:
        X_train: Training features
        y_train: Training labels
        best_params: Best hyperparameters from grid search
        random_state: Random state for reproducibility
        
    Returns:
        RandomForestClassifier: Trained model with metrics
    """
    print("Training final Random Forest model with anti-overfitting measures...")
    
    # Add anti-overfitting parameters to best params
    final_params = {
        **best_params,
        'random_state': random_state,
        'n_jobs': -1,
        'oob_score': True,
        # Anti-overfitting measures
        'min_impurity_decrease': 0.001,  # Require minimum improvement to split
        'ccp_alpha': 0.01,               # Cost complexity pruning
    }
    
    # Create model with best parameters + anti-overfitting measures
    rf_model = RandomForestClassifier(**final_params)
    
    # Train the model
    rf_model.fit(X_train, y_train)
    train_accuracy = rf_model.score(X_train, y_train)
    oob_score = rf_model.oob_score_
    
    print("="*50)
    print("ANTI-OVERFITTING TRAINING SUMMARY")
    print("="*50)
    print(f"Number of trees: {rf_model.n_estimators}")
    print(f"Max depth: {rf_model.max_depth}")
    print(f"Min samples leaf: {rf_model.min_samples_leaf}")
    print(f"Min samples split: {rf_model.min_samples_split}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Out-of-bag score: {oob_score:.4f}")
    print(f"Train-OOB gap: {train_accuracy - oob_score:.4f}")
    
    # Check for overfitting
    if train_accuracy - oob_score > 0.1:
        print("⚠️  WARNING: Large train-OOB gap! Model may still be overfitting.")
        print("   Consider: higher min_samples_leaf, lower max_depth, or higher ccp_alpha")
    else:
        print("✅ Good train-OOB gap. Model appears well-regularized.")
    
    return rf_model, {"train_acc": train_accuracy, "oob_score": oob_score}


def train_rf_nyxus(features_group: str = "All", random_states: list[int] = [42, 123, 1234]):
    """
    Train Random Forest on Nyxus features.
    """
    print("Training Random Forest with Nyxus features...")
    
    # Paths to your data
    image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    
    accuracies = {'train_acc': [], 'oob_score': [], 'test_acc': []}
    all_classification_reports = []
    for random_state in random_states:
        # Prepare data

        print(f"Training with random state: {random_state}")
        print("Preparing and splitting data...")
        np.random.seed(random_state)
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
   
        print(f"Training set: {X_train.shape[0]} samples, {len(np.unique(patient_ids_train))} patients")
        print(f"Test set: {X_test.shape[0]} samples, {len(np.unique(patient_ids_test))} patients")
        print(f"Classes: {class_names}")
    
        # Get feature names (you might need to modify this based on your feature extraction)
        # For now, create generic feature names
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    

        # Hyperparameter search
        best_params, best_rf_model = search_best_rf_model(
            X_train, y_train, patient_ids_train, random_state=random_state, cv_folds=5
        )
        # Train final model with the RF model
        rf_model, metrics = train_random_forest(
            X_train, y_train, best_params, random_state
        )

        # Create models directory if it doesn't exist
        os.makedirs("./models", exist_ok=True)

        # Save the model using joblib
        joblib.dump(rf_model, f"./models/rf_model_Nyxus_{features_group}_{random_state}.pkl")
        print(f"Model saved to: ./models/rf_model_Nyxus_{features_group}_{random_state}.pkl")
        X_test = X_test.astype(np.float32)
        test_predictions = rf_model.predict(X_test)
        test_accuracy = np.mean(test_predictions == y_test)

        # Store results with correct metrics
        accuracies['train_acc'].append(metrics['train_acc'])
        accuracies['oob_score'].append(metrics['oob_score'])
        accuracies['test_acc'].append(test_accuracy)
        
        from sklearn.metrics import classification_report
        report_dict = classification_report(y_test, test_predictions, target_names=class_names, output_dict=True)
        all_classification_reports.append(report_dict)

    # Calculate averaged classification report
    averaged_report = calculate_averaged_classification_report(all_classification_reports, class_names)
    print_averaged_classification_report(averaged_report, class_names, len(random_states))
    
    # Fix the final summary - remove val_acc since it doesn't exist
    print(f"\nTraining accuracy: {np.mean(accuracies['train_acc']):.4f} ± {np.std(accuracies['train_acc']):.4f}")
    print(f"OOB score: {np.mean(accuracies['oob_score']):.4f} ± {np.std(accuracies['oob_score']):.4f}")
    print(f"Test accuracy: {np.mean(accuracies['test_acc']):.4f} ± {np.std(accuracies['test_acc']):.4f}")


def train_rf_pyradiomics(features_group: str = "All", random_states: list[int] = [42, 123, 1234]):
    """
    Train Random Forest on Nyxus features.
    """
    print("Training Random Forest with Nyxus features...")
    
    # Paths to your data
    image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    features_dir = "/home/ubuntu/data/ADNI_dataset/Pyradiomics_features/"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    
    accuracies = {'train_acc': [], 'oob_score': [], 'test_acc': []}
    all_classification_reports = []
    for random_state in random_states:
        # Prepare data

        print(f"Training with random state: {random_state}")
        print("Preparing and splitting data...")
        np.random.seed(random_state)
        X_train, X_val, X_test, y_train, y_val, y_test, class_names, patient_ids_train, patient_ids_val, patient_ids_test = prepare_features_Pyradiomics(
            features_dir, image_dir, info_csv, 
            features_group=features_group, 
            random_state=random_state, 
            test_size=0.2,
            classes=['CN', 'MCI', 'AD']
        )

        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)
        patient_ids_train = np.concatenate((patient_ids_train, patient_ids_val), axis=0)
   
        print(f"Training set: {X_train.shape[0]} samples, {len(np.unique(patient_ids_train))} patients")
        print(f"Test set: {X_test.shape[0]} samples, {len(np.unique(patient_ids_test))} patients")
        print(f"Classes: {class_names}")
    
        # Get feature names (you might need to modify this based on your feature extraction)
        # For now, create generic feature names
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    

        # Hyperparameter search
        best_params, best_rf_model = search_best_rf_model(
            X_train, y_train, patient_ids_train, random_state=random_state, cv_folds=5
        )
        # Train final model with the RF model
        rf_model, metrics = train_random_forest(
            X_train, y_train, best_params, random_state
        )

        # Create models directory if it doesn't exist
        os.makedirs("./models", exist_ok=True)

        # Save the model using joblib
        # joblib.dump(rf_model, f"./models/rf_model_Nyxus_{features_group}_{random_state}.pkl")
        # print(f"Model saved to: ./models/rf_model_Nyxus_{features_group}_{random_state}.pkl")
        X_test = X_test.astype(np.float32)
        test_predictions = rf_model.predict(X_test)
        test_accuracy = np.mean(test_predictions == y_test)

        # Store results with correct metrics
        accuracies['train_acc'].append(metrics['train_acc'])
        accuracies['oob_score'].append(metrics['oob_score'])
        accuracies['test_acc'].append(test_accuracy)
        
        from sklearn.metrics import classification_report
        report_dict = classification_report(y_test, test_predictions, target_names=class_names, output_dict=True)
        all_classification_reports.append(report_dict)

    # Calculate averaged classification report
    averaged_report = calculate_averaged_classification_report(all_classification_reports, class_names)
    print_averaged_classification_report(averaged_report, class_names, len(random_states))
    
    # Fix the final summary - remove val_acc since it doesn't exist
    print(f"\nTraining accuracy: {np.mean(accuracies['train_acc']):.4f} ± {np.std(accuracies['train_acc']):.4f}")
    print(f"OOB score: {np.mean(accuracies['oob_score']):.4f} ± {np.std(accuracies['oob_score']):.4f}")
    print(f"Test accuracy: {np.mean(accuracies['test_acc']):.4f} ± {np.std(accuracies['test_acc']):.4f}")


def train_rf_brainiac(random_states: list[int] = [42, 123, 1234]):
    """
    Train Random Forest on BrainIAC features.
    """
    print("Training Random Forest with BrainIAC features...")
    
    # Paths to the data
    features_csv_path = "/home/ubuntu/data/ADNI_dataset/BrainIAC_features/features.csv"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    
    accuracies = {'train_acc': [], 'oob_score': [], 'test_acc': []}
    all_classification_reports = []
    for random_state in random_states:
        print(f"Training with random state: {random_state}")
        print("Preparing and splitting data...")
        np.random.seed(random_state)
        X_train, X_val, X_test, y_train, y_val, y_test, class_names, patient_ids_train, patient_ids_val, patient_ids_test = prepare_features_BrainAIC(
            features_csv_path, info_csv, test_size=0.2, random_state=random_state, classes=['CN', 'MCI', 'AD']
        )
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)
        patient_ids_train = np.concatenate((patient_ids_train, patient_ids_val), axis=0)
   
        print(f"Training set: {X_train.shape[0]} samples, {len(np.unique(patient_ids_train))} patients")
        print(f"Test set: {X_test.shape[0]} samples, {len(np.unique(patient_ids_test))} patients")
        print(f"Classes: {class_names}")
    
        # Get feature names (you might need to modify this based on your feature extraction)
        # For now, create generic feature names
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
   


        # Get feature names (you might need to modify this based on your feature extraction)
        # For now, create generic feature names
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    
        # Hyperparameter search
        best_params, best_rf_model = search_best_rf_model(
            X_train, y_train, patient_ids_train, random_state=random_state, cv_folds=5
        )
        
        # Train final model with the RF model
        rf_model, metrics = train_random_forest(
            X_train, y_train, best_params, random_state
        )

        X_test = X_test.astype(np.float32)
        test_predictions = rf_model.predict(X_test)
        test_accuracy = np.mean(test_predictions == y_test)

        # Store results with correct metrics
        accuracies['train_acc'].append(metrics['train_acc'])
        accuracies['oob_score'].append(metrics['oob_score'])
        accuracies['test_acc'].append(test_accuracy)
        
        from sklearn.metrics import classification_report
        report_dict = classification_report(y_test, test_predictions, target_names=class_names, output_dict=True)
        all_classification_reports.append(report_dict)

    # Calculate averaged classification report
    averaged_report = calculate_averaged_classification_report(all_classification_reports, class_names)
    print_averaged_classification_report(averaged_report, class_names, len(random_states))
    
    # Fix the final summary - remove val_acc since it doesn't exist
    print(f"\nTraining accuracy: {np.mean(accuracies['train_acc']):.4f} ± {np.std(accuracies['train_acc']):.4f}")
    print(f"OOB score: {np.mean(accuracies['oob_score']):.4f} ± {np.std(accuracies['oob_score']):.4f}")
    print(f"Test accuracy: {np.mean(accuracies['test_acc']):.4f} ± {np.std(accuracies['test_acc']):.4f}")

 
# Update the main function
if __name__ == "__main__":
    train_rf_nyxus(features_group="Texture", random_states=[42, 123, 1234])
    # train_rf_brainiac(random_states=[42, 123, 1007])
    # train_rf_pyradiomics(features_group="Shape", random_states=[42, 123, 1234])

