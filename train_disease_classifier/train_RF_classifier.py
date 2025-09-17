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
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from calssifier_utils import calculate_averaged_classification_report, print_averaged_classification_report

#from xgboost import XGBClassifier

RANDOM_STATE = 123

def make_rf_pipeline(best_params=None):
    """
    Leak-proof RF pipeline:
      impute (median) -> drop constant cols -> (optional) SelectKBest -> RandomForest
    You can change k in GridSearch or fix it here.
    """
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=True,
        **(best_params or {})
    )
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("varth", VarianceThreshold(threshold=0.0)),
        ("kbest", SelectKBest(score_func=f_classif, k=256)),  # k will be tuned below
        ("rf", rf),
    ])
    return pipe

def search_best_rf_model_pipeline(X_train, y_train, patient_ids_train, random_state: int = 42, cv_folds=3):
    print("Starting FAST Random Forest hyperparameter search...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of unique patients: {len(np.unique(patient_ids_train))}")

    param_grid = {
        "kbest__k": [50, 100],           # tune feature selection size
        "rf__n_estimators": [400],
        "rf__max_depth": [5, 15, None],
        "rf__min_samples_split": [2, 10],
        "rf__min_samples_leaf": [1, 2],
        "rf__max_features": ["sqrt", 0.3],
        "rf__class_weight": ["balanced", "balanced_subsample"],
        "rf__bootstrap": [True],
    }

    macro_f1 = make_scorer(f1_score, average="macro")
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=make_rf_pipeline(),
        param_grid=param_grid,
        cv=cv,
        scoring=macro_f1,
        verbose=1,
        n_jobs=-1,
        return_train_score=False
    )

    print("Fitting GridSearchCV...")
    grid_search.fit(X_train, y_train, groups=patient_ids_train)

    print("="*50)
    print("FAST RANDOM FOREST SEARCH RESULTS")
    print("="*50)
    print(f"Best CV macro-F1: {grid_search.best_score_:.4f}")
    print("Best parameters:")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")

    # If you still want bare RF params (optional):
    best_params = {k.replace("rf__", ""): v for k, v in grid_search.best_params_.items() if k.startswith("rf__")}
    best_params["_kbest_k"] = grid_search.best_params_.get("kbest__k")

    # Return both the params and the fitted best pipeline
    return best_params, grid_search.best_estimator_

def train_random_forest_pipeline(X_train, y_train, best_params, random_state):
    """
    Train the best pipeline returned by GridSearchCV (no leakage).
    """
    print("Training final Random Forest pipeline...")
    best_pipeline = make_rf_pipeline()
    best_pipeline.set_params(**best_params)
    best_pipeline.fit(X_train, y_train)
    rf = best_pipeline.named_steps["rf"]
    if getattr(rf, "oob_score", False) and getattr(rf, "bootstrap", True):
        print(f"OOB score (informational): {rf.oob_score_:.4f}")
    train_accuracy = best_pipeline.score(X_train, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")

    return best_pipeline, {"train_acc": train_accuracy, "oob_score": rf.oob_score_}

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

def search_best_rf_model(X_train, y_train, patient_ids_train, random_state: int = 42, cv_folds=3):
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
        'n_estimators': [25, 50],  # Reduced from [100, 200, 300]
        'max_depth': [3, 5, None],     # Reduced from [10, 15, 20, None]
        'min_samples_split': [5, 10], # Reduced from [2, 5, 10]
        'max_features': ['sqrt', 0.3], # Reduced from ['sqrt', 'log2', 0.3]
        'class_weight': ['balanced'],   # Only use balanced (most important)
        # Removed min_samples_leaf and bootstrap to reduce combinations
    }
    
    print(f"Parameter combinations to test: {np.prod([len(v) for v in param_grid.values()])}")
    
    # Create Random Forest classifier
    rf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,  # Use all cores
        #oob_score=True
    )
    
    # Use fewer CV folds
    group_kfold = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
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


def train_random_forest(X_train, y_train, best_params, random_state):
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
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    train_accuracy = rf_model.score(X_train, y_train)
    oob_score = rf_model.oob_score_
    
    print("="*50)
    print("RANDOM FOREST TRAINING SUMMARY")
    print("="*50)
    print(f"Number of trees: {rf_model.n_estimators}")
    print(f"Out-of-bag score: {oob_score:.4f}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    
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
        best_params, best_pipeline = search_best_rf_model(
            X_train, y_train, patient_ids_train, random_state=random_state, cv_folds=3
        )
        
        # Train final model with the pipeline
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


def train_rf_nyxus_pipeline(features_group: str = "All", random_states: list[int] = [42, 123, 1234]):
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
        best_params, best_pipeline = search_best_rf_model_pipeline(
            X_train, y_train, patient_ids_train, random_state=random_state, cv_folds=3
        )
        
        # ANTI-OVERFITTING parameters - much more conservative

        # Train final model - UNPACK THE TUPLE
        rf_model, metrics = train_random_forest_pipeline(
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
        best_params, best_pipeline = search_best_rf_model(
            X_train, y_train, patient_ids_train, random_state=random_state, cv_folds=3
        )
        
        # Train final model with the pipeline
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
    #train_rf_nyxus_pipeline(features_group="All", random_states=[42])
    #train_rf_nyxus(features_group="Shape", random_states=[42, 123, 1234])
    train_rf_brainiac(random_states=[42, 123, 1234])