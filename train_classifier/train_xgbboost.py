"""
XGBoost Training for ADNI Classification
Optimized for 3-way classification (CN, MCI, AD) with high accuracy targeting 80-90%.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score,
    f1_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from calssifier_utils import (
    prepare_features_Nyxus, prepare_features_BrainAIC, 
    apply_smote_resampling, calculate_patient_level_results,
)

RANDOM_STATE = 123

def create_optimized_xgboost_classifier(n_classes: int = 3, for_cv: bool = False) -> xgb.XGBClassifier:
    """
    Create an optimized XGBoost classifier for ADNI data.
    
    Args:
        n_classes: Number of classes (3 for CN/MCI/AD)
        for_cv: Whether this is for cross-validation (disables early stopping)
        
    Returns:
        XGBoost classifier with optimized parameters
    """
    params = {
        # Core parameters optimized for medical data
        'objective': 'multi:softprob' if n_classes > 2 else 'binary:logistic',
        'n_estimators': 200,  # Good balance of performance vs speed
        'max_depth': 6,       # Prevent overfitting
        'learning_rate': 0.1, # Conservative learning rate
        'subsample': 0.8,     # Row sampling to prevent overfitting
        'colsample_bytree': 0.8,  # Column sampling
        'colsample_bylevel': 0.8, # Additional regularization
        
        # Regularization
        'reg_alpha': 0.1,     # L1 regularization
        'reg_lambda': 1.0,    # L2 regularization
        
        # Performance and reproducibility
        'random_state': RANDOM_STATE,
        'n_jobs': -1,         # Use all cores
        'tree_method': 'hist', # Faster training
        
        # Evaluation
        'eval_metric': 'mlogloss' if n_classes > 2 else 'logloss',
    }
    
    # Only add early stopping for final training, not for CV
    if not for_cv:
        params['early_stopping_rounds'] = 20
    
    return xgb.XGBClassifier(**params)

def hyperparameter_search_xgboost(X_train: np.ndarray, y_train: np.ndarray, 
                                 patient_ids_train: np.ndarray,
                                 search_type: str = 'grid',
                                 cv_folds: int = 5,
                                 n_iter: int = 50) -> dict:
    """
    Perform hyperparameter search for XGBoost.
    
    Args:
        X_train: Training features
        y_train: Training labels
        patient_ids_train: Patient IDs for group-aware CV
        search_type: 'grid' or 'random'
        cv_folds: Number of CV folds
        n_iter: Number of iterations for random search
        
    Returns:
        Dictionary with best parameters and results
    """
    print("="*60)
    print("XGBOOST HYPERPARAMETER SEARCH")
    print("="*60)
    print(f"Search type: {search_type}")
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"CV folds: {cv_folds}")
    
    # Base classifier - NO early stopping for CV
    n_classes = len(np.unique(y_train))
    xgb_classifier = create_optimized_xgboost_classifier(n_classes, for_cv=True)
    
    # Simplified parameter grids to avoid issues
    if search_type == 'grid':
        # Focused grid search - most important parameters
        param_grid = {
            'n_estimators': [100],  # Reduced options
            'max_depth': [4],
            'learning_rate': [0.05],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1.0]
        }
        
        search = GridSearchCV(
            xgb_classifier,
            param_grid,
            cv=StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
            scoring='balanced_accuracy',
            n_jobs=1,  # Reduce parallelism to avoid conflicts
            verbose=2,
            return_train_score=True,
            error_score='raise'  # This will help debug issues
        )
        
    else:  # random search
        # Broader parameter space for random search
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.7],
            'colsample_bytree': [0.7],
            'reg_alpha': [0.1],
            'reg_lambda': [0.1],
            'min_child_weight': [1],
            'gamma': [0.1]
        }
        
        search = RandomizedSearchCV(
            xgb_classifier,
            param_dist,
            n_iter=n_iter,
            cv=StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
            scoring='balanced_accuracy',
            n_jobs=1,  # Reduce parallelism
            verbose=2,
            random_state=RANDOM_STATE,
            return_train_score=True,
            error_score='raise'
        )
    
    # Perform search
    print(f"Starting {search_type} search...")
    try:
        search.fit(X_train, y_train, groups=patient_ids_train)
        
        # Results
        print(f"\nBest parameters:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\nBest CV score: {search.best_score_:.4f}")
        print(f"Best CV std: {search.cv_results_['std_test_score'][search.best_index_]:.4f}")
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'cv_results': search.cv_results_,
            'search_object': search
        }
        
    except Exception as e:
        print(f"Error during hyperparameter search: {str(e)}")
        print("Falling back to default parameters...")
        
        # Return default parameters if search fails
        default_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }
        
        return {
            'best_params': default_params,
            'best_score': 0.0,
            'best_estimator': None,
            'cv_results': None,
            'search_object': None,
            'error': str(e)
        }

def train_xgboost_model(X_train: np.ndarray, X_val: np.ndarray,
                       y_train: np.ndarray, y_val: np.ndarray,
                       best_params: dict = None,
                       use_early_stopping: bool = True) -> xgb.XGBClassifier:
    """
    Train XGBoost model with optimal parameters.
    
    Args:
        X_train, X_val: Feature matrices
        y_train, y_val: Labels
        best_params: Best parameters from hyperparameter search
        use_early_stopping: Whether to use early stopping
        
    Returns:
        Trained XGBoost classifier
    """
    print("="*60)
    print("TRAINING XGBOOST MODEL")
    print("="*60)
    
    n_classes = len(np.unique(y_train))
    
    # Create classifier with best parameters
    if best_params:
        # Remove any CV-specific parameters and add training-specific ones
        train_params = best_params.copy()
        
        # Add parameters for final training
        train_params.update({
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'tree_method': 'hist',
            'eval_metric': 'mlogloss' if n_classes > 2 else 'logloss'
        })
        
        if use_early_stopping:
            train_params['early_stopping_rounds'] = 20
        
        xgb_classifier = xgb.XGBClassifier(**train_params)
        print("Using optimized parameters from hyperparameter search")
    else:
        xgb_classifier = create_optimized_xgboost_classifier(n_classes, for_cv=False)
        print("Using default optimized parameters")
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {n_classes}")
    
    # Class distribution
    unique_classes, counts = np.unique(y_train, return_counts=True)
    print(f"Training class distribution:")
    for cls, count in zip(unique_classes, counts):
        percentage = (count / len(y_train)) * 100
        print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
    
    # Train with or without early stopping
    try:
        if use_early_stopping and X_val is not None:
            print("Training with early stopping...")
            xgb_classifier.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_names=['train', 'val'],
                verbose=50  # Print every 50 rounds
            )
            
            # Get best iteration if available
            try:
                best_iteration = xgb_classifier.get_booster().best_iteration
                print(f"Best iteration: {best_iteration}")
            except:
                print("Could not get best iteration info")
                
        else:
            print("Training without early stopping...")
            xgb_classifier.fit(X_train, y_train)
        
        # Training evaluation
        train_pred = xgb_classifier.predict(X_train)
        train_acc = balanced_accuracy_score(y_train, train_pred)
        
        if X_val is not None:
            val_pred = xgb_classifier.predict(X_val)
            val_acc = balanced_accuracy_score(y_val, val_pred)
            
            print(f"\nTraining Results:")
            print(f"  Training balanced accuracy: {train_acc:.4f}")
            print(f"  Validation balanced accuracy: {val_acc:.4f}")
            
            # Check for overfitting
            if train_acc > val_acc + 0.1:
                print("⚠️  WARNING: Potential overfitting detected!")
        else:
            print(f"\nTraining Results:")
            print(f"  Training balanced accuracy: {train_acc:.4f}")
        
        return xgb_classifier
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Trying to train without early stopping...")
        
        # Fallback: train without early stopping
        simple_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
        
        xgb_classifier = xgb.XGBClassifier(**simple_params)
        xgb_classifier.fit(X_train, y_train)
        
        return xgb_classifier

def evaluate_xgboost_model(model: xgb.XGBClassifier, 
                          X_test: np.ndarray, y_test: np.ndarray,
                          patient_ids_test: np.ndarray,
                          class_names: list,
                          save_plots: bool = True) -> dict:
    """
    Comprehensive evaluation of XGBoost model.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
        patient_ids_test: Patient IDs for test samples
        class_names: Names of classes
        save_plots: Whether to save plots
        
    Returns:
        Dictionary with evaluation results
    """
    print("="*60)
    print("XGBOOST MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Basic metrics
    accuracy = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Test Results:")
    print(f"  Balanced Accuracy: {accuracy:.4f}")
    print(f"  Macro F1-Score: {macro_f1:.4f}")
    print(f"  Weighted F1-Score: {weighted_f1:.4f}")
    
    # AUC (if multi-class)
    try:
        if len(np.unique(y_test)) > 2:
            auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
            print(f"  Macro AUC: {auc_score:.4f}")
        else:
            auc_score = roc_auc_score(y_test, y_proba[:, 1])
            print(f"  AUC: {auc_score:.4f}")
    except:
        auc_score = None
        print("  AUC: Could not calculate")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Patient-level results
    print(f"\nPatient-Level Results:")
    patient_results = calculate_patient_level_results(
        y_pred, y_test, patient_ids_test, class_names
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    if save_plots:
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('XGBoost - Confusion Matrix\n3-way Classification (CN/MCI/AD)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('./plots/xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance plot
        plot_xgboost_feature_importance(model, top_n=20, save_path='./plots/xgboost_feature_importance.png')
        
        print(f"Plots saved to ./plots/")
    
    results = {
        'balanced_accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'auc': auc_score,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, target_names=class_names, output_dict=True),
        'patient_results': patient_results,
        'predictions': y_pred,
        'probabilities': y_proba
    }
    
    return results

def plot_xgboost_feature_importance(model: xgb.XGBClassifier, 
                                   feature_names: list = None,
                                   top_n: int = 20,
                                   importance_type: str = 'weight',
                                   save_path: str = 'xgboost_feature_importance.png'):
    """
    Plot XGBoost feature importance.
    
    Args:
        model: Trained XGBoost model
        feature_names: Names of features
        top_n: Number of top features to plot
        importance_type: 'weight', 'gain', 'cover', or 'total_gain'
        save_path: Path to save plot
    """
    # Get feature importance
    importance = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importance))]
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
    plt.title(f'Top {top_n} XGBoost Feature Importances\n3-way ADNI Classification')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to: {save_path}")
    return importance_df

def train_xgboost_nyxus():
    """
    Complete XGBoost training pipeline for Nyxus features.
    """
    print("="*80)
    print("XGBOOST TRAINING FOR ADNI CLASSIFICATION - NYXUS FEATURES")
    print("="*80)
    
    # Data loading
    image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    
    X_train, X_val, X_test, y_train, y_val, y_test, class_names, patient_ids_train, patient_ids_val, patient_ids_test = prepare_features_Nyxus(
        features_dir, image_dir, info_csv, test_size=0.2, val_size=0.15, features_group="All", classes=['CN','AD', 'MCI']
    )
    
    print(f"Data loaded successfully:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # Hyperparameter search with error handling
    print(f"\nStarting hyperparameter search...")
    search_results = hyperparameter_search_xgboost(
        X_train, y_train, patient_ids_train,
        search_type='grid',  # Start with grid search
        cv_folds=3,
        n_iter=20  # Reduced for faster execution
    )
    
    # Train final model
    print(f"\nTraining final XGBoost model...")
    model = train_xgboost_model(
        X_train, X_val, y_train, y_val,
        best_params=search_results['best_params'],
        use_early_stopping=True
    )
    
    # Evaluate model
    print(f"\nEvaluating model on test set...")
    results = evaluate_xgboost_model(
        model, X_test, y_test, patient_ids_test, class_names, save_plots=False
    )
    
    # Save model
    model_save_path = './models/xgboost_nyxus_model.pkl'
    joblib.dump(model, model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Save results
    results_save_path = './models/xgboost_nyxus_results.pkl'
    joblib.dump(results, results_save_path)
    print(f"Results saved to: {results_save_path}")
    
    return model, results, search_results

def train_xgboost_brainaic():
    """
    Complete XGBoost training pipeline for BrainAIC features.
    """
    print("="*80)
    print("XGBOOST TRAINING FOR ADNI CLASSIFICATION - BRAINAIC FEATURES")
    print("="*80)
    
    # Data loading
    features_csv_path = "/home/ubuntu/data/ADNI_dataset/BrainIAC_features/features.csv"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    
    X_train, X_val, X_test, y_train, y_val, y_test, class_names, patient_ids_train, patient_ids_val, patient_ids_test = prepare_features_BrainAIC(
        features_csv_path, info_csv, test_size=0.2, val_size=0.15
    )
    
    print(f"Data loaded successfully:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # Apply SMOTE
    X_train, y_train, patient_ids_train = apply_smote_resampling(
        X_train, y_train, patient_ids_train, method='smote'
    )
    
    # Hyperparameter search
    search_results = hyperparameter_search_xgboost(
        X_train, y_train, patient_ids_train,
        search_type='grid',
        cv_folds=3
    )
    
    # Train and evaluate
    model = train_xgboost_model(
        X_train, X_val, y_train, y_val,
        best_params=search_results['best_params']
    )
    
    results = evaluate_xgboost_model(
        model, X_test, y_test, patient_ids_test, class_names
    )
    
    # Save
    joblib.dump(model, './models/xgboost_brainaic_model.pkl')
    joblib.dump(results, './models/xgboost_brainaic_results.pkl')
    
    return model, results, search_results

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('./models', exist_ok=True)
    
    # Choose training method
    use_simple_training = True  # Set to False for full hyperparameter search
    

    print("Running simple XGBoost training...")
    model, results, search_results = train_xgboost_nyxus()
    search_results = None

    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"Final Test Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Final Test Macro F1: {results['macro_f1']:.4f}")
    print(f"Patient-level Balanced Accuracy: {results['patient_results']['patient_balanced_accuracy']:.4f}")
