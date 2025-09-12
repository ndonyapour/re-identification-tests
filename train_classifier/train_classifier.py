import numpy as np
from calssifier_utils import (prepare_features_Nyxus, train_model,
    prepare_features_BrainAIC, plot_training_curves, search_best_model, calculate_patient_level_results, apply_smote_resampling)
import torch

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd


def train_models_Nyxus():
    image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    
    # Use the new 3-way split function that returns train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test, class_names, patient_ids_train, patient_ids_val, patient_ids_test = prepare_features_Nyxus(
        features_dir, image_dir, info_csv, test_size=0.2, val_size=0.1, features_group="All", classes=['CN','AD', 'MCI']
    )

    print("The grid search started")
    # For grid search, we can use a simple approach or modify search function to use explicit validation
    # For now, let's use default parameters or a simplified search

    #X_train, y_train = apply_smote_resampling(X_train, y_train)
    best_params = search_best_model(X_train, y_train, patient_ids_train, cv_folds=3)

    print("Using default parameters:", best_params)

    print("The model training started")
    # Use the new training function with explicit validation set
    model = train_model(X_train, X_val, y_train, y_val, best_params)
    print("The model training finished")
    
    # Plot training curves
    plot_training_curves(model, './plots/ADNI_Nyxus_Texture_training_curves.png')
    
    # Save the model
    #torch.save(model, "./models/ADNI_Nyxus_Texture_model.pkl")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("MAKING PREDICTIONS ON TEST SET")
    print("="*60)
    
    X_test = X_test.astype(np.float32)
    test_predictions = model.predict(X_test)
    test_probabilities = model.predict_proba(X_test)
    
    print(f"Test set size: {len(X_test)} images from {len(np.unique(patient_ids_test))} patients")
    print(f"Predictions shape: {test_predictions.shape}")
    print(f"Probabilities shape: {test_probabilities.shape}")
    
    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    
    from sklearn.metrics import classification_report, balanced_accuracy_score
    test_accuracy = balanced_accuracy_score(y_test, test_predictions)
    print(f"Test balanced accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions, target_names=class_names))

    patient_level_results = calculate_patient_level_results(test_predictions, y_test, patient_ids_test, class_names)
    
    return model, best_params, X_test, y_test, test_predictions, class_names


def train_models_BrainAIC():
    # Paths to your data
    features_csv_path = "/home/ubuntu/data/ADNI_dataset/BrainIAC_features/features.csv"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    
    # Prepare data
    print("Preparing and splitting data")
    X_train, X_test, y_train, y_test, class_names, patient_ids_train, patient_ids_test = prepare_features_BrainAIC(
        features_csv_path, info_csv
    )
    
    import pdb; pdb.set_trace()
    # Train model with patient-aware CV
    input_size = X_train.shape[1]
    print("The grid search started")
    best_params = search_best_model(X_train, y_train, patient_ids_train, input_size)
    print("The grid search finished")
    print(best_params)

    print("The model training started")
    model = train_model(X_train, y_train, patient_ids_train, best_params, input_size)
    print("The model training finished")
    plot_training_curves(model, './plots/ADNI_BrainAIC_training_curves.png')
    torch.save(model, "./models/ADNI_BrainAIC_model.pkl")



def train_logistic_regression_Nyxus():
    image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    
    X_train, X_val, X_test, y_train, y_val, y_test, class_names, patient_ids_train, patient_ids_val, patient_ids_test = prepare_features_Nyxus(
        features_dir, image_dir, info_csv, test_size=0.2, val_size=0.15, features_group="All", classes=['CN','AD', 'MCI']
    )

    score = lambda model, X_train, y_train, X_val, y_val : pd.Series([model.score(X_train, y_train), model.score(X_val, y_val)],index=['Model Training Accuracy', 'Model Validation Accuracy'])
    
    # Fix the convergence issue by increasing max_iter and using appropriate solver
    print("Training Logistic Regression with Cross-Validation...")
    clf_mult = LogisticRegressionCV(
        multi_class='multinomial',
        max_iter=2000,  # Increase iterations to ensure convergence
        solver='lbfgs',  # Explicitly specify solver
        cv=5,  # 5-fold cross-validation
        random_state=123,
        class_weight='balanced',  # Handle class imbalance
        scoring='balanced_accuracy'  # Use balanced accuracy for evaluation
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    # Fit the model
    clf_mult.fit(X_train, y_train)
    
    # Evaluate performance
    clf_mult_score = score(clf_mult, X_train, y_train, X_val, y_val)
    clf_mult_test_score = clf_mult.score(X_test, y_test)
    
    print("\nTraining and Validation Scores:")
    print(clf_mult_score)
    print(f"\nTest Score: {clf_mult_test_score:.4f}")
    
    # Get predictions for detailed evaluation
    y_pred_val = clf_mult.predict(X_val)
    y_pred_test = clf_mult.predict(X_test)
    
    # Print detailed classification reports
    print("\n" + "="*60)
    print("VALIDATION SET DETAILED EVALUATION")
    print("="*60)
    print(classification_report(y_val, y_pred_val, target_names=class_names))
    
    print("\n" + "="*60)
    print("TEST SET DETAILED EVALUATION")
    print("="*60)
    print(classification_report(y_test, y_pred_test, target_names=class_names))
    
    # Print best regularization parameter
    print(f"\nBest regularization parameter (C): {clf_mult.C_}")
    print(f"Number of iterations: {clf_mult.n_iter_}")
    
    return clf_mult, X_test, y_test, y_pred_test, class_names

if __name__ == "__main__":
    train_models_Nyxus()
    #train_logistic_regression_Nyxus()