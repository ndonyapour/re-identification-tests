import numpy as np
from calssifier_utils import (prepare_features_Nyxus, train_model,
    prepare_features_BrainAIC, plot_training_curves, search_best_model)
import torch

from sklearn.metrics import classification_report


def train_models_Nyxus():
    # Paths to your data
    image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    
    # Prepare data
    print("Preparing and splitting data")
    X_train, X_test, y_train, y_test, class_names, patient_ids_train, patient_ids_test = prepare_features_Nyxus(
        features_dir, image_dir, info_csv, test_size=0.2
    )
    
    # Train model with patient-aware CV
    input_size = X_train.shape[1]
    print("The grid search started")
    best_params = search_best_model(X_train, y_train, patient_ids_train, input_size)
    print("The grid search finished")
    print(best_params)

    print("The model training started")
    model = train_model(X_train, y_train, patient_ids_train, best_params, input_size)
    print("The model training finished")
    plot_training_curves(model, './plots/ADNI_Nyxus_ALL_training_curves.png')
    torch.save(model, "./models/ADNI_Nyxus_ALL_model.pkl")

    # Make predictions on test set
    print("\n" + "="*60)
    print("MAKING PREDICTIONS ON TEST SET")
    print("="*60)
    
    # Convert test data to the right format
    X_test = X_test.astype(np.float32)
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Print results
    print(f"\nTest set size: {len(X_test)} images from {len(np.unique(patient_ids_test))} patients")
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Probabilities shape: {y_pred_proba.shape}")
    
    # Calculate and print classification metrics
    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

def train_models_BrainAIC():
    # Paths to your data
    features_csv_path = "/home/ubuntu/data/ADNI_dataset/BrainIAC_features/features.csv"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    
    # Prepare data
    print("Preparing and splitting data")
    X_train, X_test, y_train, y_test, class_names, patient_ids_train, patient_ids_test = prepare_features_BrainAIC(
        features_csv_path, info_csv
    )
    
    
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


if __name__ == "__main__":
    train_models_Nyxus()
