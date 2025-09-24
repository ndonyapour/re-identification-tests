import numpy as np
from calssifier_utils import (prepare_features_Nyxus, train_model,
    prepare_features_BrainAIC, plot_training_curves, search_best_model, calculate_patient_level_results, 
    print_averaged_classification_report,
    calculate_averaged_classification_report,
    divide_test_set_reidentification)
import torch

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
from sklearn.metrics import classification_report
import time


def train_models_Nyxus(features_group: str = "All", random_states: list[int] = [42, 123, 1234]):
    image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    reidentification_results = "../Nyxus_reidentification_analysis/matches_diff_dates.csv" 
    
    # Store results for all runs
    accuracies = {'train_acc': [], 'val_acc': [], 'test_acc': [], 'test_acc_re_success': [], 'test_acc_re_failure': []}
    all_classification_reports_re_success = []  # Store all classification reports
    all_classification_reports_re_failure = []  # Store all classification reports
    
    for random_state in random_states:
        print(f"Training with random state: {random_state}")
        
        # ADD THESE LINES - Same as BrainIAC function
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        
        X_train, X_val, X_test, y_train, y_val, y_test, class_names, patient_ids_train, patient_ids_val, patient_ids_test, image_vol_names_test = prepare_features_Nyxus(
            features_dir, image_dir, info_csv, random_state=random_state, test_size=0.2, val_size=0.1, features_group=features_group, classes=['M', 'F']
        )

        print("Searching best parameters")
        best_params = search_best_model(X_train, y_train, patient_ids_train, random_state=random_state, cv_folds=5)

        # best_params = {
        #     'batch_size': 32,
        #     'max_epochs': 120,
        #     'module__dropout_rate': 0.4,
        #     'module__num_hidden_layers': 2
        # }
        print("Using default parameters:", best_params)

        print("The model training started")
        model, results = train_model(X_train, X_val, y_train, y_val, best_params, random_state=random_state)
        print("The model training finished")
        
        plot_training_curves(model, f'./plots/ADNI_Nyxus_{features_group}_training_curves.png')
        
        # Evaluate on test set
        print("\n" + "="*60)
        print("MAKING PREDICTIONS ON TEST SET")
        print("="*60)
        accuracies['train_acc'].append(results['train_acc'])
        accuracies['val_acc'].append(results['val_acc'])

        test_re_success, test_re_failure = divide_test_set_reidentification(X_test, y_test, image_vol_names_test, reidentification_results)

        test_predictions_re_success = model.predict(test_re_success['X'].astype(np.float32))
        test_probabilities_re_success = model.predict_proba(test_re_success['X'].astype(np.float32))
        test_predictions_re_failure = model.predict(test_re_failure['X'].astype(np.float32))
        test_probabilities_re_failure = model.predict_proba(test_re_failure['X'].astype(np.float32))

        accuracies['test_acc_re_success'].append(np.mean(test_predictions_re_success == test_re_success['y']))
        accuracies['test_acc_re_failure'].append(np.mean(test_predictions_re_failure == test_re_failure['y']))

        print("class distribution for re-identification success: ", np.bincount(test_re_success['y']))
        print("class distribution for re-identification failure: ", np.bincount(test_re_failure['y']))

        # Get classification report as dictionary and store it
        
        report_dict = classification_report(test_re_success['y'], test_predictions_re_success, target_names=class_names, output_dict=True)
        all_classification_reports_re_success.append(report_dict)
        report_dict = classification_report(test_re_failure['y'], test_predictions_re_failure, target_names=class_names, output_dict=True)
        all_classification_reports_re_failure.append(report_dict)

    # Calculate averaged classification report
    averaged_report_re_success = calculate_averaged_classification_report(all_classification_reports_re_success, class_names)
    averaged_report_re_failure = calculate_averaged_classification_report(all_classification_reports_re_failure, class_names)
    print_averaged_classification_report(averaged_report_re_success, class_names, len(random_states), re_identification=True)
    print_averaged_classification_report(averaged_report_re_failure, class_names, len(random_states), re_identification=False)

    
    print(f"\nTraining accuracy: {np.mean(accuracies['train_acc']):.4f} ± {np.std(accuracies['train_acc']):.4f}")
    print(f"Validation accuracy: {np.mean(accuracies['val_acc']):.4f} ± {np.std(accuracies['val_acc']):.4f}")
    print(f"Test accuracy for re-identification success: {np.mean(accuracies['test_acc_re_success']):.4f} ± {np.std(accuracies['test_acc_re_success']):.4f}")
    print(f"Test accuracy for re-identification failure: {np.mean(accuracies['test_acc_re_failure']):.4f} ± {np.std(accuracies['test_acc_re_failure']):.4f}")





def train_models_BrainAIC(random_states: list[int] = [42, 123, 1234]):
    # Paths to your data
    features_csv_path = "/home/ubuntu/data/ADNI_dataset/BrainIAC_features/features.csv"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    reidentification_results = "../BrainIAC_reidentification_analysis/matches_diff_dates.csv" 

# Store results for all runs
    accuracies = {'train_acc': [], 'val_acc': [], 'test_acc': [], 'test_acc_re_success': [], 'test_acc_re_failure': []}
    all_classification_reports_re_success = []  # Store all classification reports
    all_classification_reports_re_failure = []  # Store all classification reports
    
    for random_state in random_states:

        print(f"Training with random state: {random_state}")
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

        # Prepare data
        print("Preparing and splitting data")
        X_train, X_val, X_test, y_train, y_val, y_test, class_names, patient_ids_train, patient_ids_val, patient_ids_test, image_vol_names_test = prepare_features_BrainAIC(
            features_csv_path, info_csv, random_state=random_state, test_size=0.2, val_size=0.1, classes=['M', 'F']
        )
    
        print("Searching best parameters")
        best_params = search_best_model(X_train, y_train, patient_ids_train, random_state=random_state, cv_folds=5)

        # best_params = {
        #     'batch_size': 32,
        #     'max_epochs': 120,
        #     'module__dropout_rate': 0.4,
        #     'module__num_hidden_layers': 2
        # }
        print("Using default parameters:", best_params)

        print("The model training started")
        model, results = train_model(X_train, X_val, y_train, y_val, best_params, random_state=random_state)
        print("The model training finished")
        
        plot_training_curves(model, f'./plots/ADNI_BrainAIC_training_curves.png')
        
        # Evaluate on test set
        print("\n" + "="*60)
        print("MAKING PREDICTIONS ON TEST SET")
        print("="*60)
        accuracies['train_acc'].append(results['train_acc'])
        accuracies['val_acc'].append(results['val_acc'])

        test_re_success, test_re_failure = divide_test_set_reidentification(X_test, y_test, image_vol_names_test, reidentification_results)

        test_predictions_re_success = model.predict(test_re_success['X'].astype(np.float32))
        test_probabilities_re_success = model.predict_proba(test_re_success['X'].astype(np.float32))
        test_predictions_re_failure = model.predict(test_re_failure['X'].astype(np.float32))
        test_probabilities_re_failure = model.predict_proba(test_re_failure['X'].astype(np.float32))

        accuracies['test_acc_re_success'].append(np.mean(test_predictions_re_success == test_re_success['y']))
        accuracies['test_acc_re_failure'].append(np.mean(test_predictions_re_failure == test_re_failure['y']))

        print("class distribution for re-identification success: ", np.bincount(test_re_success['y']))
        print("class distribution for re-identification failure: ", np.bincount(test_re_failure['y']))

        # Get classification report as dictionary and store it
        
        report_dict = classification_report(test_re_success['y'], test_predictions_re_success, target_names=class_names, output_dict=True)
        all_classification_reports_re_success.append(report_dict)
        report_dict = classification_report(test_re_failure['y'], test_predictions_re_failure, target_names=class_names, output_dict=True)
        all_classification_reports_re_failure.append(report_dict)

    # Calculate averaged classification report
    averaged_report_re_success = calculate_averaged_classification_report(all_classification_reports_re_success, class_names)
    averaged_report_re_failure = calculate_averaged_classification_report(all_classification_reports_re_failure, class_names)
    print_averaged_classification_report(averaged_report_re_success, class_names, len(random_states), re_identification=True)
    print_averaged_classification_report(averaged_report_re_failure, class_names, len(random_states), re_identification=False)

    
    print(f"\nTraining accuracy: {np.mean(accuracies['train_acc']):.4f} ± {np.std(accuracies['train_acc']):.4f}")
    print(f"Validation accuracy: {np.mean(accuracies['val_acc']):.4f} ± {np.std(accuracies['val_acc']):.4f}")
    print(f"Test accuracy for re-identification success: {np.mean(accuracies['test_acc_re_success']):.4f} ± {np.std(accuracies['test_acc_re_success']):.4f}")
    print(f"Test accuracy for re-identification failure: {np.mean(accuracies['test_acc_re_failure']):.4f} ± {np.std(accuracies['test_acc_re_failure']):.4f}")



if __name__ == "__main__":
    start_time = time.time()
    # train_models_Nyxus(features_group="Shape", random_states=[42, 123, 1234])
    train_models_BrainAIC(random_states=[42, 123, 1234])
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time)/60} minutes")