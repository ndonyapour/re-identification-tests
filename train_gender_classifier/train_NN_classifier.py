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
    

    accuracies = {'train_acc': [], 'val_acc': [], 'test_acc': []}
    all_classification_reports = []  # Store all classification reports
    for random_state in random_states:

        print(f"Training with random state: {random_state}")
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

        # Prepare data
        print("Preparing and splitting data")
        X_train, X_val, X_test, y_train, y_val, y_test, class_names, patient_ids_train, patient_ids_val, patient_ids_test = prepare_features_BrainAIC(
            features_csv_path, info_csv, random_state=random_state, test_size=0.2, val_size=0.1, classes=['M', 'F']
        )
    
        print("Searching best parameters")
        best_params = search_best_model(X_train, y_train, patient_ids_train, random_state=random_state, cv_folds=3)

        print("Using default parameters:", best_params)

        print("The model training started")
        model, results = train_model(X_train, X_val, y_train, y_val, best_params, random_state=random_state)
        print("The model training finished")
        
        plot_training_curves(model, f'./plots/ADNI_BrainAIC_training_curves.png')
        
        # Evaluate on test set
        print("\n" + "="*60)
        print("MAKING PREDICTIONS ON TEST SET")
        print("="*60)
        X_test = X_test.astype(np.float32)
        test_predictions = model.predict(X_test)
        test_probabilities = model.predict_proba(X_test)

        test_accuracy = np.mean(test_predictions == y_test)
        
        print(f"Test set size: {len(X_test)} images from {len(np.unique(patient_ids_test))} patients")
        print(f"Predictions shape: {test_predictions.shape}")
        print(f"Probabilities shape: {test_probabilities.shape}")

        print("\n" + "="*50)
        print("TEST SET EVALUATION")
        print("="*50)

        accuracies['train_acc'].append(results['train_acc'])
        accuracies['val_acc'].append(results['val_acc'])
        accuracies['test_acc'].append(test_accuracy)
        print("class distribution: ", np.bincount(y_test))

        # Get classification report as dictionary and store it
        from sklearn.metrics import classification_report
        report_dict = classification_report(y_test, test_predictions, target_names=class_names, output_dict=True)
        all_classification_reports.append(report_dict)

    # Calculate averaged classification report
    averaged_report = calculate_averaged_classification_report(all_classification_reports, class_names)
    print_averaged_classification_report(averaged_report, class_names, len(random_states))
    
    print(f"\nTraining accuracy: {np.mean(accuracies['train_acc']):.4f} ± {np.std(accuracies['train_acc']):.4f}")
    print(f"Validation accuracy: {np.mean(accuracies['val_acc']):.4f} ± {np.std(accuracies['val_acc']):.4f}")
    print(f"Test accuracy: {np.mean(accuracies['test_acc']):.4f} ± {np.std(accuracies['test_acc']):.4f}")


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
    train_models_Nyxus(features_group="Shape", random_states=[42, 123, 1234])
    # train_models_BrainAIC(random_states=[42, 123, 1234])