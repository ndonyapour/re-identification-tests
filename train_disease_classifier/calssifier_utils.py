import os
import sys
import nibabel as nib
import numpy as np

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import StepLR
from skorch.callbacks import EpochScoring, EarlyStopping
from skorch.callbacks import Callback
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reidentification_utils import get_subject_and_date, SHAPE_FEATURES, TEXTURE_FEATURES


def remove_constant_features(X_train: np.ndarray, X_test: np.ndarray, threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove constant and near-constant features using VarianceThreshold.
    
    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix  
        threshold: Variance threshold - features with variance below this are removed
                  (default: 0.0 removes only constant features)
    
    Returns:
        tuple: (X_train_filtered, X_test_filtered, selected_features_mask)
    """
    # Create variance threshold selector
    variance_selector = VarianceThreshold(threshold=threshold)
    
    # Fit on training data and transform both train and test
    X_train_filtered = variance_selector.fit_transform(X_train)
    X_test_filtered = variance_selector.transform(X_test)
    
    # Get mask of selected features
    selected_features_mask = variance_selector.get_support()
    
    # Report results
    n_original = X_train.shape[1]
    n_selected = X_train_filtered.shape[1] 
    n_removed = n_original - n_selected
    
    print(f"Feature filtering results:")
    print(f"  Original features: {n_original}")
    print(f"  Removed constant/low-variance features: {n_removed}")
    print(f"  Remaining features: {n_selected}")
    print(f"  Variance threshold: {threshold}")
    
    if n_removed > 0:
        removed_indices = np.where(~selected_features_mask)[0]
        print(f"  Removed feature indices: {removed_indices[:10]}{'...' if len(removed_indices) > 10 else ''}")
    
    return X_train_filtered, X_test_filtered, selected_features_mask


def patient_level_three_way_split(X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray, 
                                test_size: float = 0.2, val_size: float = 0.2, 
                                random_state: int = 42, remove_constant: bool = True
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/validation/test ensuring no patient appears in multiple sets.
    This prevents the issue where validation and test sets have different difficulty levels.
    
    Args:
        X: Feature matrix
        y: Labels  
        patient_ids: Array of patient IDs corresponding to each sample
        test_size: Proportion of patients to include in test set
        val_size: Proportion of patients to include in validation set
        random_state: Random state for reproducibility
        remove_constant: Whether to remove constant features after scaling
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, patient_ids_train, patient_ids_val, patient_ids_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Get unique patients and their labels for stratified splitting
    unique_patients = np.unique(patient_ids)
    patient_labels = []
    for patient in unique_patients:
        # Get the most common label for this patient (for stratification only)
        patient_mask = patient_ids == patient
        patient_label_counts = np.bincount(y[patient_mask])
        most_common_label = np.argmax(patient_label_counts)
        patient_labels.append(most_common_label)
    
    # First split: separate test patients from train+val patients
    patients_trainval, patients_test = train_test_split(
        unique_patients, test_size=test_size, random_state=random_state, 
        stratify=patient_labels
    )
    
    # Get labels for remaining patients (train+val)
    trainval_labels = []
    for patient in patients_trainval:
        patient_mask = patient_ids == patient
        patient_label_counts = np.bincount(y[patient_mask])
        most_common_label = np.argmax(patient_label_counts)
        trainval_labels.append(most_common_label)
    
    # Second split: separate train and validation patients
    # Adjust val_size to account for the remaining patients
    adjusted_val_size = val_size / (1 - test_size)
    patients_train, patients_val = train_test_split(
        patients_trainval, test_size=adjusted_val_size, 
        random_state=random_state + 1,  # Different seed for second split
        stratify=trainval_labels
    )
    
    # Create masks for each split
    train_mask = np.isin(patient_ids, patients_train)
    val_mask = np.isin(patient_ids, patients_val) 
    test_mask = np.isin(patient_ids, patients_test)
    
    # Split the data - each image keeps its original label
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    
    # Get patient IDs for each image in each set
    patient_ids_train = patient_ids[train_mask]
    patient_ids_val = patient_ids[val_mask]
    patient_ids_test = patient_ids[test_mask]
    
    # Scale features using only training data
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Remove constant features after scaling
    if remove_constant:
        # Fit on training, transform all sets
        variance_selector = VarianceThreshold(threshold=0.0)
        X_train = variance_selector.fit_transform(X_train)
        X_val = variance_selector.transform(X_val)
        X_test = variance_selector.transform(X_test)
        
        selected_features_mask = variance_selector.get_support()
        n_removed = np.sum(~selected_features_mask)
        print(f"Removed {n_removed} constant features")

    # Print split statistics
    print(f"Patient-level 3-way split:")
    print(f"  Train: {len(patients_train)} patients ({len(X_train)} images)")
    print(f"  Val:   {len(patients_val)} patients ({len(X_val)} images)")  
    print(f"  Test:  {len(patients_test)} patients ({len(X_test)} images)")
    
    # Check class distributions
    print(f"Class distributions:")
    for i, split_name in enumerate(['Train', 'Val', 'Test']):
        y_split = [y_train, y_val, y_test][i]
        class_counts = np.bincount(y_split)
        class_props = class_counts / len(y_split)
        print(f"  {split_name}: {class_counts} -> {class_props}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, patient_ids_train, patient_ids_val, patient_ids_test

# Update the prepare_features_Nyxus function to use 3-way split
def prepare_features_Nyxus(features_dir: str, image_dir: str, info_csv: str,
                                   features_group: str = "All", random_state: int = 42, test_size: float = 0.2, 
                                   val_size: float = 0.2, classes: list[str] = ['CN', 'MCI', 'AD']):
    """
    Prepare features with proper 3-way split to ensure fair validation/test comparison.
    """
    # ... existing feature loading code stays the same ...
    metadata = {
        'file_name': [],
        'patient_id': [],
        'scan_date': []
    }
    features_list = []
    file_names = os.listdir(features_dir)
    adni_info_df = pd.read_excel(info_csv, engine="openpyxl")
    labels = []
    patient_ids = []
    for file_name in file_names:
        if file_name.endswith(".csv"):
            file_path = os.path.join(features_dir, file_name)
            image_path = os.path.join(image_dir, file_name.replace(".csv", ".nii.gz"))
            
            # Extract subject ID from filename (not from NIfTI header)
            subject_id, scan_date = get_subject_and_date(image_path)
            
            label = adni_info_df[(adni_info_df['Subject'] == subject_id)]['Group'].values[0]
            if label in classes:
                labels.append(label)
                patient_ids.append(subject_id)

                # load features
                feat_df = pd.read_csv(file_path,
                                    engine='python',  # More robust parsing
                                    encoding='utf-8', sep=',', usecols=lambda x: x != 'Unnamed: 0')  
                
                # To remove specific columns, you can do:
                columns_to_remove = ['intensity_image', 'mask_image', 'ROI_label','Unnamed: 0', 'index', 'id']
                numeric_cols = [col for col in feat_df.select_dtypes(include=[np.number]).columns 
                                if col not in columns_to_remove] 
                
                if features_group == "Shape":
                    numeric_cols = [col for col in numeric_cols if col in SHAPE_FEATURES]
                elif features_group == "Texture":
                    numeric_cols = [col for col in numeric_cols if col in TEXTURE_FEATURES]
                elif features_group == "All":
                    pass
                else:
                    raise ValueError(f"Invalid features group: {features_group}")

                features_list.append(feat_df[numeric_cols].values[0, :])

    # Convert to arrays
    X = np.array(features_list)
    patient_ids = np.array(patient_ids)
    
    # Convert diagnosis labels to numeric
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Use the 3-way split instead of 2-way + random validation
    X_train, X_val, X_test, y_train, y_val, y_test, patient_ids_train, patient_ids_val, patient_ids_test = patient_level_three_way_split(
        X, y, patient_ids, test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, le.classes_, patient_ids_train, patient_ids_val, patient_ids_test


def prepare_features_BrainAIC(features_csv_path: str, info_csv: str, random_state: int = 42, test_size: float = 0.2, val_size: float = 0.2, 
    classes: list[str] = ['CN', 'MCI', 'AD']) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Prepare features for BrainAIC with patient-level train/test split.
    """
    # load features
    features_df = pd.read_csv(features_csv_path)
    adni_info_df = pd.read_excel(info_csv, engine="openpyxl")
    features_list = []
    labels = []
    patient_ids = []
    
    print(f'Loading features for {len(features_df)} images and creating metadata')
    for nifti_path in features_df['image_path']:
        
        # Extract subject ID from filename (not from NIfTI header)
        subject_id, scan_date = get_subject_and_date(nifti_path)
        patient_ids.append(subject_id)
        
        label = adni_info_df[(adni_info_df['Subject'] == subject_id)]['Group'].values[0]
        labels.append(label)
        features_cols = [f for f in features_df.columns if f.startswith('Feature_')]
        features_list.append(features_df[features_df['image_path'] == nifti_path][features_cols].values[0, :])

    # Convert to arrays
    X = np.array(features_list)
    patient_ids = np.array(patient_ids)
    
    # Convert diagnosis labels to numeric
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Use the 3-way split instead of 2-way + random validation
    X_train, X_val, X_test, y_train, y_val, y_test, patient_ids_train, patient_ids_val, patient_ids_test = patient_level_three_way_split(
        X, y, patient_ids, test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, le.classes_, patient_ids_train, patient_ids_val, patient_ids_test

def get_hidden_size(input_size):
    if input_size >= 500:
        return 512
    elif input_size >= 200:
        return 128
    elif input_size >= 100:
        return 128
    elif input_size >= 50:
        return 64
    else:
        return 32


class ADNIClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3, num_hidden_layers=1, num_classes=3):
        """
        Simple neural network for ADNI classification with noise augmentation.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layer (default: 128)
            num_hidden_layers: Number of additional hidden layers
            dropout_rate: Dropout rate for regularization
            noise_factor: Factor for input noise augmentation
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer: input_size -> hidden_size
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.batch_norms.append(nn.BatchNorm1d(hidden_size))
        
        # Additional hidden layers: hidden_size -> hidden_size
        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
        
        # Second to last layer: hidden_size -> hidden_size//2
        self.layers.append(nn.Linear(hidden_size, hidden_size//2))
        self.batch_norms.append(nn.BatchNorm1d(hidden_size//2))
        
        # Final layer: hidden_size//2 -> 3 (no BatchNorm on output)
        self.layers.append(nn.Linear(hidden_size//2, num_classes))
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):      
        # Apply all layers except the last one with BatchNorm + LeakyReLU + Dropout
        for i, (layer, batch_norm) in enumerate(zip(self.layers[:-1], self.batch_norms)):
            x = layer(x)
            x = batch_norm(x)
            x = F.leaky_relu(x, negative_slope=0.01)
            x = self.dropout(x)
        
        # Final layer (no activation, no BatchNorm, no dropout)
        x = self.layers[-1](x)
        return x


def plot_training_curves(classifier, save_path='training_curves.png'):
    """
    Plot training and validation loss/accuracy curves from skorch classifier.
    """
    # Get training history
    history = classifier.history
    
    # Extract metrics
    epochs = range(1, len(history) + 1)
    train_loss = [h['train_loss'] for h in history]
    train_acc = [h.get('train_acc', None) for h in history]
    
    # Check if validation metrics exist
    valid_loss = [h.get('valid_loss', None) for h in history]
    valid_acc = [h.get('valid_acc', None) for h in history]
    
    # Filter out None values
    train_acc = [acc for acc in train_acc if acc is not None]
    valid_loss = [loss for loss in valid_loss if loss is not None]
    valid_acc = [acc for acc in valid_acc if acc is not None]
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Loss curves
    axes[0].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    if valid_loss:
        valid_epochs = range(1, len(valid_loss) + 1)
        axes[0].plot(valid_epochs, valid_loss, 'r-', label='Validation Loss', linewidth=2)
    
    axes[0].set_title('Patient-Aware Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    if train_acc:
        acc_epochs = range(1, len(train_acc) + 1)
        axes[1].plot(acc_epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
        
        if valid_acc:
            valid_acc_epochs = range(1, len(valid_acc) + 1)
            axes[1].plot(valid_acc_epochs, valid_acc, 'r-', label='Validation Accuracy', linewidth=2)
        
        axes[1].set_title('Patient-Aware Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No accuracy data available\nOnly loss was tracked', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('Training Information')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print training summary
    print(f"\n{'='*50}")
    print("PATIENT-AWARE TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"Total epochs: {len(history)}")
    print(f"Final training loss: {train_loss[-1]:.4f}")
    if train_acc:
        print(f"Final training accuracy: {train_acc[-1]:.4f}")
    if valid_loss:
        print(f"Final validation loss: {valid_loss[-1]:.4f}")
    if valid_acc:
        print(f"Final validation accuracy: {valid_acc[-1]:.4f}")




def search_best_model(X_train, y_train, patient_ids_train, random_state: int = 42, cv_folds=3):
    """
    Train with patient-aware cross-validation - improved approach.
    """
    torch.manual_seed(random_state)
    input_size = X_train.shape[1]
    # Convert data types
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.int64)
    hidden_size = get_hidden_size(input_size)

    # Add class weights for balanced training
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_tensor = torch.FloatTensor(class_weights)

    lr_policy = LRScheduler(StepLR, step_size=20, gamma=0.7)
    
    train_acc = EpochScoring(scoring='balanced_accuracy', on_train=True, name='train_acc', lower_is_better=False)
    early_stopping = EarlyStopping(
        monitor='valid_loss',
        patience=20,  # Increase patience
        threshold=0.005,
        lower_is_better=True
    )
    
    classifier = NeuralNetClassifier(
        ADNIClassifier,
        module__input_size=input_size,
        module__hidden_size=hidden_size,
        module__num_classes=len(np.unique(y_train)),
        criterion=nn.CrossEntropyLoss,
        criterion__weight=class_weight_tensor,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.0001,
        optimizer__weight_decay=1e-4,
        train_split=False,  # No validation split during GridSearchCV
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=0,
        callbacks=[lr_policy, train_acc],
    )
    # Reduced parameter space to prevent overtraining
    params = {
        'module__num_hidden_layers': [1, 2],
        'module__dropout_rate': [0.4, 0.5],  # Increase dropout options
        'max_epochs': [50, 80, 120],  # Reduce max epochs to prevent overtraining
        'batch_size': [32, 64],
    }

    # Use GroupKFold to ensure patients don't appear in both train and validation
    group_kfold = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    gs = GridSearchCV(
        classifier, 
        params, 
        refit=False,
        cv=group_kfold,  # This provides patient-aware validation
        scoring='balanced_accuracy', 
        verbose=2,
        n_jobs=1  # Reduce parallelism to avoid memory issues
    )
    
    # Pass patient_ids as groups parameter
    gs.fit(X_train, y_train, groups=patient_ids_train)
    
    print(f"Best parameters found: {gs.best_params_}")
    print(f"Best CV score: {gs.best_score_:.4f}")
    
    return gs.best_params_

    
def create_validation_split(val_dataset):
    """
    Create a validation split function that can be pickled.
    This replaces the lambda function that causes serialization issues.
    """
    def validation_split(dataset, y, **kwargs):
        return dataset, val_dataset
    return validation_split


def train_model(X_train, X_val, y_train, y_val, best_params, random_state: int = 42):
    """
    Train model using explicit validation set instead of random splits.
    This ensures fair comparison between validation and test performance.
    """
    torch.manual_seed(random_state)
    input_size = X_train.shape[1]
    
    # Convert data types
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    hidden_size = get_hidden_size(input_size)

    # Add class weights for balanced training
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_tensor = torch.FloatTensor(class_weights)
    
    print(f"Class weights: {dict(enumerate(class_weights))}")

    # Learning rate scheduler
    lr_policy = LRScheduler(StepLR, step_size=20, gamma=0.7)
    
    # Callbacks for monitoring
    train_acc = EpochScoring(scoring='balanced_accuracy', on_train=True, name='train_acc', lower_is_better=False)
    valid_acc = EpochScoring(scoring='balanced_accuracy', on_train=False, name='valid_acc', lower_is_better=False)
    
    # Early stopping based on validation loss
    early_stopping = EarlyStopping(
        monitor='valid_loss',
        patience=15,
        threshold=0.001,
        lower_is_better=True
    )
    
    # Create explicit validation dataset
    from torch.utils.data import TensorDataset
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    classifier = NeuralNetClassifier(
        ADNIClassifier,
        module__input_size=input_size,
        module__hidden_size=hidden_size,
        module__dropout_rate=best_params['module__dropout_rate'],
        module__num_classes=len(np.unique(y_train)),
        module__num_hidden_layers=best_params.get('module__num_hidden_layers', 1),
        max_epochs=best_params['max_epochs'],
        batch_size=best_params['batch_size'],
        criterion=nn.CrossEntropyLoss,
        criterion__weight=class_weight_tensor,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.0001,
        optimizer__weight_decay=1e-4,
        train_split=create_validation_split(val_dataset),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[lr_policy, train_acc, valid_acc, early_stopping],
        verbose=0
    )

    classifier.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_pred = classifier.predict(X_val)
    val_acc = np.mean(val_pred == y_val)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED WITH EXPLICIT VALIDATION")
    print(f"{'='*60}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    print(f"Training epochs: {len(classifier.history)}")
    
    # Check for overfitting
    final_train_acc = classifier.history[-1]['train_acc']
    if final_train_acc > val_acc + 0.1:
        print("⚠️  WARNING: Potential overfitting detected (train acc >> val acc)")
    
    return classifier, {'train_acc': final_train_acc, 'val_acc': val_acc}



def calculate_patient_level_results(y_pred, y_true, patient_ids, class_names=None):
    """
    Calculate patient-level results from image-level predictions using majority voting.
    
    Args:
        y_pred: Image-level predictions
        y_true: Image-level true labels
        patient_ids: Patient IDs corresponding to each image
        class_names: Names of classes (optional)
        
    Returns:
        dict: Patient-level metrics
    """
    from collections import defaultdict
    from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
    
    # Group images by patient
    patient_groups = defaultdict(list)
    for i, pid in enumerate(patient_ids):
        patient_groups[pid].append(i)
    
    patient_true_labels = []
    patient_pred_labels = []
    
    # Aggregate for each patient using majority voting
    for patient_id, image_indices in patient_groups.items():
        # True label: most common label for this patient
        patient_true_label = np.bincount(y_true[image_indices]).argmax()
        
        # Predicted label: majority vote of predictions
        patient_pred_label = np.bincount(y_pred[image_indices]).argmax()
        
        patient_true_labels.append(patient_true_label)
        patient_pred_labels.append(patient_pred_label)
    
    patient_true = np.array(patient_true_labels)
    patient_pred = np.array(patient_pred_labels)
    
    # Calculate metrics
    patient_balanced_acc = balanced_accuracy_score(patient_true, patient_pred)
    patient_macro_f1 = f1_score(patient_true, patient_pred, average='macro')
    
    # Image-level metrics for comparison
    image_balanced_acc = balanced_accuracy_score(y_true, y_pred)
    image_macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Print results
    print(f"\nPatient-Level Results:")
    print(f"  Total patients: {len(patient_true)}")
    print(f"  Total images: {len(y_true)}")
    print(f"  Patient Balanced Accuracy: {patient_balanced_acc:.4f}")
    print(f"  Patient Macro F1: {patient_macro_f1:.4f}")
    print(f"  Image Balanced Accuracy: {image_balanced_acc:.4f}")
    print(f"  Image Macro F1: {image_macro_f1:.4f}")
    
    if class_names is not None and len(class_names) > 0:
        print(f"\nPatient-Level Classification Report:")
        print(classification_report(patient_true, patient_pred, target_names=class_names))
    
    return {
        'patient_balanced_accuracy': patient_balanced_acc,
        'patient_macro_f1': patient_macro_f1,
        'image_balanced_accuracy': image_balanced_acc,
        'image_macro_f1': image_macro_f1,
        'n_patients': len(patient_true),
        'n_images': len(y_true)
    }



def apply_smote_resampling(X_train: np.ndarray, y_train: np.ndarray,
                          method: str = 'smote', 
                          random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE or other resampling techniques to handle class imbalance.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        method: 'smote', 'adasyn', 'smoteenn', 'borderline', 'svmsmote'
        random_state: Random state for reproducibility
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print(f"Applying {method.upper()} resampling for class imbalance...")
    
    # Check original class distribution
    unique_classes, counts = np.unique(y_train, return_counts=True)
    print(f"Original class distribution:")
    for cls, count in zip(unique_classes, counts):
        percentage = (count / len(y_train)) * 100
        print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
    
    # Calculate imbalance ratio
    max_count = np.max(counts)
    min_count = np.min(counts)
    imbalance_ratio = max_count / min_count
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # If classes are already balanced, skip resampling
    if imbalance_ratio < 1.5:
        print("Classes are already well balanced. Skipping resampling.")
        return X_train, y_train
    
    try:
        # Determine appropriate k_neighbors based on minority class size
        min_class_size = np.min(counts)
        k_neighbors = min(5, min_class_size - 1)  # Ensure k < minority class size
        
        if k_neighbors < 1:
            print(f"Warning: Minority class too small ({min_class_size} samples). Cannot apply SMOTE.")
            return X_train, y_train
        
        print(f"Using k_neighbors={k_neighbors} for SMOTE")
        
        if method == 'smote':
            from imblearn.over_sampling import SMOTE
            resampler = SMOTE(
                random_state=random_state,
                k_neighbors=k_neighbors
            )
            
        elif method == 'adasyn':
            from imblearn.over_sampling import ADASYN
            resampler = ADASYN(
                random_state=random_state,
                n_neighbors=k_neighbors
            )
            
        elif method == 'borderline':
            from imblearn.over_sampling import BorderlineSMOTE
            resampler = BorderlineSMOTE(
                random_state=random_state,
                k_neighbors=k_neighbors
            )
            
        elif method == 'svmsmote':
            from imblearn.over_sampling import SVMSMOTE
            resampler = SVMSMOTE(
                random_state=random_state,
                k_neighbors=k_neighbors
            )
            
        elif method == 'smoteenn':
            from imblearn.combine import SMOTEENN
            resampler = SMOTEENN(
                random_state=random_state,
                smote__k_neighbors=k_neighbors
            )
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        # Apply resampling
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        
        # Check new class distribution
        unique_classes_new, counts_new = np.unique(y_resampled, return_counts=True)
        print(f"Resampled class distribution:")
        for cls, count in zip(unique_classes_new, counts_new):
            percentage = (count / len(y_resampled)) * 100
            print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
        
        print(f"Dataset size change: {len(X_train)} -> {len(X_resampled)} samples "
              f"({((len(X_resampled) - len(X_train)) / len(X_train) * 100):+.1f}%)")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        print(f"Error applying {method}: {str(e)}")
        print("Falling back to original data without resampling.")
        return X_train, y_train


def print_averaged_classification_report(averaged_report, class_names, n_runs):
    """
    Print averaged classification report in a readable format.
    
    Args:
        averaged_report: Averaged report dictionary
        class_names: List of class names
        n_runs: Number of runs averaged
    """
    print(f"\n{'='*80}")
    print(f"📊 AVERAGED CLASSIFICATION REPORT ACROSS {n_runs} RUNS")
    print(f"{'='*80}")
    
    # Header
    print(f"{'':>12} {'precision':>12} {'recall':>12} {'f1-score':>12} {'support':>12}")
    print(f"{'':>12} {'mean ± std':>12} {'mean ± std':>12} {'mean ± std':>12} {'mean ± std':>12}")
    print("-" * 80)
    
    # Per-class results
    for class_name in class_names:
        if class_name in averaged_report:
            precision = averaged_report[class_name]['precision']
            recall = averaged_report[class_name]['recall']
            f1 = averaged_report[class_name]['f1-score']
            support = averaged_report[class_name]['support']
            
            print(f"{class_name:>12} "
                  f"{precision['mean']:.2f}±{precision['std']:.2f}    "
                  f"{recall['mean']:.2f}±{recall['std']:.2f}    "
                  f"{f1['mean']:.2f}±{f1['std']:.2f}    "
                  f"{support['mean']:.0f}±{support['std']:.0f}")
    
    print()
    
    # Overall accuracy
    if 'accuracy' in averaged_report:
        accuracy = averaged_report['accuracy']
        print(f"{'accuracy':>12} {'':>12} {'':>12} "
              f"{accuracy['mean']:.2f}±{accuracy['std']:.2f}    "
              f"{sum(averaged_report[cls]['support']['mean'] for cls in class_names):.0f}")
    
    # Macro and weighted averages
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in averaged_report:
            precision = averaged_report[avg_type]['precision']
            recall = averaged_report[avg_type]['recall']
            f1 = averaged_report[avg_type]['f1-score']
            
            print(f"{avg_type:>12} "
                  f"{precision['mean']:.2f}±{precision['std']:.2f}    "
                  f"{recall['mean']:.2f}±{recall['std']:.2f}    "
                  f"{f1['mean']:.2f}±{f1['std']:.2f}    "
                  f"{sum(averaged_report[cls]['support']['mean'] for cls in class_names):.0f}")

def print_simple_averaged_report(all_reports, class_names, n_runs):
    """
    Print a simple averaged classification report.
    """
    print(f"\n{'='*60}")
    print(f"📊 AVERAGED RESULTS ACROSS {n_runs} RUNS")
    print(f"{'='*60}")
    
    # Calculate averages for key metrics
    metrics = {}
    
    # Overall accuracy
    accuracies = [report['accuracy'] for report in all_reports]
    metrics['accuracy'] = {'mean': np.mean(accuracies), 'std': np.std(accuracies)}
    
    # Macro averages
    macro_precision = [report['macro avg']['precision'] for report in all_reports]
    macro_recall = [report['macro avg']['recall'] for report in all_reports]
    macro_f1 = [report['macro avg']['f1-score'] for report in all_reports]
    
    metrics['macro_precision'] = {'mean': np.mean(macro_precision), 'std': np.std(macro_precision)}
    metrics['macro_recall'] = {'mean': np.mean(macro_recall), 'std': np.std(macro_recall)}
    metrics['macro_f1'] = {'mean': np.mean(macro_f1), 'std': np.std(macro_f1)}
    
    # Per-class F1 scores
    for class_name in class_names:
        class_f1 = [report[class_name]['f1-score'] for report in all_reports if class_name in report]
        metrics[f'{class_name}_f1'] = {'mean': np.mean(class_f1), 'std': np.std(class_f1)}
    
    # Print results
    print(f"Overall Accuracy:     {metrics['accuracy']['mean']:.4f} ± {metrics['accuracy']['std']:.4f}")
    print(f"Macro Precision:      {metrics['macro_precision']['mean']:.4f} ± {metrics['macro_precision']['std']:.4f}")
    print(f"Macro Recall:         {metrics['macro_recall']['mean']:.4f} ± {metrics['macro_recall']['std']:.4f}")
    print(f"Macro F1-Score:       {metrics['macro_f1']['mean']:.4f} ± {metrics['macro_f1']['std']:.4f}")
    
    print(f"\nPer-Class F1-Scores:")
    for class_name in class_names:
        key = f'{class_name}_f1'
        if key in metrics:
            print(f"  {class_name}: {metrics[key]['mean']:.4f} ± {metrics[key]['std']:.4f}")



def calculate_averaged_classification_report(all_reports, class_names):
    """
    Calculate averaged classification report across multiple runs.
    
    Args:
        all_reports: List of classification report dictionaries
        class_names: List of class names
        
    Returns:
        Dictionary with averaged metrics and standard deviations
    """
    import numpy as np
    
    averaged_report = {}
    
    # Metrics to average
    metrics = ['precision', 'recall', 'f1-score']
    average_types = ['macro avg', 'weighted avg']
    
    # Initialize storage for each class
    for class_name in class_names:
        averaged_report[class_name] = {}
        for metric in metrics:
            values = [report[class_name][metric] for report in all_reports if class_name in report]
            averaged_report[class_name][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        # Support (should be the same across runs, but let's average anyway)
        supports = [report[class_name]['support'] for report in all_reports if class_name in report]
        averaged_report[class_name]['support'] = {
            'mean': np.mean(supports),
            'std': np.std(supports),
            'values': supports
        }
    
    # Average the macro and weighted averages
    for avg_type in average_types:
        averaged_report[avg_type] = {}
        for metric in metrics:
            values = [report[avg_type][metric] for report in all_reports if avg_type in report]
            averaged_report[avg_type][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
    
    # Overall accuracy
    accuracy_values = [report['accuracy'] for report in all_reports]
    averaged_report['accuracy'] = {
        'mean': np.mean(accuracy_values),
        'std': np.std(accuracy_values),
        'values': accuracy_values
    }
    
    return averaged_report