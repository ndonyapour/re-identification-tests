import os
import sys
import nibabel as nib
import numpy as np

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reidentification_utils import get_subject_and_date, SHAPE_FEATURES, TEXTURE_FEATURES

def patient_level_train_test_split(X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray, 
                                 test_size: float = 0.2, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data ensuring no patient appears in both train and test sets.
    
    Args:
        X: Feature matrix
        y: Labels
        patient_ids: Array of patient IDs corresponding to each sample
        test_size: Proportion of patients to include in test set
        random_state: Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, patient_ids_train, patient_ids_test)
    """
    # Get unique patients and their labels for stratified splitting
    unique_patients = np.unique(patient_ids)
    patient_labels = []
    for patient in unique_patients:
        # Get the most common label for this patient (for stratification only)
        patient_mask = patient_ids == patient
        patient_label_counts = np.bincount(y[patient_mask])
        most_common_label = np.argmax(patient_label_counts)
        patient_labels.append(most_common_label)
    
    # Split patients (not images) into train/test
    patients_train, patients_test = train_test_split(
        unique_patients, test_size=test_size, random_state=random_state, 
        stratify=patient_labels
    )
    
    # Create train/test masks based on patient assignment
    # If patient is in training set, ALL their images go to training
    # If patient is in test set, ALL their images go to test
    train_mask = np.isin(patient_ids, patients_train)
    test_mask = np.isin(patient_ids, patients_test)
    
    # Split the data - each image keeps its original label
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # Get patient IDs for each image in train and test sets
    patient_ids_train = patient_ids[train_mask]  # Patient ID for each training image
    patient_ids_test = patient_ids[test_mask]    # Patient ID for each test image
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Patient-level split: {len(patients_train)} patients in train, {len(patients_test)} patients in test")
    print(f"Image-level split: {len(X_train)} images in train, {len(X_test)} images in test")
    
    return X_train, X_test, y_train, y_test, patient_ids_train, patient_ids_test

def prepare_features_Nyxus(features_dir: str, image_dir: str, info_csv: str,
 features_group: str = "All", test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, np.ndarray, np.ndarray]:
    """
    Find the closest neighbors for each image in the dataset.
    
    Args:
        features_dir: Directory containing feature files
        image_data_csv: Path to the image data CSV file
        n_neighbors: Number of neighbors to find
        standardize: Whether to standardize features
        features_group: Which features to use ("All", "Shape", or "Texture")
        exclude_same_date: Whether to exclude matches from same date for same patient
        distance_threshold: Only keep matches with distance <= threshold (-1 to disable)
        output_dir: Directory to save matches.csv (if None, don't save)
    """

    
    # load metadata
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
            patient_ids.append(subject_id)
            
            label = adni_info_df[(adni_info_df['Subject'] == subject_id)]['Group'].values[0]
            labels.append(label)

            # load features
            feat_df = pd.read_csv(file_path,
                                engine='python',  # More robust parsing
                                encoding='utf-8', sep=',', usecols=lambda x: x != 'Unnamed: 0')  
            
            # To remove specific columns, you can do:
            columns_to_remove = ['intensity_image', 'mask_image', 'ROI_label','Unnamed: 0', 'index', 'id']  # add any column names you want to remove
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
    
    # Use the shared patient-level splitting function
    X_train, X_test, y_train, y_test, patient_ids_train, patient_ids_test = patient_level_train_test_split(
        X, y, patient_ids, test_size=test_size, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, le.classes_, patient_ids_train, patient_ids_test


def prepare_features_BrainAIC(features_csv_path: str, info_csv: str, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
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
    
    # Use the shared patient-level splitting function
    X_train, X_test, y_train, y_test, patient_ids_train, patient_ids_test = patient_level_train_test_split(
        X, y, patient_ids, test_size=test_size, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, le.classes_, patient_ids_train, patient_ids_test
    

# def plot_results(y_true, y_pred, class_names):
#     """
#     Plot confusion matrix and print classification report.
#     """
#     # Print classification report
#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred, target_names=class_names))
    
#     # Plot confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names,
#                 yticklabels=class_names)
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     plt.savefig('confusion_matrix.png')
#     plt.close()

def get_hidden_size(input_size):
    if input_size >= 500:
        return 512
    elif input_size >= 200:
        return 256
    elif input_size >= 100:
        return 128
    elif input_size >= 50:
        return 64
    else:
        return 32


class ADNIClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3, num_hidden_layers=1):
        """
        Simple neural network for ADNI classification.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layer (default: 128)
            num_hidden_layers: Number of additional hidden layers
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
        self.layers.append(nn.Linear(hidden_size//2, 3))
        
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


class PatientAwareTrainSplit:
    """
    Patient-aware train/validation splitter for skorch.
    """
    def __init__(self, patient_ids, validation_size=0.2, random_state=42):
        self.patient_ids = patient_ids
        self.validation_size = validation_size
        self.random_state = random_state
        
    def __call__(self, dataset, y, **fit_params):
        """
        Split dataset into train and validation ensuring no patient overlap.
        """
        import numpy as np
        from sklearn.model_selection import train_test_split
        
        # Get the actual indices being used in this fold
        if hasattr(dataset, 'indices'):
            # This is a Subset from GridSearchCV
            fold_indices = dataset.indices
            fold_patient_ids = self.patient_ids[fold_indices]
        else:
            # This is the full dataset
            fold_indices = np.arange(len(y))
            fold_patient_ids = self.patient_ids
        
        # Ensure dimensions match
        assert len(fold_patient_ids) == len(y), f"Patient IDs ({len(fold_patient_ids)}) and labels ({len(y)}) must have same length"
        
        # Get unique patients and their labels for stratified splitting
        unique_patients = np.unique(fold_patient_ids)
        patient_labels = []
        
        for patient in unique_patients:
            # Get the most common label for this patient (for stratification)
            patient_mask = fold_patient_ids == patient
            patient_label_counts = np.bincount(y[patient_mask])
            most_common_label = np.argmax(patient_label_counts)
            patient_labels.append(most_common_label)
        
        # Split patients (not images) into train/validation
        patients_train, patients_valid = train_test_split(
            unique_patients, 
            test_size=self.validation_size, 
            random_state=self.random_state,
            stratify=patient_labels
        )
        
        # Create train/validation masks based on patient assignment
        train_mask = np.isin(fold_patient_ids, patients_train)
        valid_mask = np.isin(fold_patient_ids, patients_valid)
        
        # Create indices for train and validation (relative to the current fold)
        train_indices = np.where(train_mask)[0]
        valid_indices = np.where(valid_mask)[0]
        
        print(f"Patient-aware split: {len(patients_train)} patients ({len(train_indices)} images) for training")
        print(f"Patient-aware split: {len(patients_valid)} patients ({len(valid_indices)} images) for validation")
        
        # Create train and validation datasets using the correct import
        try:
            from skorch.dataset import Subset
        except ImportError:
            # Fallback: use torch.utils.data.Subset
            from torch.utils.data import Subset
        
        train_dataset = Subset(dataset, train_indices)
        valid_dataset = Subset(dataset, valid_indices)
        
        return train_dataset, valid_dataset

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

def search_best_model(X_train, y_train, patient_ids_train, input_size):
    """
    Train with patient-aware cross-validation - simplified approach.
    """
    torch.manual_seed(42)
    
    # Convert data types
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.int64)
    hidden_size = get_hidden_size(input_size)

    lr_policy = LRScheduler(StepLR, step_size=15, gamma=0.5)
    
    train_acc = EpochScoring(scoring='balanced_accuracy', on_train=True, name='train_acc', lower_is_better=False)
    early_stopping = EarlyStopping(
        monitor='valid_loss',
        patience=15,  # Stop if no improvement for 15 epochs
        threshold=0.01,
        lower_is_better=True
    )
    classifier = NeuralNetClassifier(
        ADNIClassifier,
        module__input_size=input_size,
        module__hidden_size=hidden_size,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=.005,
        train_split=False,  # No validation split during GridSearchCV
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=0,
        callbacks=[lr_policy],
    )
    
    params = {
        'module__num_hidden_layers': [1, 2],
        'module__dropout_rate': [0.2, 0.3],#[0.4, 0.5],
        'max_epochs': [100, 200], #50, 100
        'batch_size': [64, 128],
    }

    # Use GroupKFold to ensure patients don't appear in both train and validation
    group_kfold = GroupKFold(n_splits=3)
    
    gs = GridSearchCV(
        classifier, 
        params, 
        refit=False,
        cv=group_kfold,  # This provides patient-aware validation
        scoring='balanced_accuracy', 
        verbose=2,
        n_jobs=-1
    )
    
    # Pass patient_ids as groups parameter
    gs.fit(X_train, y_train, groups=patient_ids_train)
    return gs.best_params_


def train_model(X_train, y_train, patient_ids_train, best_params, input_size):
    """
    Train final model with patient-aware validation split for detailed curves.
    """
    torch.manual_seed(42)
    
    # Convert data types
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.int64)
    hidden_size = get_hidden_size(input_size)

    lr_policy = LRScheduler(StepLR, step_size=15, gamma=0.5)
    
    # Create patient-aware train/validation splitter
    patient_aware_split = PatientAwareTrainSplit(
        patient_ids=patient_ids_train,
        validation_size=0.2,
        random_state=42
    )
    
    # Add standard accuracy callbacks
    train_acc = EpochScoring(scoring='balanced_accuracy', on_train=True, name='train_acc', lower_is_better=False)
    valid_acc = EpochScoring(scoring='balanced_accuracy', on_train=False, name='valid_acc', lower_is_better=False)
    
    # Add early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='valid_loss',
        patience=15,  # Stop if no improvement for 15 epochs
        threshold=0.01,
        lower_is_better=True
    )
    
    classifier = NeuralNetClassifier(
        ADNIClassifier,
        module__input_size=input_size,
        module__hidden_size=hidden_size,
        module__dropout_rate=best_params['module__dropout_rate'],
        module__num_hidden_layers=best_params['module__num_hidden_layers'],
        max_epochs=best_params['max_epochs'],
        batch_size=best_params['batch_size'],
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=1e-3,  # Add L2 regularization
        lr=0.0005,  # Lower learning rate
        train_split=patient_aware_split,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[lr_policy, train_acc, valid_acc],  # Add early stopping
        verbose=0
    )
    
    # Train the model
    classifier.fit(X_train, y_train)
    return classifier

    