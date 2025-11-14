import os
from traceback import print_tb
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Suppress PyRadiomics warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger('radiomics').setLevel(logging.ERROR)

class PyRadiomicsExtractor:
    """
    PyRadiomics feature extractor that generates radiomics features 
    equivalent to Nyxus features for both 2D and 3D images.
    """
    
    def __init__(self, 
                 dimension: str = '3D',
                 bin_width: int = 5,
                 voxel_array_shift: int = 0,
                 normalize: bool = False,
                 normalize_scale: int = 100,
                 interpolator: str = 'sitkBSpline',
                 resample_pixel_spacing: Optional[List[float]] = None):
        """
        Initialize PyRadiomics feature extractor.
        
        Args:
            dimension: '2D' or '3D' for feature extraction
            bin_width: Bin width for discretization (default: 25)
            voxel_array_shift: Shift applied to voxel intensities
            normalize: Whether to normalize images
            normalize_scale: Scale for normalization
            interpolator: Interpolation method for resampling
            resample_pixel_spacing: Target pixel spacing for resampling
        """
        self.dimension = dimension
        self.bin_width = bin_width
        self.voxel_array_shift = voxel_array_shift
        self.normalize = normalize
        self.normalize_scale = normalize_scale
        self.interpolator = interpolator
        self.resample_pixel_spacing = resample_pixel_spacing
        
        # Initialize feature extractor
        self.extractor = self._setup_extractor()
    
    def _setup_extractor(self) -> featureextractor.RadiomicsFeatureExtractor:
        """Setup PyRadiomics feature extractor with appropriate settings."""
        
        # Initialize extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
        
        # Configure settings based on dimension
        if self.dimension == '2D':
            settings = {
                'binWidth': self.bin_width,
                'voxelArrayShift': self.voxel_array_shift,
                'normalize': self.normalize,
                'normalizeScale': self.normalize_scale,
                'interpolator': self.interpolator,
                'force2D': True,
                'force2Ddimension': 0,  # Extract features from axial slices
                'correctMask': True,
                'additionalInfo': False
            }
        else:  # 3D
            settings = {
                'binWidth': self.bin_width,
                'voxelArrayShift': self.voxel_array_shift,
                'normalize': self.normalize,
                'normalizeScale': self.normalize_scale,
                'interpolator': self.interpolator,
                'correctMask': True,
                'additionalInfo': False
            }
        
        # Add resampling if specified
        if self.resample_pixel_spacing:
            settings['resampledPixelSpacing'] = self.resample_pixel_spacing
        
        # Apply settings
        for key, value in settings.items():
            extractor.settings[key] = value
        
        # Enable all feature classes
        extractor.enableAllFeatures()
        
        return extractor
    
    def extract_features_from_files(self, 
                                   image_path: str, 
                                   mask_path: str,
                                   label: int = 1) -> Dict[str, float]:
        """
        Extract features from image and mask files.
        
        Args:
            image_path: Path to the image file
            mask_path: Path to the mask file
            label: Label value in mask to extract features from
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Extract features using PyRadiomics
            features = self.extractor.execute(image_path, mask_path, label=label)
            
            # Debug: Print all returned features
            print(f"Total features returned by PyRadiomics: {len(features)}")
            print(f"Sample feature types: {[(k, type(v).__name__) for k, v in list(features.items())[:20]]}")
            
            # Filter out non-numeric features (metadata)
            numeric_features = {}
            non_numeric_count = 0
            nan_count = 0
            
            non_numeric_features = []
            for key, value in features.items():
                # Handle numpy arrays (extract scalar if single element)
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = value.item()  # Convert single-element array to scalar
                    else:
                        non_numeric_count += 1
                        non_numeric_features.append(key)
                        continue  # Skip multi-element arrays
                
                if isinstance(value, (int, float, np.number)):
                    # Check for NaN values
                    if np.isnan(value):
                        nan_count += 1
                        continue  # Skip NaN values
                    # Remove 'original_' prefix from feature names
                    clean_key = key.replace('original_', '')
                    numeric_features[clean_key] = float(value)
                else:
                    non_numeric_count += 1
                    non_numeric_features.append(key)
            
            for feature in non_numeric_features:
                print(f"Non-numeric feature: {feature} with value: {features[feature]}")
            import pdb; pdb.set_trace()
            print(f"Non-numeric features filtered out: {non_numeric_count}")
            print(f"NaN features filtered out: {nan_count}")
            print(f"Valid numeric features: {len(numeric_features)}")
            
            return numeric_features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def extract_features_from_arrays(self, 
                                    image_array: np.ndarray, 
                                    mask_array: np.ndarray,
                                    spacing: Optional[Tuple[float, ...]] = None,
                                    origin: Optional[Tuple[float, ...]] = None,
                                    label: int = 1) -> Dict[str, float]:
        """
        Extract features from numpy arrays.
        
        Args:
            image_array: Image as numpy array
            mask_array: Mask as numpy array
            spacing: Pixel/voxel spacing
            origin: Image origin
            label: Label value in mask to extract features from
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Convert numpy arrays to SimpleITK images
            image_sitk = sitk.GetImageFromArray(image_array)
            mask_sitk = sitk.GetImageFromArray(mask_array.astype(np.uint8))
            
            # Set spacing and origin if provided
            if spacing:
                image_sitk.SetSpacing(spacing)
                mask_sitk.SetSpacing(spacing)
            if origin:
                image_sitk.SetOrigin(origin)
                mask_sitk.SetOrigin(origin)
            
            # Extract features
            features = self.extractor.execute(image_sitk, mask_sitk, label=label)
            
            # Filter out non-numeric features
            numeric_features = {}
            for key, value in features.items():
                if isinstance(value, (int, float, np.number)):
                    clean_key = key.replace('original_', '')
                    numeric_features[clean_key] = float(value)
            
            return numeric_features
            
        except Exception as e:
            print(f"Error extracting features from arrays: {str(e)}")
            return {}
    
    def batch_extract_features(self, 
                              image_paths: List[str], 
                              mask_paths: List[str],
                              output_csv: Optional[str] = None,
                              patient_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract features from multiple image-mask pairs.
        
        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            output_csv: Optional path to save results as CSV
            patient_ids: Optional list of patient IDs
            
        Returns:
            DataFrame with extracted features
        """
        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of images and masks must be equal")
        
        results = []
        
        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            # Extract features
            features = self.extract_features_from_files(img_path, mask_path)
            
            if features:
                # Add metadata
                features['image_path'] = img_path
                features['mask_path'] = mask_path
                if patient_ids:
                    features['patient_id'] = patient_ids[i]
                else:
                    features['patient_id'] = os.path.basename(img_path).split('.')[0]
                
                results.append(features)
            else:
                print(f"Failed to extract features from {img_path}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns to put metadata first
        metadata_cols = ['patient_id', 'image_path', 'mask_path']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        df = df[metadata_cols + sorted(feature_cols)]
        
        # Save to CSV if specified
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")
        
        return df
    
    def get_available_features(self) -> Dict[str, List[str]]:
        """
        Get list of all available PyRadiomics features by category.
        
        Returns:
            Dictionary with feature categories and their features
        """
        # Import the known feature list
        try:
            from Pyradiomics_Features import Features_3D
            all_features = Features_3D
        except ImportError:
            # Fallback: try to extract from dummy image if import fails
            # Create appropriate dummy image based on dimension
            if self.dimension == '2D':
                # For 2D, create a 2D image
                dummy_image = sitk.GetImageFromArray(np.random.rand(50, 50).astype(np.float32))
                dummy_mask = sitk.GetImageFromArray(np.ones((50, 50), dtype=np.uint8))
                dummy_image.SetSpacing([1.0, 1.0])
                dummy_mask.SetSpacing([1.0, 1.0])
            else:  # 3D
                # For 3D, create a 3D image
                dummy_image = sitk.GetImageFromArray(np.random.rand(50, 50, 50).astype(np.float32))
                dummy_mask = sitk.GetImageFromArray(np.ones((50, 50, 50), dtype=np.uint8))
                dummy_image.SetSpacing([1.0, 1.0, 1.0])
                dummy_mask.SetSpacing([1.0, 1.0, 1.0])
            
            features_dict = self.extractor.execute(dummy_image, dummy_mask)
            all_features = [name for name in features_dict.keys() 
                          if isinstance(features_dict[name], (int, float, np.number))]

        # Categorize features
        categories = {
            'firstorder': [],
            'shape': [],
            'glcm': [],
            'glrlm': [],
            'glszm': [],
            'gldm': [],
            'ngtdm': [],
            'diagnostics': [],
            'Misc': []
        }

        for feature_name in all_features:
            # Try to categorize the feature
            categorized = False
            for category in ['firstorder', 'shape', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']:
                # Check if feature name contains the pattern: original_{category}_
                if f'original_{category}_' in feature_name:
                    categories[category].append(feature_name)
                    categorized = True
                    break
            
            # Check for diagnostics features
            if not categorized and feature_name.startswith('diagnostics_'):
                categories['diagnostics'].append(feature_name)
                categorized = True
            
            # Only add to Misc if it wasn't categorized
            if not categorized:
                categories['Misc'].append(feature_name)

        return all_features, categories

def create_pyradiomics_config(dimension: str = '3D', 
                             bin_width: int = 25,
                             output_file: str = 'pyradiomics_config.yaml') -> str:
    """
    Create a PyRadiomics configuration file.
    
    Args:
        dimension: '2D' or '3D'
        bin_width: Bin width for discretization
        output_file: Output configuration file path
        
    Returns:
        Path to created configuration file
    """
    
    config_content = f"""
# PyRadiomics Configuration File
# Generated for {dimension} feature extraction

# Image and mask settings
setting:
  binWidth: {bin_width}
  voxelArrayShift: 0
  normalize: true
  normalizeScale: 100
  interpolator: 'sitkBSpline'
  correctMask: true
  additionalInfo: true
"""
    
    if dimension == '2D':
        config_content += """
  force2D: true
  force2Ddimension: 0
"""
    
    config_content += """
# Feature classes to extract
featureClass:
  firstorder: []
  shape: []
  glcm: []
  glrlm: []
  glszm: []
  gldm: []
  ngtdm: []

# Image filters (optional)
imageType:
  Original: {}
  # Wavelet: {}
  # LoG: {}
"""
    
    with open(output_file, 'w') as f:
        f.write(config_content)
    
    return output_file

# Example usage functions
def extract_2d_features(image_path: str, mask_path: str) -> pd.DataFrame:
    """Extract 2D radiomics features from a single image-mask pair."""
    extractor = PyRadiomicsExtractor(dimension='2D')
    features = extractor.extract_features_from_files(image_path, mask_path)
    return pd.DataFrame([features])

def extract_3d_features(image_path: str, mask_path: str) -> pd.DataFrame:
    """Extract 3D radiomics features from a single image-mask pair."""
    extractor = PyRadiomicsExtractor(dimension='3D')
    features = extractor.extract_features_from_files(image_path, mask_path)
    return pd.DataFrame([features])

def batch_extract_2d_features(image_dir: str, mask_dir: str, output_csv: str):
    """Extract 2D features from all images in directories."""
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.nii', '.nii.gz', '.dcm'))]
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f) for f in image_files]  # Assuming same names
    
    extractor = PyRadiomicsExtractor(dimension='2D')
    df = extractor.batch_extract_features(image_paths, mask_paths, output_csv)
    return df

def batch_extract_3d_features(image_dir: str, mask_dir: str, output_csv: str):
    """Extract 3D features from all images in directories."""
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.nii', '.nii.gz', '.dcm'))]
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f) for f in image_files]  # Assuming same names
    
    extractor = PyRadiomicsExtractor(dimension='3D')
    df = extractor.batch_extract_features(image_paths, mask_paths, output_csv)
    return df

if __name__ == "__main__":
    # # Example usage
    # print("PyRadiomics Feature Extractor")
    # print("============================")
    
    # # Create configuration files
    # config_2d = create_pyradiomics_config('2D', output_file='config_2d.yaml')
    # config_3d = create_pyradiomics_config('3D', output_file='config_3d.yaml')
    
    # print(f"Created configuration files: {config_2d}, {config_3d}")
    
    # config_3d = create_pyradiomics_config('3D', output_file='config_3d.yaml')
    # print(f"Created configuration file: {config_3d}")

    # Get available 3D features    
    extractor_3d = PyRadiomicsExtractor(dimension='3D', bin_width=5, voxel_array_shift=0, normalize=False, 
    normalize_scale=100, interpolator='sitkBSpline', resample_pixel_spacing=None)

    config_3d = create_pyradiomics_config('3D', output_file='ADNI_config_3d.yaml')
    # all_features_3d, categories_3d = extractor_3d.get_available_features()
    # for category, features in categories_3d.items():
    #     print(f"{category}: {features}")

    input_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/ADNI_137_S_1414_MR_MPR__GradWarp__N3__Scaled_Br_20101111123508835_S90858_I204818_0000.nii.gz"
    mask_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/masks/ADNI_137_S_1414_MR_MPR__GradWarp__N3__Scaled_Br_20101111123508835_S90858_I204818_0000_mask.nii.gz"

    # import nibabel as nib
    # import numpy as np

    # # Check mask values
    # mask_img = nib.load(mask_dir)
    # mask_data = mask_img.get_fdata()
    # unique_values = np.unique(mask_data)
    # print(f"Unique mask values: {unique_values}")
    # print(f"Mask shape: {mask_data.shape}")
    
    # # Count voxels for each label
    # for val in unique_values:
    #     count = np.sum(mask_data == val)
    #     print(f"Label {val}: {count} voxels ({count / mask_data.size * 100:.2f}% of mask)")

    features = extractor_3d.extract_features_from_files(input_dir, mask_dir)
    print(f"\nNumber of features: {len(features)}")
    print(f"Feature names: {list(features.keys())}")


    # Get available 2D features
    # extractor_2d = PyRadiomicsExtractor(dimension='2D')
    # all_features_2d, categories_2d = extractor_2d.get_available_features()
    # for key, value in categories_2d.items():
    #     print(f"{key}: {value}")
    # import pdb; pdb.set_trace()