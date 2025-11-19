import os
from traceback import print_tb
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

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

    # Add this method to enable only specific features
    def enable_specific_features(self, feature_names: List[str]) -> None:
        """
        Enable only specific PyRadiomics features by name.
        This is much faster than calculating all features.
        
        Args:
            feature_names: List of feature names to enable (e.g., ['original_firstorder_Mean', 'original_glcm_Contrast'])
        """
        # Disable all features first
        self.extractor.disableAllFeatures()
        
        # Enable only the requested features
        for feature_name in feature_names:
            # Parse feature name to get class and feature
            # Format: original_<class>_<feature>
            if feature_name.startswith('original_'):
                parts = feature_name.replace('original_', '').split('_', 1)
                if len(parts) == 2:
                    feature_class, feature = parts
                    try:
                        self.extractor.enableFeatureByName(feature_class, feature)
                    except Exception as e:
                        print(f"Warning: Could not enable {feature_name}: {e}")
            else:
                print(f"Warning: Feature {feature_name} doesn't match PyRadiomics naming convention")

    # Add this method to get PyRadiomics features corresponding to Nyxus wholeslide
    def get_pyradiomics_wholeslide_features(self) -> List[str]:
        """
        Get list of PyRadiomics features that correspond to Nyxus wholeslide features.
        Note: PyRadiomics doesn't support directional features or specialized features.
        
        Returns:
            List of PyRadiomics feature names that can be calculated
        """
        # Features that PyRadiomics can calculate (matching Nyxus wholeslide)
        pyradiomics_wholeslide = [
            # Firstorder (partial - only features PyRadiomics has)
            'original_firstorder_10Percentile',
            'original_firstorder_90Percentile',
            'original_firstorder_Energy',
            'original_firstorder_Entropy',
            'original_firstorder_InterquartileRange',
            'original_firstorder_Kurtosis',
            'original_firstorder_Maximum',
            'original_firstorder_MeanAbsoluteDeviation',
            'original_firstorder_Mean',
            'original_firstorder_Median',
            'original_firstorder_Minimum',
            'original_firstorder_Range',
            'original_firstorder_RobustMeanAbsoluteDeviation',
            'original_firstorder_RootMeanSquared',
            'original_firstorder_Skewness',
            'original_firstorder_TotalEnergy',
            'original_firstorder_Uniformity',
            'original_firstorder_Variance',
            # Shape (PyRadiomics shape features - note: different from Nyxus PERIMETER/DIAMETER)
            'original_shape_Elongation',
            'original_shape_Flatness',
            'original_shape_LeastAxisLength',
            'original_shape_MajorAxisLength',
            'original_shape_Maximum2DDiameterColumn',
            'original_shape_Maximum2DDiameterRow',
            'original_shape_Maximum2DDiameterSlice',
            'original_shape_Maximum3DDiameter',
            'original_shape_MeshVolume',
            'original_shape_MinorAxisLength',
            'original_shape_Sphericity',
            'original_shape_SurfaceArea',
            'original_shape_SurfaceVolumeRatio',
            'original_shape_VoxelVolume',
            # GLCM (averaged - PyRadiomics doesn't have directional)
            'original_glcm_Autocorrelation',
            'original_glcm_ClusterProminence',
            'original_glcm_ClusterShade',
            'original_glcm_ClusterTendency',
            'original_glcm_Contrast',
            'original_glcm_Correlation',
            'original_glcm_DifferenceAverage',
            'original_glcm_DifferenceEntropy',
            'original_glcm_DifferenceVariance',
            'original_glcm_Id',
            'original_glcm_Idm',
            'original_glcm_Idmn',
            'original_glcm_Idn',
            'original_glcm_Imc1',
            'original_glcm_Imc2',
            'original_glcm_InverseVariance',
            'original_glcm_JointAverage',
            'original_glcm_JointEnergy',
            'original_glcm_JointEntropy',
            'original_glcm_MCC',
            'original_glcm_MaximumProbability',
            'original_glcm_SumAverage',
            'original_glcm_SumEntropy',
            'original_glcm_SumSquares',
            # GLRLM (averaged - PyRadiomics doesn't have directional)
            'original_glrlm_GrayLevelNonUniformity',
            'original_glrlm_GrayLevelNonUniformityNormalized',
            'original_glrlm_GrayLevelVariance',
            'original_glrlm_HighGrayLevelRunEmphasis',
            'original_glrlm_LongRunEmphasis',
            'original_glrlm_LongRunHighGrayLevelEmphasis',
            'original_glrlm_LongRunLowGrayLevelEmphasis',
            'original_glrlm_LowGrayLevelRunEmphasis',
            'original_glrlm_RunEntropy',
            'original_glrlm_RunLengthNonUniformity',
            'original_glrlm_RunLengthNonUniformityNormalized',
            'original_glrlm_RunPercentage',
            'original_glrlm_RunVariance',
            'original_glrlm_ShortRunEmphasis',
            'original_glrlm_ShortRunHighGrayLevelEmphasis',
            'original_glrlm_ShortRunLowGrayLevelEmphasis',
            # GLSZM
            'original_glszm_GrayLevelNonUniformity',
            'original_glszm_GrayLevelNonUniformityNormalized',
            'original_glszm_GrayLevelVariance',
            'original_glszm_HighGrayLevelZoneEmphasis',
            'original_glszm_LargeAreaEmphasis',
            'original_glszm_LargeAreaHighGrayLevelEmphasis',
            'original_glszm_LargeAreaLowGrayLevelEmphasis',
            'original_glszm_LowGrayLevelZoneEmphasis',
            'original_glszm_SizeZoneNonUniformity',
            'original_glszm_SizeZoneNonUniformityNormalized',
            'original_glszm_SmallAreaEmphasis',
            'original_glszm_SmallAreaHighGrayLevelEmphasis',
            'original_glszm_SmallAreaLowGrayLevelEmphasis',
            'original_glszm_ZoneEntropy',
            'original_glszm_ZonePercentage',
            'original_glszm_ZoneVariance',
            # GLDM
            'original_gldm_DependenceEntropy',
            'original_gldm_DependenceNonUniformity',
            'original_gldm_DependenceNonUniformityNormalized',
            'original_gldm_DependenceVariance',
            'original_gldm_GrayLevelNonUniformity',
            'original_gldm_GrayLevelVariance',
            'original_gldm_HighGrayLevelEmphasis',
            'original_gldm_LargeDependenceEmphasis',
            'original_gldm_LargeDependenceHighGrayLevelEmphasis',
            'original_gldm_LargeDependenceLowGrayLevelEmphasis',
            'original_gldm_LowGrayLevelEmphasis',
            'original_gldm_SmallDependenceEmphasis',
            'original_gldm_SmallDependenceHighGrayLevelEmphasis',
            'original_gldm_SmallDependenceLowGrayLevelEmphasis',
            # NGTDM
            'original_ngtdm_Busyness',
            'original_ngtdm_Coarseness',
            'original_ngtdm_Complexity',
            'original_ngtdm_Contrast',
            'original_ngtdm_Strength'
        ]
        return pyradiomics_wholeslide
    
    def extract_features_from_files(self, 
                                   image_path: str, 
                                   mask_path: str,
                                   csv_path: str,
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
                    numeric_features[key] = float(value)
                else:
                    non_numeric_count += 1
                    non_numeric_features.append(key)
            
            # Convert dictionary to DataFrame and add metadata

            numeric_features_df = pd.DataFrame([numeric_features])
            numeric_features_df.insert(0, 'image_path', image_path)
            numeric_features_df.insert(1, 'mask_path', mask_path)
            numeric_features_df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")
            
        except Exception as e:
            raise Exception(f"Error extracting features from {image_path}: {str(e)}")

    def create_automatic_mask(self, image_array: np.ndarray, threshold_background: Optional[int] = None) -> sitk.Image:
        """
        Create an automatic mask from image array.
        If threshold_background is provided, creates mask excluding background pixels.
        Otherwise, creates mask covering entire image.
        
        Args:
            image_array: Image as numpy array (2D or 3D)
            threshold_background: Intensity threshold below which pixels are considered background (optional)
            
        Returns:
            SimpleITK Image object for the mask
        """
        # Create mask
        if threshold_background is not None:
            # Create mask excluding background (pixels above threshold)
            mask_array = (image_array > threshold_background).astype(np.uint8)
        else:
            # Create mask covering entire image
            mask_array = np.ones_like(image_array, dtype=np.uint8)
        
        # Convert to SimpleITK image
        mask = sitk.GetImageFromArray(mask_array)
        return mask

    def extract_features_from_arrays(self,
                                 image_array: np.ndarray,
                                 mask_array: Optional[np.ndarray] = None,
                                 threshold_background: Optional[int] = None,
                                 label: int = 1) -> Dict[str, float]:
        """
        Extract features from image and mask arrays (no file I/O).
        
        Args:
            image_array: Image as numpy array (2D or 3D)
            mask_array: Mask as numpy array (optional, will be auto-generated if None)
            threshold_background: Intensity threshold for background exclusion (optional)
            label: Label value in mask (default: 1)
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Convert image array to SimpleITK image
            image_sitk = sitk.GetImageFromArray(image_array)
            
            # Create mask if not provided
            if mask_array is None:
                mask_sitk = self.create_automatic_mask(image_array, threshold_background)
            else:
                mask_sitk = sitk.GetImageFromArray(mask_array.astype(np.uint8))
                # Copy spacing/origin from image if available
                if image_sitk.GetSpacing():
                    mask_sitk.SetSpacing(image_sitk.GetSpacing())
                if image_sitk.GetOrigin():
                    mask_sitk.SetOrigin(image_sitk.GetOrigin())
            
            # Extract features using PyRadiomics (can accept SimpleITK images directly)
            features = self.extractor.execute(image_sitk, mask_sitk, label=label)
            
            # Filter out non-numeric features (metadata)
            numeric_features = {}
            for key, value in features.items():
                # Handle numpy arrays (extract scalar if single element)
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = value.item()
                    else:
                        continue  # Skip multi-element arrays
                
                if isinstance(value, (int, float, np.number)):
                    # Check for NaN values
                    if np.isnan(value):
                        continue
                    numeric_features[key] = float(value)
            
            return numeric_features
        
        except Exception as e:
            raise Exception(f"Error extracting features from arrays: {str(e)}")

    def extract_features_from_image_only(self, 
                                    image_path: Optional[str] = None,
                                    csv_path: Optional[str] = None,
                                    threshold_background: Optional[int] = None,
                                    label: int = 1) -> Dict[str, float]:
        """
        Extract features from 2D whole slide image without requiring a mask.
        Can work with either file path or image array.
        
        Args:
            image_path: Path to the 2D image file (optional if image_array provided)
            image_array: Image as numpy array (optional if image_path provided)
            csv_path: Path to save the CSV file with features (optional)
            threshold_background: Intensity threshold for background exclusion (optional)
            label: Label value in mask (default: 1)
            
        Returns:
            Dictionary of extracted features
        """
    # Load image if path provided
        if image_path is not None:
            image = sitk.ReadImage(image_path)
            image_array = sitk.GetArrayFromImage(image)
        elif image_array is None:
            raise ValueError("Either image_path or image_array must be provided")
        
        # Extract features from arrays
        features = self.extract_features_from_arrays(
            image_array=image_array,
            mask_array=None,
            threshold_background=threshold_background,
            label=label
        )
        
        # Save to CSV if path provided
        if csv_path is not None:
            features_df = pd.DataFrame([features])
            features_df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")
        
        return features

    def extract_features_parallel(self, 
                                image_paths: List[str], 
                                mask_paths: List[str],
                                csv_paths: List[str],
                                num_workers: Optional[int] = None,
                                label: int = 1) -> None:
            """
            Extract features from multiple image-mask pairs using multiprocessing.
            
            Args:
                image_paths: List of image file paths
                mask_paths: List of mask file paths
                csv_paths: List of CSV output file paths (one per image)
                num_workers: Number of worker processes (default: cpu_count())
                label: Label value in mask to extract features from
            """
            if len(image_paths) != len(mask_paths):
                raise ValueError("Number of images and masks must be equal")
            if len(image_paths) != len(csv_paths):
                raise ValueError("Number of images and CSV paths must be equal")
            
            # Determine number of workers
            if num_workers is None:
                num_workers = cpu_count()
            num_workers = min(num_workers, len(image_paths))  # Don't use more workers than tasks
            
            print(f"Processing {len(image_paths)} images using {num_workers} workers")
            
            # Prepare arguments for worker function
            extractor_params = {
                'dimension': self.dimension,
                'bin_width': self.bin_width,
                'voxel_array_shift': self.voxel_array_shift,
                'normalize': self.normalize,
                'normalize_scale': self.normalize_scale,
                'interpolator': self.interpolator,
                'resample_pixel_spacing': self.resample_pixel_spacing
            }
            
            # Create argument tuples for each task
            tasks = list(zip(image_paths, mask_paths, csv_paths, [label] * len(image_paths)))
            
            # Use multiprocessing pool
            with Pool(processes=num_workers) as pool:
                # Create partial function with extractor params
                worker_func = partial(_extract_features_worker, extractor_params=extractor_params)
                
                # Process in parallel with progress tracking
                results = []
                for i, result in enumerate(pool.starmap(worker_func, tasks), 1):
                    if result['success']:
                        print(f"Completed {i}/{len(image_paths)}: {os.path.basename(result['image_path'])}")
                    else:
                        print(f"Failed {i}/{len(image_paths)}: {os.path.basename(result['image_path'])} - {result['error']}")
                    results.append(result)
            
            # Summary
            successful = sum(1 for r in results if r['success'])
            failed = len(results) - successful
            print(f"\nProcessing complete: {successful} successful, {failed} failed")

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

    def create_pyradiomics_config(self, output_file: str = 'pyradiomics_config.yaml') -> str:
        """
        Create a PyRadiomics configuration file.
        
        Args:
            output_file: Output configuration file path
            
        Returns:
            Path to created configuration file
        """
        all_features, categories = self.get_available_features()
        dimension = self.dimension
        
        # Extract feature names without 'original_' prefix for config
        def clean_feature_names(feature_list: List[str]) -> List[str]:
            """Remove 'original_' prefix and extract just the feature name."""
            cleaned = []
            for feat in feature_list:
                # Remove 'original_' prefix if present
                if feat.startswith('original_'):
                    feat = feat.replace('original_', '', 1)
                # Extract feature name after category (e.g., 'firstorder_Mean' -> 'Mean')
                parts = feat.split('_', 1)
                if len(parts) > 1:
                    cleaned.append(parts[1])
                else:
                    cleaned.append(feat)
            return cleaned
        
        # Format feature lists as YAML lists
        def format_yaml_list(items: List[str]) -> str:
            """Format list as YAML array."""
            if not items:
                return '[]'
            # Format as YAML list with proper indentation
            items_str = '\n    - '.join([''] + items)
            return items_str
        
        # Build settings section
        settings_lines = [
            f"  binWidth: {self.bin_width}",
            f"  voxelArrayShift: {self.voxel_array_shift}",
            f"  normalize: {str(self.normalize).lower()}",
            f"  normalizeScale: {self.normalize_scale}",
            f"  interpolator: '{self.interpolator}'",
            "  correctMask: true",
            "  additionalInfo: false"
        ]
        
        # Add resampling if specified
        if self.resample_pixel_spacing:
            spacing_str = ', '.join(map(str, self.resample_pixel_spacing))
            settings_lines.append(f"  resampledPixelSpacing: [{spacing_str}]")
        
        # Add 2D-specific settings
        if dimension == '2D':
            settings_lines.append("  force2D: true")
            settings_lines.append("  force2Ddimension: 0")
        
        settings_section = '\n'.join(settings_lines)
        
        # Build feature classes section
        feature_class_lines = []
        for category in ['firstorder', 'shape', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']:
            feature_names = clean_feature_names(categories.get(category, []))
            if feature_names:
                # Format as YAML list
                items_str = '\n    - '.join([''] + sorted(feature_names))
                feature_class_lines.append(f"  {category}:{items_str}")
            else:
                feature_class_lines.append(f"  {category}: []")
        
        feature_class_section = '\n'.join(feature_class_lines)
        
        # Build complete config
        config_content = f"""# PyRadiomics Configuration File
# Generated for {dimension} feature extraction

# Image and mask settings
setting:
{settings_section}

# Feature classes to extract
featureClass:
{feature_class_section}

# Image filters (optional)
imageType:
  Original: {{}}
  # Wavelet: {{}}
  # LoG: {{}}
"""
        
        with open(output_file, 'w') as f:
            f.write(config_content)
        
        return output_file


def _extract_features_worker(image_path: str, 
                            mask_path: str, 
                            csv_path: str, 
                            label: int,
                            extractor_params: Dict) -> Dict:
    """
    Worker function for multiprocessing feature extraction.
    Creates a PyRadiomicsExtractor instance and calls extract_features_from_files.
    
    Args:
        image_path: Path to the image file
        mask_path: Path to the mask file
        csv_path: Path to save CSV output
        label: Label value in mask
        extractor_params: Dictionary of extractor parameters
        
    Returns:
        Dictionary with success status and results/error message
    """
    try:
        # Create extractor instance in worker process
        extractor = PyRadiomicsExtractor(
            dimension=extractor_params['dimension'],
            bin_width=extractor_params['bin_width'],
            voxel_array_shift=extractor_params['voxel_array_shift'],
            normalize=extractor_params['normalize'],
            normalize_scale=extractor_params['normalize_scale'],
            interpolator=extractor_params['interpolator'],
            resample_pixel_spacing=extractor_params['resample_pixel_spacing']
        )
        
        # Use the existing extract_features_from_files method
        extractor.extract_features_from_files(image_path, mask_path, csv_path, label=label)
        
        return {
            'success': True,
            'image_path': image_path,
            'mask_path': mask_path,
            'csv_path': csv_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'image_path': image_path,
            'mask_path': mask_path,
            'csv_path': csv_path,
            'error': str(e)
        }


