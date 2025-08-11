import os
import numpy as np

import nibabel as nib
from nyxusmed import Nyxus3DFeatureExtractor
from intensity_normalization import NyulNormalizer, WhiteStripeNormalizer, KDENormalizer
from intensity_normalization.adapters.images import create_image 
from intensity_normalization.adapters.io import  save_image
from typing import List, Tuple
from pathlib import Path
from collections import defaultdict
import re
import random
from tqdm import tqdm


# https://www.medrxiv.org/content/10.1101/2021.02.24.21252322v2.full
def get_percentile(image_data, percentile: float = 98):
    # Flatten the image array to 1D
    flat_data = image_data.flatten()
    
    # Remove background/zero values
    nonzero_data = flat_data[flat_data > 0]
    
    # Calculate the percentile
    p = np.percentile(nonzero_data, percentile)
    return p

def extract_date_from_filename(filepath: str) -> str:
    """Extract date from filename.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Extracted date string
    """
    # Implement date extraction logic based on your filename format
    return os.path.basename(filepath)


def run_3d_extraction(input_dir: str, seg_dir: str, out_dir: str) -> None:
    """Test 3D volumetric feature extraction.
    
    Args:
        input_files: Path pattern for input intensity images
        seg_files: Path pattern for segmentation masks
        out_dir: Output directory for results
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    input_dir = Path(input_dir)
    seg_dir = Path(seg_dir)

    for input_file in tqdm(input_dir.iterdir()):
        if str(input_file).endswith(".nii.gz") or str(input_file).endswith(".nii"):
            seg_file = input_file.stem.split(".")[0]+".nii.gz"
            seg_file = Path(seg_dir).joinpath(seg_file)
            print("seg_file", seg_file)
            #import pdb; pdb.set_trace()              
            outpath = Path(out_dir)
            # outfile = outpath.joinpath(input_file.stem.split(".")[0]) + ".csv"
            extractor = Nyxus3DFeatureExtractor(
                        input_file=str(input_file),
                        seg_files=str(seg_file),
                        out_dir=outpath ,
                        features=["ALL"],
                        per_slice=False,
                        out_format="csv"
                    )
                
            print(f"Running 3D feature extraction for {input_file.name}...")
            extractor.run_parallel()
            print(f"3D results saved to: {outpath}")

def get_subject_id(nifti_img, filename: str = '') -> str:
    header = nifti_img.header
    descrip = header.get('descrip', '').tobytes().decode('utf-8').lower()
    db_name = header.get('db_name', '').tobytes().decode('utf-8').lower()
    
    # Try to find subject ID
    subject_patterns = [
        r'(sub-\w+)',           # BIDS format
        r'(adni_\d+)',          # ADNI format
        r'(s\d{4})',            # ADNI S#### format
        r'(subject[-_]\w+)'     # General format
    ]
    
    # Look in both header fields and filename
    text_to_search = f"{descrip} {db_name} {filename}".lower()
    
    for pattern in subject_patterns:
        match = re.search(pattern, text_to_search)
        if match:
            subject_id = match.group(1)
            break
    return subject_id

def select_training_images(input_dir: str, filename: str = '') -> list[str]:
    """
    Get subject ID and scan date from a NIfTI file.
    
    Args:
        nifti_img: A NiBabel image object
        filename: Optional filename to extract information from
        
    Returns:
        tuple: (subject_id, scan_date) where either could be 'unknown' if not found
    """
    # Set random seed for reproducibility
    random.seed(42)
    
    subject_id = 'unknown'
    subject_ids = defaultdict(list)
    for file in os.listdir(input_dir):
        if file.endswith('.nii.gz') or file.endswith('.nii'):
            file_path = os.path.join(input_dir, file)
            img = nib.load(file_path)
            subject_id = get_subject_id(img, file)
            subject_ids[subject_id].append(file)

    # Select 54 images for training
    
    train_files = []
    for subject_id, files in subject_ids.items():
        train_files.extend(random.sample(files, 1))
 
    return train_files

def learn_Nyul_normalization(image_dir: str, seg_dir: str, train_files: list[str] = [],
                             min_percentile: float = 1.0, max_percentile: float = 99.99, 
                             output_min: float = 0.0, standard_histogram_path: str = None,
                             percentile_step: float = 10.0) -> NyulNormalizer:
    """Learn Nyul normalization parameters using white matter masks directly in fit.
    
    Args:
        image_dir: Directory containing input images
        seg_dir: Directory containing segmentation masks
        out_dir: Output directory
        min_percentile: Minimum percentile for normalization
        max_percentile: Maximum percentile for normalization
        output_min: Minimum output value
        train_files: List of training files
        standard_histogram_path: Path to save standard histogram
    """
    train_images = []
    train_masks = []
    
    # Process each image and its mask
    for file_name in train_files:
        image_file = os.path.join(image_dir, file_name)
        mask_file = os.path.join(seg_dir, Path(file_name).stem.split(".")[0], 
                              f"{Path(file_name).stem.split('.')[0]}_mask.nii.gz")

        print(f"Processing {file_name}...")
        
        
        train_images.append(create_image(image_file))
        train_masks.append(create_image(mask_file))

    print(f"Training on {len(train_images)} images with masks")
    
    # Initialize and fit normalizer with images and masks
    #
    data_flattened = []
    for img in train_images:
        data_flattened.extend(img.get_data().flatten())

    
    print("Calculating output max...")
    output_max = get_percentile(np.array(data_flattened), percentile=98)
    print(f"Output max: {output_max}")
 
    normalizer = NyulNormalizer(min_percentile=min_percentile, 
                              max_percentile=max_percentile,
                              output_min_value=output_min, 
                              output_max_value=output_max, 
                              percentile_step=percentile_step)
    
    normalizer.fit_population(train_images, masks=train_masks)
    
    normalizer.save_standard_histogram(standard_histogram_path)
    
    print(f"Normalizer saved to {standard_histogram_path}")
    
    return normalizer

def apply_nyul_normalization(image_dir: str, seg_dir: str, out_dir: str, normalizer_path: str) -> None:
    """Apply Nyul normalization using white matter masks.
    
    Args:
        image_dir: Directory containing input images
        seg_dir: Directory containing segmentation masks
        out_dir: Output directory for normalized images
        normalizer_path: Path to saved normalizer
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith('.nii.gz') or f.endswith('.nii')])

    normalizer = NyulNormalizer()
    normalizer.load_standard_histogram(normalizer_path)
    
    for file_name in tqdm(image_files, total=len(image_files)):
        print(f"Processing {file_name}...")
        
        # Load image and mask
        img_nib = nib.load(os.path.join(image_dir, file_name))
        img = img_nib.get_fdata()
        
        mask_file = os.path.join(seg_dir, Path(file_name).stem.split(".")[0], 
                              f"{Path(file_name).stem.split('.')[0]}_mask.nii.gz")
        
        # Apply normalization with mask
        normalized = normalizer.transform(create_image(os.path.join(image_dir, file_name)), 
                                          mask=create_image(mask_file))
        
        # Save normalized image
        output_path = os.path.join(out_dir, f"{Path(file_name).stem.split('.')[0]}.nii.gz")
        save_image(normalized, output_path)

# def apply_whitestripe_normalization(image_dir: str, seg_dir: str, out_dir: str, scale: float = 100.0) -> None:
#     """Apply WhiteStripe normalization using the WhiteStripeNormalize class.
    
#     Args:
#         image_dir: Directory containing input images
#         seg_dir: Directory containing segmentation masks
#         out_dir: Output directory for normalized images
#         scale: Scale factor for normalization (default=100.0)
#     """
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
        
#     image_files = sorted([f for f in os.listdir(image_dir) 
#                          if f.endswith('.nii.gz') or f.endswith('.nii')])

#     # Initialize WhiteStripe normalizer
#     whitestripe = WhiteStripeNormalize(norm_value=scale)
    
#     for file_name in tqdm(image_files, total=len(image_files)):
#         print(f"Processing {file_name}...")
        
#         # Load image and mask
#         img_nib = nib.load(os.path.join(image_dir, file_name))
#         img = img_nib.get_fdata()
        
#         mask_file = os.path.join(seg_dir, Path(file_name).stem.split(".")[0], 
#                               f"{Path(file_name).stem.split('.')[0]}_white_matter_mask.nii.gz")
        
#         if os.path.exists(mask_file):
#             mask_nib = nib.load(mask_file)
#             mask = mask_nib.get_fdata() > 0
#         else:
#             print(f"Warning: No mask found for {file_name}, using whole brain")
#             mask = np.ones_like(img, dtype=bool)
        
#         # Apply WhiteStripe normalization
#         normalized = whitestripe(img, mask=mask)
        
#         # Create new nifti image
#         normalized_nifti = nib.Nifti1Image(normalized, img_nib.affine, img_nib.header)
        
#         # Save normalized image
#         output_path = os.path.join(out_dir, f"{Path(file_name).stem.split('.')[0]}.nii.gz")
#         nib.save(normalized_nifti, output_path)

def apply_kde_normalization(image_dir: str, seg_dir: str, out_dir: str) -> None:    
    """Run KDE normalization on a set of images.
    
    Args:
        image_dir: Directory containing input images
        out_dir: Output directory for normalized images
    """
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)

    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith('.nii.gz') or f.endswith('.nii')])
    
    kde_normalizer = KDENormalizer()

    for file_name in tqdm(image_files, total=len(image_files)):
        print(f"Processing {file_name}...")
        img_file = os.path.join(image_dir, file_name)
        if seg_dir is not None:
            mask_file = os.path.join(seg_dir, Path(file_name).stem.split(".")[0], 
                                f"{Path(file_name).stem.split('.')[0]}_mask.nii.gz")
            normalized = kde_normalizer.fit_transform(create_image(img_file), mask=create_image(mask_file))
        else:
            normalized = kde_normalizer.fit_transform(create_image(img_file))
   
        
        output_path = os.path.join(out_dir, f"{Path(file_name).stem.split('.')[0]}.nii.gz")
        save_image(normalized, output_path)

def remove_negative_values(input_dir: str, output_dir: str) -> None:
    """
    Remove negative values from the images in the input directory and save the results in the output directory.
    """
    min_value = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Starting to find minimum value...")
    for file in tqdm(os.listdir(input_dir), total=len(os.listdir(input_dir))):
        if file.endswith('.nii.gz') or file.endswith('.nii'):
            file_path = os.path.join(input_dir, file)
            img = create_image(file_path)
            img_data = img.get_data()
            min_value.append(img_data.min())

    shift = - min(min_value)
    print(f"Shift: {shift}")
    print("Starting to remove negative values...")
    for file in tqdm(os.listdir(input_dir), total=len(os.listdir(input_dir))):
        if file.endswith('.nii.gz') or file.endswith('.nii'):
            file_path = os.path.join(input_dir, file)
            img = create_image(file_path)
            img_data = img.get_data()
            img_data = img_data + shift
            img = img.with_data(img_data)
            save_image(img, os.path.join(output_dir, file))

    print(f"Removed negative values for all images in {input_dir} and saved to {output_dir}")
       