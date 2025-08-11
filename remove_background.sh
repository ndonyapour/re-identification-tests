#!/bin/bash

# Directory containing all T1 images
INPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_n4_corrected"
# Directory for outputs
OUTPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_segmentations"
# INPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_test"
# # Directory for outputs
# OUTPUT_DIR="/Users/donyapourn2/Desktop/projects/datasets/ADNI/t1_mpr_segmentations_v2"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each .nii.gz file in the input directory
for input_image in "$INPUT_DIR"/*.nii.gz; do
    # Get the filename without path and extension
    filename=$(basename "$input_image" .nii.gz)
    echo "Processing $filename..."
    echo "Input image: $input_image"
    
    # Verify input image exists
    if [ ! -f "$input_image" ]; then
        echo "ERROR: Input image not found: $input_image"
        continue
    fi
    
    # Create output directory for this subject
    subject_dir="$OUTPUT_DIR/$filename"
    mkdir -p "$subject_dir"
    echo "Output directory: $subject_dir"

    # Generate a simple brain mask using Otsu thresholding
    echo "Generating brain mask..."
    ThresholdImage 3 ${input_image} ${subject_dir}/${filename}_mask.nii.gz Otsu 4
    # This will set everything > 0 to 1, effectively making it binary
    ThresholdImage 3 ${subject_dir}/${filename}_mask.nii.gz ${subject_dir}/${filename}_binary_mask.nii.gz 0.5 4 1 0
    # Remove the original mask
    rm ${subject_dir}/${filename}_mask.nii.gz
done