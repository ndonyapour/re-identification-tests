from gettext import find
import os
import nibabel as nib
from pathlib import Path
import re
from datetime import datetime
import shutil

def get_scan_type(nifti_img) -> str:
    """
    Identifies the type of scan from a NIfTI image header.
    
    Args:
        nifti_img: A NiBabel image object.
        
    Returns:
        str: The identified scan type or the raw description if type not identified.
    """
    header = nifti_img.header
    descrip = header.get('descrip', '').tobytes().decode('utf-8').lower()
    
    # Check for MPR/MPRAGE sequences first as they're usually T1
    if 'mpr' in descrip:
        if 'mprage' in descrip:
            return 't1_mprage'
        return 't1_mpr'
    elif 't1' in descrip:
        return 't1'
    elif 't2' in descrip:
        return 't2'
    elif 'flair' in descrip:
        return 'flair'
    elif 'dwi' in descrip or 'dti' in descrip:
        return 'dwi'
    elif 'bold' in descrip or 'func' in descrip:
        return 'functional'
    else:
        return descrip 

def is_t1_scan(nifti_img):
    header = nifti_img.header
    descrip = header.get('descrip', '').tobytes().decode('utf-8').lower()
    #print(descrip)
    return 't1' in descrip

def get_subject_and_date(filename: str) -> tuple[str, str]:
    """
    Get subject ID and scan date from a NIfTI file.
    
    Args:
        filename: Filename to extract information from (ADNI format expected)
        
    Returns:
        tuple: (subject_id, scan_date) where either could be 'unknown' if not found
    """
    subject_id = 'unknown'
    scan_date = 'unknown'
    
    # ADNI filename pattern: ADNI_035_S_0048_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080206161657886_S44467_I89626.nii
    # Subject ID is in format: XXX_S_XXXX (e.g., 035_S_0048)
    # Date is in format: YYYYMMDD (e.g., 20080206)
    
    # Extract ADNI subject ID pattern (XXX_S_XXXX)
    subject_pattern = r'ADNI_(\d{3}_S_\d{4})'
    subject_match = re.search(subject_pattern, filename)
    if subject_match:
        subject_id = subject_match.group(1)
    
    # Extract date pattern (YYYYMMDD) - look for 8 digits that represent a valid date
    date_pattern = r'_(\d{8})\d+'  # 8 digits followed by more digits (timestamp)
    date_match = re.search(date_pattern, filename)
    if date_match:
        date_str = date_match.group(1)
        # Validate it's a reasonable date (starts with 19xx or 20xx)
        if date_str.startswith(('19', '20')):
            # Convert YYYYMMDD to YYYY-MM-DD format
            scan_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    
    return subject_id, scan_date

def find_scans(directory: str) -> dict[str, list[str]]:
    """
    Find and categorize all NIfTI scans in the given directory by their type.
    
    Args:
        directory: Path to the directory containing NIfTI files.
        
    Returns:
        dict: Dictionary mapping scan types to lists of filenames.
    """
    scan_files = {
        't1': [],
        't2': [],
        'flair': [],
        'dwi': [],
        'functional': [],
        'unknown': [],
        't1_mprage': [],
        't1_mpr': []
    }
    info = {
        'subject_id': [],
        'scan_date': []
    }
    scan_types = []
    print(f"Scanning directory: {directory}")
    for file in os.listdir(directory):
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            file_path = os.path.join(directory, file)
            try:
                img = nib.load(file_path)
                scan_type = get_scan_type(img)
                scan_types.append(scan_type)
                scan_files[scan_type].append(file)
                subject_id, scan_date = get_subject_and_date(file)
                info['subject_id'].append(subject_id)
                info['scan_date'].append(scan_date)
            except Exception as e:
                #print(f"Failed to read {file}: {e}")
                continue

    print(f"_"*100)
    print(f"Number of nii files: {len(os.listdir(directory))}")
    print(f'Numbmer of nii files can not be identified: {len(os.listdir(directory)) - len(scan_types)}')
    for scan_type, files in scan_files.items():
        print(f"{scan_type.upper()} scans ({len(files)}):")

    print(f'Number of unique scan dates: {len(set(info["scan_date"]))}')
    print(f'Number of unique subject IDs: {len(set(info["subject_id"]))}')
    return scan_files

def print_nifti_header_info(nifti_img) -> None:
    """
    Prints detailed header information from a NIfTI image.
    
    Args:
        nifti_img: A NiBabel image object.
    """
    header = nifti_img.header
    
    # Get the raw description field
    descrip = header.get('descrip', '').tobytes().decode('utf-8')
    
    print("\nNIfTI Header Information:")
    print("-" * 50)
    print(f"Description field: {descrip}")
    print(f"Image dimensions: {header.get_data_shape()}")
    print(f"Voxel sizes: {header.get_zooms()}")
    print(f"Units: {header.get_xyzt_units()}")
    print(f"Data type: {header.get_data_dtype()}")
    
    # Print any sequence information if available
    if hasattr(header, 'get_slice_duration'):
        print(f"Slice duration: {header.get_slice_duration()}")
    
    scan_type = get_scan_type(nifti_img)
    print(f"\nIdentified scan type: {scan_type.upper()}")
    print("-" * 50)

def analyze_single_nifti(file_path: str) -> None:
    """
    Analyze a single NIfTI file and print its information.
    
    Args:
        file_path: Path to the NIfTI file.
    """
    try:
        img = nib.load(file_path)
        print_nifti_header_info(img)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

def organize_scans_by_type(input_dir: str, output_dir: str, scans: dict[str, list[str]], scan_type: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # import ipdb; ipdb.set_trace()
    for file in scans[scan_type]: 
        shutil.move(os.path.join(input_dir, file), os.path.join(output_dir, file))
    print(f"Moved {len(scans[scan_type])} files to {output_dir}")

def is_normal_control(nifti_img, filename: str = '') -> bool:
    """
    Check if the scan is from a Normal Control subject in ADNI.
    
    Args:
        nifti_img: A NiBabel image object
        filename: Optional filename to extract information from
        
    Returns:
        bool: True if the subject is a Normal Control, False otherwise
    """
    header = nifti_img.header
    for field in header.keys():
        print(f"{field}: {header[field]}")
    
    descrip = header.get('descrip', '').tobytes().decode('utf-8').lower()
    db_name = header.get('db_name', '').tobytes().decode('utf-8').lower()
    
    # Combine all text fields to search
    text_to_search = f"{descrip} {db_name} {filename}".lower()
    
    # ADNI Normal Control identifiers
    nc_patterns = [
        r'(cn_)',              # Control Normal prefix
        r'_(cn)_',             # Control Normal in middle
        r'(normal[\s_]control)',
        r'(control[\s_]normal)',
        r'(healthy[\s_]control)',
        r'(normal[\s_]subject)',
    ]
    
    return any(re.search(pattern, text_to_search) for pattern in nc_patterns)

def select_normal_controls(input_dir: str, output_dir: str) -> None:
    """
    Select and copy Normal Control scans to a separate directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to store Normal Control images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    nc_files = []
    total_files = 0
    
    print(f"Scanning for Normal Control subjects in {input_dir}...")
    
    for file in os.listdir(input_dir):
        if file.endswith('.nii.gz') or file.endswith('.nii'):
            total_files += 1
            file_path = os.path.join(input_dir, file)
            try:
                img = nib.load(file_path)
                if is_normal_control(img, file):
                    nc_files.append(file)
                    # Copy instead of move to preserve original data
                    # shutil.copy2(file_path, os.path.join(output_dir, file))
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
    
    print(f"\nNormal Control Selection Results:")
    print("-" * 50)
    print(f"Total files processed: {total_files}")
    print(f"Normal Control files found: {len(nc_files)}")
    print(f"Files copied to: {output_dir}")
   

    return nc_files

def main():
    input_dir = "/home/ubuntu/data/ADNI_dataset/images"
    # output_dir = "../datasets/ADNI/t1_mpr"
    t1_mpr_output_dir = "/home/ubuntu/data/ADNI_dataset/t1_mpr"
    scans = find_scans(input_dir)
    organize_scans_by_type(input_dir, t1_mpr_output_dir, scans, 't1_mpr')
    # print(scans)
    
    
if __name__ == "__main__":
    #main()
    input_dir = "/home/ubuntu/data/ADNI_dataset/t1_mpr"
    scans = find_scans(input_dir)