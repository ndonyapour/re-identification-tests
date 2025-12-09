"""
Extract CXR features from CheXpert dataset.

Filters:
- Only patients with at least 2 studies containing frontal images
- Only frontal images (view*_frontal.jpg)

Output naming: patientXXXXX_studyN.npz
"""

import os
import re
import time
import logging
from collections import defaultdict

import boto3
from tqdm import tqdm

from cxr_embedding_extractor import CXREmbeddingExtractor

logging.getLogger('tensorflow').setLevel(logging.ERROR)


def parse_chexpert_path(s3_uri: str) -> dict | None:
    """
    Parse a CheXpert image S3 URI to extract patient, study, and view info.
    
    Args:
        s3_uri: e.g., 's3://bucket/prefix/patient02930/study2/view1_frontal.jpg'
        
    Returns:
        Dict with 'patient', 'study', 'view', 'is_frontal', or None if invalid
    """
    # Match pattern: .../patientXXXXX/studyN/view*_frontal.jpg or view*_lateral.jpg
    pattern = r'(patient\d+)/(study\d+)/(view\d+_(frontal|lateral)\.jpg)$'
    match = re.search(pattern, s3_uri, re.IGNORECASE)
    
    if not match:
        return None
    
    return {
        'patient': match.group(1),
        'study': match.group(2),
        'filename': match.group(3),
        'is_frontal': 'frontal' in match.group(3).lower(),
        's3_uri': s3_uri
    }


def get_patients_with_min_studies(
    s3_client,
    s3_prefix: str,
    min_studies: int = 2
) -> dict[str, list[dict]]:
    """
    Find patients with at least min_studies studies containing frontal images.
    
    Args:
        s3_client: boto3 S3 client
        s3_prefix: S3 prefix to search (e.g., 's3://bucket/prefix/batch/')
        min_studies: Minimum number of studies with frontal images required
        
    Returns:
        Dict mapping patient_id -> list of frontal image info dicts
    """
    # Parse S3 prefix
    if not s3_prefix.startswith('s3://'):
        raise ValueError(f"Invalid S3 prefix: {s3_prefix}")
    
    s3_path = s3_prefix[5:]
    bucket_name, prefix = s3_path.split('/', 1)
    if not prefix.endswith('/'):
        prefix += '/'
    
    print(f"Scanning {s3_prefix} for patients with ≥{min_studies} studies...")
    
    # Collect all images
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    # Group by patient -> study -> frontal images
    patient_studies = defaultdict(lambda: defaultdict(list))
    
    for page in pages:
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            key = obj['Key']
            s3_uri = f's3://{bucket_name}/{key}'
            
            parsed = parse_chexpert_path(s3_uri)
            if parsed and parsed['is_frontal']:
                patient_id = parsed['patient']
                study_id = parsed['study']
                patient_studies[patient_id][study_id].append(parsed)
    
    # Filter to patients with >= min_studies studies containing frontal images
    filtered_patients = {}
    for patient_id, studies in patient_studies.items():
        if len(studies) >= min_studies:
            # Flatten: list of all frontal images for this patient
            all_frontal_images = []
            for study_id, images in studies.items():
                all_frontal_images.extend(images)
            filtered_patients[patient_id] = all_frontal_images
    
    total_images = sum(len(imgs) for imgs in filtered_patients.values())
    print(f"Found {len(filtered_patients)} patients with ≥{min_studies} studies")
    print(f"Total frontal images to process: {total_images}")
    
    return filtered_patients


def extract_chexpert_features(
    extractor: CXREmbeddingExtractor,
    patients_data: dict[str, list[dict]],
    output_dir: str,
    resume: bool = True
):
    """
    Extract features for filtered patients and save with patient_study naming.
    
    Args:
        extractor: CXREmbeddingExtractor instance
        patients_data: Dict from get_patients_with_min_studies()
        output_dir: Directory to save embeddings
        resume: Skip already processed images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check what's already processed
    processed = set()
    if resume:
        for f in os.listdir(output_dir):
            if f.endswith('.npz'):
                processed.add(f[:-4])  # Remove .npz
        print(f"Resume mode: {len(processed)} already processed")
    
    # Build list of images to process
    images_to_process = []
    for patient_id, images in patients_data.items():
        for img_info in images:
            # Key format: patient02930_study1
            key = f"{img_info['patient']}_{img_info['study']}"
            if key not in processed:
                images_to_process.append({
                    'key': key,
                    's3_uri': img_info['s3_uri'],
                    'patient': img_info['patient'],
                    'study': img_info['study']
                })
    
    print(f"Images to process: {len(images_to_process)}")
    
    if not images_to_process:
        print("All images already processed!")
        return
    
    # Process images
    for img_info in tqdm(images_to_process, desc="Extracting features"):
        try:
            # Extract embedding
            embedding = extractor._process_single_image(img_info['s3_uri'])
            
            # Save with patient_study key
            output_path = os.path.join(output_dir, f"{img_info['key']}.npz")
            import numpy as np
            np.savez_compressed(
                output_path,
                contrastive_img_emb=embedding['contrastive_img_emb'],
                all_contrastive_img_emb=embedding['all_contrastive_img_emb'],
                image_path=img_info['s3_uri'],
                patient=img_info['patient'],
                study=img_info['study']
            )
        except Exception as e:
            print(f"\nError processing {img_info['key']}: {e}")
            continue
    
    print(f"Done! Embeddings saved to {output_dir}")


if __name__ == "__main__":
    start_time = time.time()
    
    # Initialize
    s3_client = boto3.client('s3')
    extractor = CXREmbeddingExtractor(model_dir='./cxr_models', s3_client=s3_client)
    
    # Configuration
    s3_base = 's3://n3c-medical-imaging/ndonyapour/chexpertchestxrays-u20210408'
    output_base = './CheXpert_embeddings'
    min_studies = 2  # Patients must have at least 2 studies with frontal images
    
    batches = [
        'chexpert_v1.0_batch_2_train',
        # 'chexpert_v1.0_batch_3_train',
        # 'chexpert_v1.0_batch_4_train',
    ]
    
    for batch_name in batches:
        print(f"\n{'='*60}")
        print(f"Processing: {batch_name}")
        print('='*60)
        
        s3_prefix = f"{s3_base}/{batch_name}"
        output_dir = os.path.join(output_base, batch_name)
        
        # Find eligible patients
        patients_data = get_patients_with_min_studies(
            s3_client, 
            s3_prefix, 
            min_studies=min_studies
        )

        # n_patients = 5
        # patients_data = dict(list(patients_data.items())[:n_patients])
        
        # Extract features
        extract_chexpert_features(
            extractor,
            patients_data,
            output_dir,
            resume=True
        )
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/3600:.2f} hours")