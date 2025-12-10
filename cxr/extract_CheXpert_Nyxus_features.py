"""
Extract Nyxus 2D features from CheXpert dataset.

Filters:
- Only patients with at least 2 studies containing frontal images
- Only frontal images (view*_frontal.jpg)

Output naming: patientXXXXX_studyN.csv
"""

import os
import io
import re
import time
import logging
from collections import defaultdict

import boto3
import numpy as np
import nyxus
import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.getLogger('tensorflow').setLevel(logging.ERROR)


class NyxusFeatureExtractor:
    """Extract Nyxus 2D radiomic features from chest X-ray images."""
    
    DEFAULT_FEATURE_TYPES = ["*WHOLESLIDE*"]
    
    def __init__(
        self,
        feature_types: list[str] = None,
        n_threads: int = 5,
        s3_client=None
    ):
        """
        Initialize Nyxus feature extractor.
        
        Args:
            feature_types: List of Nyxus feature types to extract.
            n_threads: Number of threads for feature calculation
            s3_client: Optional boto3 S3 client
        """
        self.feature_types = feature_types or self.DEFAULT_FEATURE_TYPES
        self.n_threads = n_threads
        self.s3_client = s3_client
        self._nyx = None
    
    def _get_s3_client(self):
        """Lazy initialization of S3 client."""
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
        return self.s3_client
    
    def _get_nyxus(self) -> nyxus.Nyxus:
        """Get or create Nyxus instance."""
        if self._nyx is None:
            self._nyx = nyxus.Nyxus(
                self.feature_types,
                n_feature_calc_threads=self.n_threads
            )
        return self._nyx
    
    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """Parse S3 URI into bucket and key."""
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        path = s3_uri[5:]
        bucket, key = path.split('/', 1)
        return bucket, key
    
    def load_image_from_s3(self, s3_uri: str) -> np.ndarray:
        """Load grayscale image from S3."""
        bucket, key = self._parse_s3_uri(s3_uri)
        s3_client = self._get_s3_client()
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_bytes = response['Body'].read()
        
        img = Image.open(io.BytesIO(image_bytes))
        return np.array(img, dtype=np.float32)
    
    def extract_features(self, image: np.ndarray) -> pd.DataFrame:
        """Extract Nyxus features from a grayscale image."""
        nyx = self._get_nyxus()
        mask = np.ones(image.shape, dtype=np.uint8)
        return nyx.featurize(image, mask)
    
    def process_single_image(self, s3_uri: str) -> pd.DataFrame:
        """Process a single image and extract features."""
        image = self.load_image_from_s3(s3_uri)
        return self.extract_features(image)


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


def extract_chexpert_nyxus_features(
    extractor: NyxusFeatureExtractor,
    patients_data: dict[str, list[dict]],
    output_dir: str,
    resume: bool = True
):
    """
    Extract Nyxus features for filtered patients and save with patient_study naming.
    
    Args:
        extractor: NyxusFeatureExtractor instance
        patients_data: Dict from get_patients_with_min_studies()
        output_dir: Directory to save CSV files
        resume: Skip already processed images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check what's already processed
    processed = set()
    if resume:
        for f in os.listdir(output_dir):
            if f.endswith('.csv'):
                processed.add(f[:-4])  # Remove .csv
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
    for img_info in tqdm(images_to_process, desc="Extracting Nyxus features"):
        try:
            # Extract features
            features_df = extractor.process_single_image(img_info['s3_uri'])
            
            # Add metadata
            features_df['image_path'] = img_info['s3_uri']
            features_df['patient'] = img_info['patient']
            features_df['study'] = img_info['study']
            
            # Save as CSV
            output_path = os.path.join(output_dir, f"{img_info['key']}.csv")
            features_df.to_csv(output_path, index=False)
            
        except Exception as e:
            print(f"\nError processing {img_info['key']}: {e}")
            continue
    
    print(f"Done! Features saved to {output_dir}")


if __name__ == "__main__":
    start_time = time.time()
    
    # Initialize
    s3_client = boto3.client('s3')
    extractor = NyxusFeatureExtractor(
        feature_types=["*WHOLESLIDE*"],
        n_threads=5,
        s3_client=s3_client
    )
    
    # Configuration
    s3_base = 's3://n3c-medical-imaging/ndonyapour/chexpertchestxrays-u20210408'
    output_base = './CheXpert_Nyxus_features'
    min_studies = 2  # Patients must have at least 2 studies with frontal images
    
    batches = [
        'chexpert_v1.0_batch_2_train',
        'chexpert_v1.0_batch_3_train',
        'chexpert_v1.0_batch_4_train',
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
        extract_chexpert_nyxus_features(
            extractor,
            patients_data,
            output_dir,
            resume=True
        )
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/3600:.2f} hours")
