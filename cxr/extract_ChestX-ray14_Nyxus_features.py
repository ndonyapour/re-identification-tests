"""
Extract Nyxus 2D features from ChestX-ray14 images stored on S3.

This module provides a class for extracting radiomic features using Nyxus
from chest X-ray images.
"""

import os
import io
import time
import logging

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
                          Default: ["*ALL*"] for all features.
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
        """
        Load grayscale image from S3.
        
        Args:
            s3_uri: S3 URI of the image
            
        Returns:
            Grayscale image as numpy array
        """
        bucket, key = self._parse_s3_uri(s3_uri)
        s3_client = self._get_s3_client()
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_bytes = response['Body'].read()
        
        img = Image.open(io.BytesIO(image_bytes))
        return np.array(img, dtype=np.float32)
    
    def load_image_from_local(self, image_path: str) -> np.ndarray:
        """
        Load grayscale image from local path.
        
        Args:
            image_path: Local path to image
            
        Returns:
            Grayscale image as numpy array
        """
        img = Image.open(image_path)
        return np.array(img, dtype=np.float32)
    
    def load_image(self, image_path_or_uri: str) -> np.ndarray:
        """
        Load image from S3 or local path.
        
        Args:
            image_path_or_uri: S3 URI or local file path
            
        Returns:
            Grayscale image as numpy array
        """
        if image_path_or_uri.startswith('s3://'):
            return self.load_image_from_s3(image_path_or_uri)
        return self.load_image_from_local(image_path_or_uri)
    
    def extract_features(self, image: np.ndarray) -> pd.DataFrame:
        """
        Extract Nyxus features from a grayscale image.
        
        Args:
            image: Grayscale image as numpy array
            
        Returns:
            DataFrame with extracted features
        """
        nyx = self._get_nyxus()
        mask = np.ones(image.shape, dtype=np.uint8)
        return nyx.featurize(image, mask)
    
    def process_single_image(self, image_path_or_uri: str) -> dict:
        """
        Process a single image and extract features.
        
        Args:
            image_path_or_uri: S3 URI or local path to image
            
        Returns:
            Dict with 'image_path' and 'features' (DataFrame)
        """
        image = self.load_image(image_path_or_uri)
        features = self.extract_features(image)
        return {'image_path': image_path_or_uri, 'features': features}
    
    def list_s3_images(self, s3_prefix: str, file_extensions: list[str] = None) -> list[str]:
        """List all image files in an S3 bucket/prefix."""
        if file_extensions is None:
            file_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        
        if not s3_prefix.startswith('s3://'):
            raise ValueError(f"Invalid S3 prefix: {s3_prefix}")
        
        s3_path = s3_prefix[5:]
        bucket_name, prefix = s3_path.split('/', 1) if '/' in s3_path else (s3_path, '')
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        
        s3_client = self._get_s3_client()
        image_uris = []
        
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if any(key.lower().endswith(ext.lower()) for ext in file_extensions):
                        image_uris.append(f's3://{bucket_name}/{key}')
        
        return image_uris
    
    def _get_output_filename(self, image_path: str) -> str:
        """Generate output filename from image path."""
        basename = os.path.basename(image_path)
        return os.path.splitext(basename)[0] + '.csv'
    
    def process_and_save(
        self,
        image_paths: list[str],
        output_dir: str,
        resume: bool = True
    ) -> list[str]:
        """
        Process multiple images and save features to CSV files.
        
        Args:
            image_paths: List of S3 URIs or local paths
            output_dir: Directory to save CSV files
            resume: Skip already processed images
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        processed = set()
        if resume and os.path.exists(output_dir):
            processed = {f for f in os.listdir(output_dir) if f.endswith('.csv')}
        
        saved_paths = []
        
        for image_path in tqdm(image_paths, desc="Extracting Nyxus features"):
            output_filename = self._get_output_filename(image_path)
            
            if resume and output_filename in processed:
                continue
            
            try:
                result = self.process_single_image(image_path)
                output_path = os.path.join(output_dir, output_filename)
                
                result['features']['image_path'] = image_path
                result['features'].to_csv(output_path, index=False)
                saved_paths.append(output_path)
                
            except Exception as e:
                print(f"\nError processing {image_path}: {e}")
        
        print(f"Saved {len(saved_paths)} feature files to {output_dir}")
        return saved_paths


if __name__ == "__main__":
    start_time = time.time()
    
    extractor = NyxusFeatureExtractor(
        feature_types=["*WHOLESLIDE*"],
        n_threads=5
    )
    
    s3_prefix = 's3://n3c-medical-imaging/ndonyapour/ChestXray-NIHCC'
    output_dir = './ChestX-ray14_Nyxus_features'
    
    for i in range(1, 13):
        subdir = os.path.join(s3_prefix, f"images_{i:02d}")
        print(f"\nProcessing {subdir}")
        
        image_uris = extractor.list_s3_images(subdir)
        print(f"Found {len(image_uris)} images")
        
        extractor.process_and_save(
            image_uris,
            output_dir=os.path.join(output_dir, f'features_{i:02d}'),
            resume=True
        )
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/3600:.2f} hours ({elapsed:.0f} seconds)")