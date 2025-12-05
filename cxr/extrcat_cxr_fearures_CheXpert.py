"""
Extract CXR features from CheXpert dataset stored as ZIP files on S3.

This script handles large ZIP files by STREAMING directly from S3:
1. Uses S3 range requests to read ZIP central directory
2. Reads individual images on-demand without downloading the entire ZIP
3. No local storage required for the ZIP files
4. Saves embeddings incrementally to S3 or local storage
"""

import os
import io
import zipfile
import time

import boto3
import numpy as np
from PIL import Image
from tqdm import tqdm

from extract_cxr_features import CXREmbeddingExtractor


class S3File:
    """
    File-like object that reads from S3 using range requests.
    
    Supports seeking and reading, which allows zipfile to work directly
    with ZIP files on S3 without downloading them.
    """
    
    def __init__(self, s3_client, bucket: str, key: str):
        """
        Initialize S3File.
        
        Args:
            s3_client: boto3 S3 client
            bucket: S3 bucket name
            key: S3 object key
        """
        self.s3_client = s3_client
        self.bucket = bucket
        self.key = key
        self._pos = 0
        
        # Get file size
        response = s3_client.head_object(Bucket=bucket, Key=key)
        self._size = response['ContentLength']
    
    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to a position in the file."""
        if whence == 0:  # SEEK_SET
            self._pos = offset
        elif whence == 1:  # SEEK_CUR
            self._pos += offset
        elif whence == 2:  # SEEK_END
            self._pos = self._size + offset
        
        self._pos = max(0, min(self._pos, self._size))
        return self._pos
    
    def tell(self) -> int:
        """Return current position."""
        return self._pos
    
    def read(self, size: int = -1) -> bytes:
        """Read bytes from the current position."""
        if size == -1:
            size = self._size - self._pos
        
        if size <= 0 or self._pos >= self._size:
            return b''
        
        # Calculate byte range
        start = self._pos
        end = min(self._pos + size - 1, self._size - 1)
        
        # Fetch from S3 using range request
        response = self.s3_client.get_object(
            Bucket=self.bucket,
            Key=self.key,
            Range=f'bytes={start}-{end}'
        )
        data = response['Body'].read()
        
        self._pos += len(data)
        return data
    
    def __len__(self) -> int:
        """Return file size."""
        return self._size
    
    @property
    def size(self) -> int:
        """Return file size."""
        return self._size


class S3ZipFile:
    """
    Read ZIP files directly from S3 without downloading.
    
    Uses range requests to read only the parts of the ZIP needed.
    """
    
    def __init__(self, s3_client, bucket: str, key: str):
        """
        Initialize S3ZipFile.
        
        Args:
            s3_client: boto3 S3 client
            bucket: S3 bucket name
            key: S3 object key
        """
        self.s3_client = s3_client
        self.bucket = bucket
        self.key = key
        self.s3_file = S3File(s3_client, bucket, key)
        self._zip_file = None
        self._namelist = None
        self._infolist = None
    
    def _get_zip(self) -> zipfile.ZipFile:
        """Get or create the ZipFile object."""
        if self._zip_file is None:
            self._zip_file = zipfile.ZipFile(self.s3_file, 'r')
        return self._zip_file
    
    def namelist(self) -> list[str]:
        """List all files in the ZIP."""
        if self._namelist is None:
            self._namelist = self._get_zip().namelist()
        return self._namelist
    
    def infolist(self) -> list[zipfile.ZipInfo]:
        """Get ZipInfo for all files."""
        if self._infolist is None:
            self._infolist = self._get_zip().infolist()
        return self._infolist
    
    def read(self, name: str) -> bytes:
        """Read a file from the ZIP."""
        return self._get_zip().read(name)
    
    def close(self):
        """Close the ZIP file."""
        if self._zip_file is not None:
            self._zip_file.close()
            self._zip_file = None


class CheXpertZipProcessor:
    """Process CheXpert images from ZIP files stored on S3 (streaming, no local download)."""
    
    def __init__(
        self, 
        model_dir: str = './cxr_models',
        s3_client=None,
        output_to_s3: bool = False,
        s3_output_bucket: str = None,
        s3_output_prefix: str = None
    ):
        """
        Initialize the processor.
        
        Args:
            model_dir: Directory containing CXR Foundation models
            s3_client: Optional boto3 S3 client
            output_to_s3: If True, save embeddings to S3 instead of local
            s3_output_bucket: S3 bucket for output (required if output_to_s3=True)
            s3_output_prefix: S3 prefix for output files
        """
        self.s3_client = s3_client or boto3.client('s3')
        self.extractor = CXREmbeddingExtractor(model_dir=model_dir, s3_client=self.s3_client)
        self.output_to_s3 = output_to_s3
        self.s3_output_bucket = s3_output_bucket
        self.s3_output_prefix = s3_output_prefix
    
    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """Parse S3 URI into bucket and key."""
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        path = s3_uri[5:]
        bucket, key = path.split('/', 1)
        return bucket, key
    
    def list_images_in_s3_zip(
        self, 
        s3_uri: str, 
        extensions: list[str] = None
    ) -> tuple[S3ZipFile, list[str]]:
        """
        List all image files inside a ZIP archive on S3.
        
        Args:
            s3_uri: S3 URI of the ZIP file
            extensions: List of valid extensions (default: common image formats)
            
        Returns:
            Tuple of (S3ZipFile object, list of image paths)
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        bucket, key = self._parse_s3_uri(s3_uri)
        s3_zip = S3ZipFile(self.s3_client, bucket, key)
        
        image_paths = []
        for name in s3_zip.namelist():
            # Skip directories and hidden files
            if name.endswith('/') or '/__MACOSX' in name or name.startswith('.'):
                continue
            if any(name.lower().endswith(ext.lower()) for ext in extensions):
                image_paths.append(name)
        
        return s3_zip, sorted(image_paths)
    
    def load_image_from_s3_zip(self, s3_zip: S3ZipFile, image_name: str) -> Image.Image:
        """
        Load a single image from a ZIP file on S3.
        
        Args:
            s3_zip: S3ZipFile object
            image_name: Name of the image file inside the ZIP
            
        Returns:
            PIL Image object
        """
        image_bytes = s3_zip.read(image_name)
        img = Image.open(io.BytesIO(image_bytes))
        return img.copy()
    
    def _process_image_from_s3_zip(
        self, 
        s3_zip: S3ZipFile, 
        image_name: str
    ) -> dict[str, np.ndarray]:
        """
        Process a single image from a ZIP file on S3 and extract embeddings.
        
        Args:
            s3_zip: S3ZipFile object
            image_name: Name of the image file inside the ZIP
            
        Returns:
            Dictionary with embeddings and metadata
        """
        # Load image from ZIP on S3
        img = self.load_image_from_s3_zip(s3_zip, image_name)
        
        # Convert to grayscale
        img = img.convert('L')
        img_array = np.array(img, dtype=np.uint8)
        
        # Use the extractor's internal method to process
        from extract_cxr_features import png_to_tfexample
        import tensorflow as tf
        
        tf_example = png_to_tfexample(img_array)
        serialized_tf_example = tf_example.SerializeToString()
        
        # Get embeddings (thread-safe with lock)
        with self.extractor.model_lock:
            vision_output = self.extractor.elixrc_model.signatures['serving_default'](
                input_example=tf.constant([serialized_tf_example])
            )
            image_feature = vision_output['feature_maps_0'].numpy()
            
            if image_feature.shape != (1, 8, 8, 1376):
                if image_feature.size == 8 * 8 * 1376:
                    image_feature = image_feature.reshape(1, 8, 8, 1376)
                else:
                    raise ValueError(f"Unexpected image feature shape: {image_feature.shape}")
            
            tokens = np.zeros([1, 1, 128], dtype=np.int32)
            paddings = np.ones([1, 1, 128], dtype=np.float32)
            
            qformer_input = {
                'image_feature': tf.constant(image_feature, dtype=tf.float32),
                'ids': tf.constant(tokens, dtype=tf.int32),
                'paddings': tf.constant(paddings, dtype=tf.float32),
            }
            
            qformer_output = self.extractor.qformer_model.signatures['serving_default'](**qformer_input)
            
            embedding = {
                'image_path': image_name,
                'contrastive_img_emb': qformer_output['contrastive_img_emb'].numpy(),
                'all_contrastive_img_emb': qformer_output['all_contrastive_img_emb'].numpy(),
            }
            
            return embedding
    
    def _get_safe_filename(self, image_path: str) -> str:
        """Extract a safe filename from image path."""
        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            base_name = base_name.replace(char, '_')
        
        return base_name
    
    def _save_embedding(self, embedding: dict, output_dir: str) -> str:
        """Save a single embedding to .npz file (local or S3)."""
        base_name = self._get_safe_filename(embedding['image_path'])
        
        # Create npz data in memory
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            contrastive_img_emb=embedding['contrastive_img_emb'],
            all_contrastive_img_emb=embedding['all_contrastive_img_emb'],
            image_path=embedding['image_path']
        )
        buffer.seek(0)
        
        if self.output_to_s3:
            # Save to S3
            s3_key = f"{self.s3_output_prefix}/{output_dir}/{base_name}.npz"
            self.s3_client.put_object(
                Bucket=self.s3_output_bucket,
                Key=s3_key,
                Body=buffer.getvalue()
            )
            return f"s3://{self.s3_output_bucket}/{s3_key}"
        else:
            # Save locally
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base_name}.npz")
            with open(output_path, 'wb') as f:
                f.write(buffer.getvalue())
            return output_path
    
    def _get_processed_files(self, output_dir: str) -> set[str]:
        """Get set of already processed file basenames."""
        processed = set()
        
        if self.output_to_s3:
            # List from S3
            s3_prefix = f"{self.s3_output_prefix}/{output_dir}/"
            paginator = self.s3_client.get_paginator('list_objects_v2')
            try:
                for page in paginator.paginate(Bucket=self.s3_output_bucket, Prefix=s3_prefix):
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            key = obj['Key']
                            if key.endswith('.npz'):
                                basename = os.path.basename(key)[:-4]
                                processed.add(basename)
            except Exception:
                pass
        else:
            # List from local
            if os.path.exists(output_dir):
                for f in os.listdir(output_dir):
                    if f.endswith('.npz'):
                        processed.add(f[:-4])
        
        return processed
    
    def process_s3_zip(
        self,
        s3_uri: str,
        output_dir: str,
        batch_size: int = 100,
        resume: bool = True,
        limit: int = None
    ) -> list[str]:
        """
        Process images from a ZIP file on S3 (streaming, no download).
        
        Args:
            s3_uri: S3 URI of the ZIP file
            output_dir: Directory/prefix to save embeddings
            batch_size: Number of images to process before saving
            resume: If True, skip already processed images
            limit: Process only this many images (for testing)
            
        Returns:
            List of saved embedding file paths
        """
        print(f"Opening ZIP file from S3: {s3_uri}")
        print("(No download required - streaming directly from S3)")
        
        # Open ZIP from S3 and list images
        s3_zip, all_images = self.list_images_in_s3_zip(s3_uri)
        print(f"Found {len(all_images)} images in ZIP")
        
        if limit:
            all_images = all_images[:limit]
            print(f"Limited to {limit} images")
        
        # Filter already processed
        if resume:
            processed = self._get_processed_files(output_dir)
            images_to_process = [
                img for img in all_images 
                if self._get_safe_filename(img) not in processed
            ]
            print(f"Resuming: {len(processed)} already processed, {len(images_to_process)} remaining")
        else:
            images_to_process = all_images
        
        if not images_to_process:
            print("All images already processed!")
            s3_zip.close()
            return []
        
        saved_paths = []
        
        try:
            with tqdm(total=len(images_to_process), desc="Processing images") as pbar:
                for i in range(0, len(images_to_process), batch_size):
                    batch = images_to_process[i:i + batch_size]
                    
                    for img_name in batch:
                        try:
                            embedding = self._process_image_from_s3_zip(s3_zip, img_name)
                            save_path = self._save_embedding(embedding, output_dir)
                            saved_paths.append(save_path)
                        except Exception as e:
                            print(f"\nError processing {img_name}: {e}")
                            continue
                        pbar.update(1)
        finally:
            s3_zip.close()
        
        print(f"Saved {len(saved_paths)} embeddings to {output_dir}")
        return saved_paths


def main():
    """Main entry point for processing CheXpert ZIP files."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract CXR features from CheXpert ZIP files on S3 (streaming, no local download)'
    )
    parser.add_argument(
        '--s3-uri',
        type=str,
        help='S3 URI of the ZIP file to process'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./embeddings/chexpert',
        help='Local directory to save embeddings (ignored if --output-to-s3)'
    )
    parser.add_argument(
        '--output-to-s3',
        action='store_true',
        help='Save embeddings to S3 instead of local disk'
    )
    parser.add_argument(
        '--s3-output-bucket',
        type=str,
        default='n3c-medical-imaging',
        help='S3 bucket for output embeddings'
    )
    parser.add_argument(
        '--s3-output-prefix',
        type=str,
        default='ndonyapour/chexpert_embeddings',
        help='S3 prefix for output embeddings'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of images to process before saving'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of images to process (for testing)'
    )
    parser.add_argument(
        '--process-all',
        action='store_true',
        help='Process all CheXpert batches'
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = CheXpertZipProcessor(
        model_dir='./cxr_models',
        output_to_s3=args.output_to_s3,
        s3_output_bucket=args.s3_output_bucket,
        s3_output_prefix=args.s3_output_prefix
    )
    
    start_time = time.time()
    
    if args.process_all:
        # Process all CheXpert batches
        s3_base = 's3://n3c-medical-imaging/ndonyapour/chexpertchestxrays-u20210408'
        zip_files = [
            'CheXpert-v1.0 batch 1 (validate & csv).zip',
            'CheXpert-v1.0 batch 2 (train 1).zip',
            'CheXpert-v1.0 batch 3 (train 2).zip',
            'CheXpert-v1.0 batch 4 (train 3).zip',
        ]
        
        for i, zip_name in enumerate(zip_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing batch {i}/{len(zip_files)}: {zip_name}")
            print('='*60)
            
            s3_uri = f"{s3_base}/{zip_name}"
            
            if args.output_to_s3:
                output_subdir = f"batch_{i}"
            else:
                output_subdir = os.path.join(args.output_dir, f'batch_{i}')
            
            processor.process_s3_zip(
                s3_uri=s3_uri,
                output_dir=output_subdir,
                batch_size=args.batch_size,
                resume=True,
                limit=args.limit
            )
    
    elif args.s3_uri:
        # Process single ZIP file
        processor.process_s3_zip(
            s3_uri=args.s3_uri,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            resume=True,
            limit=args.limit
        )
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("STREAMING MODE - No local storage required for ZIP files!")
        print("="*60)
        print("\nExample usage:")
        print("\n  # Process a single ZIP file (save embeddings locally):")
        print("  python extrcat_cxr_fearures_CheXpert.py --s3-uri 's3://n3c-medical-imaging/ndonyapour/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv).zip'")
        print("\n  # Process all CheXpert batches (save locally):")
        print("  python extrcat_cxr_fearures_CheXpert.py --process-all")
        print("\n  # Process all and save embeddings to S3 (no local storage needed at all):")
        print("  python extrcat_cxr_fearures_CheXpert.py --process-all --output-to-s3")
        print("\n  # Test with limited images:")
        print("  python extrcat_cxr_fearures_CheXpert.py --process-all --limit 10")
        return
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/3600:.2f} hours ({elapsed:.0f} seconds)")


if __name__ == "__main__":
    main()

