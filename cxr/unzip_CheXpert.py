"""
Extract CheXpert ZIP files from S3 to S3 (streaming, no local storage required).

This script reads ZIP files directly from S3 using range requests and uploads
each extracted file back to S3 without needing any local disk space.
"""

import io
import zipfile

import boto3
from tqdm import tqdm


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
    
    def seekable(self) -> bool:
        """Return True - this file supports seeking."""
        return True
    
    def readable(self) -> bool:
        """Return True - this file supports reading."""
        return True
    
    def writable(self) -> bool:
        """Return False - this file does not support writing."""
        return False


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
    
    def read(self, name: str) -> bytes:
        """Read a file from the ZIP."""
        return self._get_zip().read(name)
    
    def close(self):
        """Close the ZIP file."""
        if self._zip_file is not None:
            self._zip_file.close()
            self._zip_file = None


def extract_s3_zip_to_s3(
    s3_client,
    source_uri: str,
    dest_bucket: str,
    dest_prefix: str,
    extensions: list[str] = None,
    include_csv: bool = True
) -> int:
    """
    Extract a ZIP file from S3 to another S3 location (streaming).
    
    Args:
        s3_client: boto3 S3 client
        source_uri: S3 URI of the ZIP file (e.g., 's3://bucket/path/file.zip')
        dest_bucket: Destination S3 bucket
        dest_prefix: Destination prefix (folder) in S3
        extensions: Only extract files with these extensions (None = images only)
        include_csv: Also extract .csv files
        
    Returns:
        Number of files extracted
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    if include_csv:
        extensions = extensions + ['.csv', '.CSV']
    
    # Parse source URI
    source_path = source_uri[5:]  # Remove 's3://'
    source_bucket, source_key = source_path.split('/', 1)
    
    # Open ZIP from S3
    print(f"Opening ZIP: {source_uri}")
    print(f"Destination: s3://{dest_bucket}/{dest_prefix}/")
    s3_zip = S3ZipFile(s3_client, source_bucket, source_key)
    
    # Get list of files to extract
    all_names = s3_zip.namelist()
    files_to_extract = [
        name for name in all_names
        if not name.endswith('/')  # Skip directories
        and '/__MACOSX' not in name  # Skip macOS metadata
        and not name.startswith('__MACOSX')
        and any(name.lower().endswith(ext.lower()) for ext in extensions)
    ]
    
    print(f"Found {len(files_to_extract)} files to extract (out of {len(all_names)} total entries)")
    
    extracted_count = 0
    errors = []
    
    # Extract each file
    for filename in tqdm(files_to_extract, desc="Extracting to S3"):
        try:
            # Read file from ZIP (streams just this file's bytes)
            data = s3_zip.read(filename)
            
            # Strip the original ZIP folder prefix if present
            # e.g., "CheXpert-v1.0 batch 1 (validate & csv)/valid/..." -> "valid/..."
            clean_filename = filename
            for prefix in [
                'CheXpert-v1.0 batch 1 (validate & csv)/',
                'CheXpert-v1.0 batch 2 (train 1)/',
                'CheXpert-v1.0 batch 3 (train 2)/',
                'CheXpert-v1.0 batch 4 (train 3)/',
            ]:
                if filename.startswith(prefix):
                    clean_filename = filename[len(prefix):]
                    break
            
            # Upload to destination
            dest_key = f"{dest_prefix}/{clean_filename}"
            s3_client.put_object(
                Bucket=dest_bucket,
                Key=dest_key,
                Body=data
            )
            extracted_count += 1
            
        except Exception as e:
            errors.append((filename, str(e)))
            if len(errors) <= 5:
                print(f"\nError extracting {filename}: {e}")
    
    s3_zip.close()
    
    if errors:
        print(f"\nCompleted with {len(errors)} errors")
    
    print(f"Successfully extracted {extracted_count} files to s3://{dest_bucket}/{dest_prefix}/")
    return extracted_count


def main():
    """Extract all CheXpert ZIP files to S3."""
    s3 = boto3.client('s3')
    
    # Configuration
    source_bucket = 'n3c-medical-imaging'
    source_prefix = 'ndonyapour/chexpertchestxrays-u20210408'
    dest_bucket = 'n3c-medical-imaging'
    dest_prefix = 'ndonyapour/chexpertchestxrays-u20210408'  # Same location, will create folders
    
    # ZIP files to extract: (source_zip_name, destination_folder_name)
    zip_files = [
        #('CheXpert-v1.0 batch 1 (validate & csv).zip', 'chexpert_v1.0_batch_1_validate'),
        #('CheXpert-v1.0 batch 2 (train 1).zip', 'chexpert_v1.0_batch_2_train'),
        #('CheXpert-v1.0 batch 3 (train 2).zip', 'chexpert_v1.0_batch_3_train'),
        ('CheXpert-v1.0 batch 4 (train 3).zip', 'chexpert_v1.0_batch_4_train'),
    ]
    
    total_extracted = 0
    
    for i, (zip_name, folder_name) in enumerate(zip_files, 1):
        print(f"\n{'='*70}")
        print(f"Processing batch {i}/{len(zip_files)}: {zip_name}")
        print(f"Destination folder: {folder_name}")
        print('='*70)
        
        source_uri = f"s3://{source_bucket}/{source_prefix}/{zip_name}"
        batch_dest_prefix = f"{dest_prefix}/{folder_name}"
        
        count = extract_s3_zip_to_s3(
            s3_client=s3,
            source_uri=source_uri,
            dest_bucket=dest_bucket,
            dest_prefix=batch_dest_prefix,
            include_csv=True
        )
        total_extracted += count
    
    print(f"\n{'='*70}")
    print(f"COMPLETE: Extracted {total_extracted} total files")
    print('='*70)


if __name__ == "__main__":
    main()

