import os
import io
import png
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as tf_hub
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import boto3
from botocore.exceptions import ClientError
import time

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Helper function for tokenizing text input
def bert_tokenize(text):
    """Tokenizes input text and returns token IDs and padding masks."""
    preprocessor = tf_hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    out = preprocessor(tf.constant([text.lower()]))
    ids = out['input_word_ids'].numpy().astype(np.int32)
    masks = out['input_mask'].numpy().astype(np.float32)
    paddings = 1.0 - masks
    end_token_idx = ids == 102
    ids[end_token_idx] = 0
    paddings[end_token_idx] = 1.0
    ids = np.expand_dims(ids, axis=1)
    paddings = np.expand_dims(paddings, axis=1)
    assert ids.shape == (1, 1, 128)
    assert paddings.shape == (1, 1, 128)
    return ids, paddings

# Helper function for processing image data
def png_to_tfexample(image_array: np.ndarray) -> tf.train.Example:
    """Creates a tf.train.Example from a NumPy array."""
    # Convert the image to float32 and shift the minimum value to zero
    image = image_array.astype(np.float32)
    image -= image.min()

    if image_array.dtype == np.uint8:
        # For uint8 images, no rescaling is needed
        pixel_array = image.astype(np.uint8)
        bitdepth = 8
    else:
        # For other data types, scale image to use the full 16-bit range
        max_val = image.max()
        if max_val > 0:
            image *= 65535 / max_val  # Scale to 16-bit range
        pixel_array = image.astype(np.uint16)
        bitdepth = 16

    # Ensure the array is 2-D (grayscale image)
    if pixel_array.ndim != 2:
        raise ValueError(f'Array must be 2-D. Actual dimensions: {pixel_array.ndim}')

    # Encode the array as a PNG image
    output = io.BytesIO()
    png.Writer(
        width=pixel_array.shape[1],
        height=pixel_array.shape[0],
        greyscale=True,
        bitdepth=bitdepth
    ).write(output, pixel_array.tolist())
    png_bytes = output.getvalue()

    # Create a tf.train.Example and assign the features
    example = tf.train.Example()
    features = example.features.feature
    features['image/encoded'].bytes_list.value.append(png_bytes)
    features['image/format'].bytes_list.value.append(b'png')

    return example


class CXREmbeddingExtractor:
    """Extract CXR Foundation embeddings from images using local models."""
    
    def __init__(self, model_dir: str = './cxr_models', s3_client=None):
        """
        Initialize CXR embedding extractor.
        
        Args:
            model_dir: Directory containing downloaded CXR Foundation models
            s3_client: Optional boto3 S3 client. If None, will create one when needed.
        """
        self.model_dir = model_dir
        self.qformer_model = None
        self.elixrc_model = None
        self.model_lock = threading.Lock()
        self.s3_client = s3_client
        self._load_models()
    
    def _get_s3_client(self):
        """Lazy initialization of S3 client."""
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
        return self.s3_client
    
    def _load_image_from_s3(self, s3_uri: str) -> Image.Image:
        """
        Load image directly from S3 into memory.
        
        Args:
            s3_uri: S3 URI in format 's3://bucket-name/path/to/image.png'
            
        Returns:
            PIL Image object
        """
        # Parse S3 URI
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {s3_uri}. Must start with 's3://'")
        
        s3_path = s3_uri[5:]  # Remove 's3://'
        bucket_name, key = s3_path.split('/', 1)
        
        # Download image to memory
        s3_client = self._get_s3_client()
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            image_bytes = response['Body'].read()
            
            # Load from bytes using PIL
            img = Image.open(io.BytesIO(image_bytes))
            return img
        except ClientError as e:
            raise FileNotFoundError(f"Failed to load image from S3: {s3_uri}. Error: {e}")
    
    def _prepare_image_for_elixr(self, image_path_or_uri: str) -> bytes:
        """
        Prepare image for ELIXR C vision encoder (requires TF Example format).
        Supports both local file paths and S3 URIs.
        
        Args:
            image_path_or_uri: Path to local image file or S3 URI (s3://bucket/path)
            
        Returns:
            Serialized TF Example bytes
        """
        # Check if it's an S3 URI or local file
        if image_path_or_uri.startswith('s3://'):
            # Load from S3
            img = self._load_image_from_s3(image_path_or_uri)
        else:
            # Load from local file
            img = Image.open(image_path_or_uri)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Convert to numpy array (uint8, 0-255)
        img_array = np.array(img, dtype=np.uint8)
        
        # Convert to TF Example format (required by ELIXR C)
        tf_example = png_to_tfexample(img_array)
        serialized = tf_example.SerializeToString()
        
        return serialized
    
    def list_s3_images(self, s3_prefix: str, file_extensions: list[str] = None) -> list[str]:
        """
        List all image files in an S3 bucket/prefix.
        
        Args:
            s3_prefix: S3 prefix like 's3://bucket-name/path/to/folder/' or 's3://bucket-name/'
            file_extensions: List of file extensions to filter (e.g., ['.png', '.jpg', '.jpeg'])
                            If None, uses common image extensions.
        
        Returns:
            List of S3 URIs for image files
        """
        if file_extensions is None:
            file_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        
        # Parse S3 prefix
        if not s3_prefix.startswith('s3://'):
            raise ValueError(f"Invalid S3 prefix: {s3_prefix}. Must start with 's3://'")
        
        s3_path = s3_prefix[5:]  # Remove 's3://'
        if '/' in s3_path:
            bucket_name, prefix = s3_path.split('/', 1)
            if not prefix.endswith('/'):
                prefix += '/'
        else:
            bucket_name = s3_path
            prefix = ''
        
        # List objects
        s3_client = self._get_s3_client()
        image_uris = []
        
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Check if file extension matches
                    if any(key.lower().endswith(ext.lower()) for ext in file_extensions):
                        image_uris.append(f's3://{bucket_name}/{key}')
        
        return image_uris
    
    def _load_models(self):
        """Load the CXR Foundation models."""
        # Load QFormer model for text/image embeddings
        qformer_path = os.path.join(self.model_dir, 'pax-elixr-b-text')
        if os.path.exists(qformer_path):
            self.qformer_model = tf.saved_model.load(qformer_path)
            print(f"Loaded QFormer model from {qformer_path}")
            print(f"QFormer signatures: {list(self.qformer_model.signatures.keys())}")
        else:
            raise FileNotFoundError(f"QFormer model not found at {qformer_path}. Run download_cxr_model.py first.")
        
        # Load ELIXR C vision encoder (for image preprocessing)
        vision_path = os.path.join(self.model_dir, 'elixr-c-v2-pooled')
        if os.path.exists(vision_path):
            self.elixrc_model = tf.saved_model.load(vision_path)
            print(f"Loaded ELIXR C vision model from {vision_path}")
            print(f"Vision model signatures: {list(self.elixrc_model.signatures.keys())}")
            # Inspect the signature to understand input/output structure
            if 'serving_default' in self.elixrc_model.signatures:
                sig = self.elixrc_model.signatures['serving_default']
                print(f"Vision model input: {sig.structured_input_signature}")
                print(f"Vision model output: {sig.structured_outputs}")
        else:
            print(f"Warning: ELIXR C model not found at {vision_path}. Image-only embeddings may not work correctly.")
    
    def get_image_embeddings(self, image_paths: list[str], max_workers: int = 4) -> list[dict[str, np.ndarray]]:
        """
        Extract embeddings with parallel preprocessing (I/O bound operations).
        Model inference is serialized to avoid GPU conflicts.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_single_image, path) for path in image_paths]
            embeddings_list = [future.result() for future in futures]
        return embeddings_list
    
    def _process_single_image(self, image_path: str) -> dict[str, np.ndarray]:
        """Process a single image (thread-safe)."""
        # Preprocessing can happen in parallel
        serialized_tf_example = self._prepare_image_for_elixr(image_path)
        
        # Model calls must be serialized (GPU access)
        with self.model_lock:
            # Step 1: Convert image to TF Example format
            # serialized_tf_example = self._prepare_image_for_elixr(image_path) # This line is now redundant
            
            # Step 2: Get image features from ELIXR C vision encoder
            vision_output = self.elixrc_model.signatures['serving_default'](
                input_example=tf.constant([serialized_tf_example])  # Shape: (1,)
            )
            image_feature = vision_output['feature_maps_0'].numpy()  # Shape: (1, 8, 8, 1376)
            
            # Ensure correct shape
            if image_feature.shape != (1, 8, 8, 1376):
                if image_feature.size == 8 * 8 * 1376:
                    image_feature = image_feature.reshape(1, 8, 8, 1376)
                else:
                    raise ValueError(f"Unexpected image feature shape: {image_feature.shape}")
            
            # Step 3: Prepare QFormer input (batch size must be 1)
            tokens = np.zeros([1, 1, 128], dtype=np.int32)
            paddings = np.ones([1, 1, 128], dtype=np.float32)
            
            # Pass as TensorFlow tensors (not lists!)
            qformer_input = {
                'image_feature': tf.constant(image_feature, dtype=tf.float32),
                'ids': tf.constant(tokens, dtype=tf.int32),
                'paddings': tf.constant(paddings, dtype=tf.float32),
            }
            
            # Step 4: Get final embeddings from QFormer
            qformer_output = self.qformer_model.signatures['serving_default'](**qformer_input)
            
            # Extract embeddings
            embedding = {
                'image_path': image_path,
                'contrastive_img_emb': qformer_output['contrastive_img_emb'].numpy(),
                'all_contrastive_img_emb': qformer_output['all_contrastive_img_emb'].numpy(),
            }
            return embedding
    
    def get_text_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Extract embeddings from text queries.
        
        Args:
            texts: List of text queries
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        embeddings_list = []
        
        for text in texts:
            tokens, paddings = self._bert_tokenize(text)
            
            # Use zeros for image input
            image_feature = np.zeros([1, 8, 8, 1376], dtype=np.float32)
            
            qformer_input = {
                'image_feature': image_feature.tolist(),
                'ids': tokens.tolist(),
                'paddings': paddings.tolist(),
            }
            
            qformer_output = self.qformer_model.signatures['serving_default'](**qformer_input)
            embedding = qformer_output['contrastive_txt_emb'].numpy()
            embeddings_list.append(embedding)
        
        return np.array(embeddings_list)

    def _get_safe_filename(self, image_path: str) -> str:
        """
        Extract a safe filename from image path (handles S3 URIs and local paths).
        
        Args:
            image_path: Image path or S3 URI
            
        Returns:
            Safe filename without extension
        """
        if image_path.startswith('s3://'):
            # Extract filename from S3 path
            filename = os.path.basename(image_path)
        else:
            # Local file path
            filename = os.path.basename(image_path)
        
        # Remove extension
        base_name = os.path.splitext(filename)[0]
        
        # Replace invalid filename characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            base_name = base_name.replace(char, '_')
        
        return base_name
    
    def save_embedding_to_npz(self, embedding: dict, output_dir: str) -> str:
        """
        Save a single image's embeddings to a .npz file.
        
        Args:
            embedding: Dictionary with 'image_path', 'contrastive_img_emb', 'all_contrastive_img_emb'
            output_dir: Directory to save the .npz file
            
        Returns:
            Path to the saved .npz file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get safe filename from image path
        image_path = embedding.get('image_path', 'unknown')
        base_name = self._get_safe_filename(image_path)
        output_path = os.path.join(output_dir, f"{base_name}.npz")
        
        # Save embeddings as compressed .npz file
        np.savez_compressed(
            output_path,
            contrastive_img_emb=embedding['contrastive_img_emb'],
            all_contrastive_img_emb=embedding['all_contrastive_img_emb'],
            image_path=image_path  # Store original path as metadata
        )
        
        return output_path
    
    def save_embeddings_batch_to_npz(
        self, 
        embeddings: list[dict], 
        output_dir: str,
        verbose: bool = True
    ) -> list[str]:
        """
        Save a batch of embeddings, each as an individual .npz file.
        
        Args:
            embeddings: List of embedding dictionaries, each with 'image_path', 
                       'contrastive_img_emb', 'all_contrastive_img_emb'
            output_dir: Directory to save the .npz files
            verbose: If True, print progress
            
        Returns:
            List of paths to saved .npz files
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for i, embedding in enumerate(embeddings):
            saved_path = self.save_embedding_to_npz(embedding, output_dir)
            saved_paths.append(saved_path)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Saved {i + 1}/{len(embeddings)} embeddings to {output_dir}")
        
        if verbose:
            print(f"Saved {len(embeddings)} embeddings to {output_dir}")
        
        return saved_paths
    
    def get_image_embeddings_with_saving(
        self, 
        image_paths: list[str], 
        output_dir: str,
        max_workers: int = 4,
        save_batch_size: int = 100,
        save_individual: bool = True
    ) -> list[dict[str, np.ndarray]]:
        """
        Extract embeddings and save them in batches as individual .npz files.
        
        Args:
            image_paths: List of image paths/URIs
            output_dir: Directory to save .npz files
            max_workers: Number of parallel workers for preprocessing
            save_batch_size: Save embeddings after processing this many images
            save_individual: If True, save each embedding as separate .npz file
            
        Returns:
            List of embedding dictionaries
        """
        os.makedirs(output_dir, exist_ok=True)
        embeddings_list = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_single_image, path) for path in image_paths]
            
            for i, future in enumerate(futures):
                embedding = future.result()
                embeddings_list.append(embedding)
                
                # Save in batches
                if save_individual and (i + 1) % save_batch_size == 0:
                    # Get the batch to save
                    batch_start = max(0, len(embeddings_list) - save_batch_size)
                    batch = embeddings_list[batch_start:]
                    self.save_embeddings_batch_to_npz(batch, output_dir, verbose=True)
                    print(f"Processed {i + 1}/{len(image_paths)} images")
        
        # Save remaining embeddings
        if save_individual and len(embeddings_list) % save_batch_size != 0:
            batch_start = (len(embeddings_list) // save_batch_size) * save_batch_size
            remaining_batch = embeddings_list[batch_start:]
            self.save_embeddings_batch_to_npz(remaining_batch, output_dir, verbose=True)
        
        return embeddings_list

# Usage example
if __name__ == "__main__":
    start_time = time.time()
    # Initialize extractor
    extractor = CXREmbeddingExtractor(model_dir='./cxr_models')
    
    s3_prefix = 's3://n3c-medical-imaging/ndonyapour/ChestXray-NIHCC'
    output_dir = './embeddings'
    for i in range(2, 13):
        subdir = os.path.join(s3_prefix, f"images_{i:02d}")
        print(f"Processing {subdir}")
        image_uris = extractor.list_s3_images(subdir)
        print(f"Found {len(image_uris)} images in {subdir}")
        
        embeddings = extractor.get_image_embeddings_with_saving(
            image_uris, 
            output_dir=os.path.join(output_dir, f'features_{i:02d}'),
            max_workers=20,
            save_batch_size=100,  # Save every 100 images
            save_individual=True
        )
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")