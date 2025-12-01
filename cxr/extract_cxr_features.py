import os
import io
import png
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as tf_hub
from PIL import Image
from pathlib import Path


import png
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as tf_hub
import numpy as np

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
    
    def __init__(self, model_dir: str = './cxr_models'):
        """
        Initialize CXR embedding extractor.
        
        Args:
            model_dir: Directory containing downloaded CXR Foundation models
        """
        self.model_dir = model_dir
        self.qformer_model = None
        self.vision_model = None
        self._load_models()
    
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
            self.vision_model = tf.saved_model.load(vision_path)
            print(f"Loaded ELIXR C vision model from {vision_path}")
            print(f"Vision model signatures: {list(self.vision_model.signatures.keys())}")
            # Inspect the signature to understand input/output structure
            if 'serving_default' in self.vision_model.signatures:
                sig = self.vision_model.signatures['serving_default']
                print(f"Vision model input: {sig.structured_input_signature}")
                print(f"Vision model output: {sig.structured_outputs}")
        else:
            print(f"Warning: ELIXR C model not found at {vision_path}. Image-only embeddings may not work correctly.")
    
    def _prepare_image_for_elixr(self, image_path: str) -> bytes:
        """
        Prepare image for ELIXR C vision encoder (requires TF Example format).
        
        Args:
            image_path: Path to image file
            
        Returns:
            Serialized TF Example bytes
        """
        # Load and convert to grayscale
        img = Image.open(image_path).convert('L')
        
        # Convert to numpy array (uint8, 0-255)
        img_array = np.array(img, dtype=np.uint8)
        
        # Convert to TF Example format (required by ELIXR C)
        tf_example = png_to_tfexample(img_array)
        serialized = tf_example.SerializeToString()
        
        return serialized
    
    def get_image_embeddings(self, image_paths: list[str]) -> np.ndarray:
        """
        Extract embeddings from images using the full pipeline:
        1. ELIXR C vision encoder (image -> image features)
        2. QFormer (image features + text -> embeddings)
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Array of embeddings (n_images, embedding_dim)
        """
        if self.vision_model is None:
            raise ValueError("ELIXR C vision model not loaded. Cannot extract image embeddings.")
        
        embeddings_list = []
        
        for image_path in image_paths:
            # Step 1: Convert image to TF Example format
            serialized_tf_example = self._prepare_image_for_elixr(image_path)
            
            # Step 2: Get image features from ELIXR C vision encoder
            # Use the serving_default signature, similar to QFormer
            vision_output = self.vision_model.signatures['serving_default'](
                input_example=tf.constant([serialized_tf_example])
            )
            image_feature = vision_output['feature_maps_0'].numpy()
            # Ensure correct shape for QFormer (1, 8, 8, 1376)
            if len(image_feature.shape) == 2:
                # If it's 2D, might need to reshape
                # This depends on the actual ELIXR C output shape
                print(f"Warning: Image feature shape {image_feature.shape} is 2D, expected 4D (1, 8, 8, 1376)")
            elif image_feature.shape != (1, 8, 8, 1376):
                print(f"Warning: Image feature shape {image_feature.shape} doesn't match expected (1, 8, 8, 1376)")
                # Try to reshape if dimensions match
                if image_feature.size == 8 * 8 * 1376:
                    image_feature = image_feature.reshape(1, 8, 8, 1376)
            
            # Step 3: Prepare QFormer input with image features
            # Use zeros for text input (image-only mode)
            tokens = np.zeros([1, 1, 128], dtype=np.int32)
            paddings = np.ones([1, 1, 128], dtype=np.float32)
            
            qformer_input = {
                'image_feature': image_feature.tolist(),
                'ids': tokens.tolist(),
                'paddings': paddings.tolist(),
            }
            
            # Step 4: Get final embeddings from QFormer
            qformer_output = self.qformer_model.signatures['serving_default'](**qformer_input)
            # contrastive_img_emb    
            embedding = qformer_output['all_contrastive_img_emb'].numpy()
            embeddings_list.append(embedding[0])
        
        return np.array(embeddings_list)
    
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

# Usage example
if __name__ == "__main__":
    # Initialize extractor (models must be downloaded first)
    extractor = CXREmbeddingExtractor(model_dir='./cxr_models')
    
    # Extract embeddings from images
    image_paths = ['Chest_Xray_PA_3-8-2010.png']
    embeddings = extractor.get_image_embeddings(image_paths)
    print(f"Extracted embeddings shape: {embeddings.shape}")
