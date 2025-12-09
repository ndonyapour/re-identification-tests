import os
from botocore.exceptions import ClientError
import time
import logging
from cxr.cxr_embedding_extractor import CXREmbeddingExtractor



logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Usage example
if __name__ == "__main__":
    start_time = time.time()
    # Initialize extractor
    extractor = CXREmbeddingExtractor(model_dir='./cxr_models')
    
    s3_prefix = 's3://n3c-medical-imaging/ndonyapour/ChestXray-NIHCC'
    output_dir = './ChestX-ray14_embeddings'
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