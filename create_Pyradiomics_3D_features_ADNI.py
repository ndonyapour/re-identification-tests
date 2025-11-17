import os 
from pathlib import Path
from utils_pyradiomics import  PyRadiomicsExtractor

if __name__ == "__main__":
      # Get available 3D features    

 
    input_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images"
    mask_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/masks"
    output_dir = "/home/ubuntu/data/ADNI_dataset/Pyradiomics_features"

    mask_pattern = "_mask"
    image_files = [Path(f) for f in os.listdir(input_dir) if f.endswith(('.nii', '.nii.gz'))]
    mask_files =[Path(f).stem.split(".")[0]+mask_pattern+".nii.gz" for f in image_files]
    csv_files = [Path(f).stem.split(".")[0]+".csv" for f in image_files]


    image_paths = [str(Path(input_dir).joinpath(f)) for f in image_files]
    mask_paths = [str(Path(mask_dir).joinpath(f)) for f in mask_files]
    csv_paths = [str(Path(output_dir).joinpath(f)) for f in csv_files]
    import pdb; pdb.set_trace()
    extractor_3d = PyRadiomicsExtractor(dimension='3D', bin_width=5, voxel_array_shift=0, normalize=False, 
    normalize_scale=100, interpolator='sitkBSpline', resample_pixel_spacing=None)
    config_3d = extractor_3d.create_pyradiomics_config(output_file='ADNI_config_3d.yaml')

    # import pdb; pdb.set_trace()
    #extractor_3d.extract_features_from_files(str(image_paths[0]), str(mask_paths[0]), str(csv_paths[0]), label=1)
    extractor_3d.extract_features_parallel(image_paths, mask_paths, csv_paths, num_workers=10, label=1)