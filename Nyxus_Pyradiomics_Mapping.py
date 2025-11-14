# Nyxus to PyRadiomics 3D Feature Mapping
NYXUS_TO_PYRADIOMICS_3D_MAPPING = {
    # ========== STATISTICAL/INTENSITY (Nyxus) ↔ FIRSTORDER (PyRadiomics) ==========
    "3MEAN": "original_firstorder_Mean",
    "3MIN": "original_firstorder_Minimum",
    "3MAX": "original_firstorder_Maximum",
    "3MEDIAN": "original_firstorder_Median",
    "3VARIANCE": "original_firstorder_Variance",
    "3SKEWNESS": "original_firstorder_Skewness",
    "3KURTOSIS": "original_firstorder_Kurtosis",
    "3ENTROPY": "original_firstorder_Entropy",
    "3ENERGY": "original_firstorder_Energy",
    "3ROOT_MEAN_SQUARED": "original_firstorder_RootMeanSquared",
    "3MEAN_ABSOLUTE_DEVIATION": "original_firstorder_MeanAbsoluteDeviation",
    "3RANGE": "original_firstorder_Range",
    "3INTERQUARTILE_RANGE": "original_firstorder_InterquartileRange",
    "3UNIFORMITY": "original_firstorder_Uniformity",
    "3P10": "original_firstorder_10Percentile",
    "3P90": "original_firstorder_90Percentile",
    "3ROBUST_MEAN_ABSOLUTE_DEVIATION": "original_firstorder_RobustMeanAbsoluteDeviation",
    "3TOTAL_ENERGY": "original_firstorder_TotalEnergy",
    
    # Note: Nyxus has additional statistical features not in PyRadiomics:
    # 3COV, 3COVERED_IMAGE_INTENSITY_RANGE, 3EXCESS_KURTOSIS, 3HYPERFLATNESS,
    # 3HYPERSKEWNESS, 3INTEGRATED_INTENSITY, 3MEDIAN_ABSOLUTE_DEVIATION, 3MODE,
    # 3P01, 3P25, 3P75, 3P99, 3QCOD, 3ROBUST_MEAN, 3STANDARD_DEVIATION,
    # 3STANDARD_DEVIATION_BIASED, 3STANDARD_ERROR, 3VARIANCE_BIASED, 3UNIFORMITY_PIU
    
    # ========== SHAPE (Nyxus) ↔ SHAPE (PyRadiomics) ==========
    "3ELONGATION": "original_shape_Elongation",
    "3FLATNESS": "original_shape_Flatness",
    "3MAJOR_AXIS_LEN": "original_shape_MajorAxisLength",
    "3MINOR_AXIS_LEN": "original_shape_MinorAxisLength",
    "3LEAST_AXIS_LEN": "original_shape_LeastAxisLength",
    "3VOXEL_VOLUME": "original_shape_VoxelVolume",
    "3MESH_VOLUME": "original_shape_MeshVolume",
    "3SURFACE_AREA": "original_shape_SurfaceArea",
    "3SPHERICITY": "original_shape_Sphericity",
    "3AREA_2_VOLUME": "original_shape_SurfaceVolumeRatio",
    
    # Note: PyRadiomics has additional shape features:
    # original_shape_Maximum2DDiameterColumn, original_shape_Maximum2DDiameterRow,
    # original_shape_Maximum2DDiameterSlice, original_shape_Maximum3DDiameter
    
    # Note: Nyxus has additional shape features not in PyRadiomics:
    # 3AREA, 3COMPACTNESS1, 3COMPACTNESS2, 3SPHERICAL_DISPROPORTION, 3VOLUME_CONVEXHULL
    
    # ========== TEXTURE: GLCM (Nyxus) ↔ GLCM (PyRadiomics) ==========
    "3GLCM_ACOR": "original_glcm_Autocorrelation",
    "3GLCM_CLUPROM": "original_glcm_ClusterProminence",
    "3GLCM_CLUSHADE": "original_glcm_ClusterShade",
    "3GLCM_CLUTEND": "original_glcm_ClusterTendency",
    "3GLCM_CONTRAST": "original_glcm_Contrast",
    "3GLCM_CORRELATION": "original_glcm_Correlation",
    "3GLCM_DIFAVE": "original_glcm_DifferenceAverage",
    "3GLCM_DIFENTRO": "original_glcm_DifferenceEntropy",
    "3GLCM_DIFVAR": "original_glcm_DifferenceVariance",
    "3GLCM_ID": "original_glcm_Id",
    "3GLCM_IDM": "original_glcm_Idm",
    "3GLCM_IDMN": "original_glcm_Idmn",
    "3GLCM_IDN": "original_glcm_Idn",
    "3GLCM_IMC1": "original_glcm_Imc1",
    "3GLCM_IMC2": "original_glcm_Imc2",
    "3GLCM_IV": "original_glcm_InverseVariance",
    "3GLCM_JAVE": "original_glcm_JointAverage",
    "3GLCM_JE": "original_glcm_JointEnergy",
    "3GLCM_JOINT_ENTROPY": "original_glcm_JointEntropy",
    "3GLCM_MCC": "original_glcm_MCC",
    "3GLCM_JMAX": "original_glcm_MaximumProbability",
    "3GLCM_SUMAVERAGE": "original_glcm_SumAverage",
    "3GLCM_SUMENTROPY": "original_glcm_SumEntropy",
    "3GLCM_SUMVARIANCE": "original_glcm_SumSquares",
    "3GLCM_VARIANCE": "original_glcm_SumSquares",  # Note: PyRadiomics uses SumSquares
    
    # Note: Nyxus has directional GLCM features (0, 45, 90, 135) and averaged versions
    # PyRadiomics provides single aggregated values for 3D
    
    # ========== TEXTURE: GLDM (Nyxus) ↔ GLDM (PyRadiomics) ==========
    "3GLDM_SDE": "original_gldm_SmallDependenceEmphasis",
    "3GLDM_LDE": "original_gldm_LargeDependenceEmphasis",
    "3GLDM_GLN": "original_gldm_GrayLevelNonUniformity",
    "3GLDM_DN": "original_gldm_DependenceNonUniformity",
    "3GLDM_DNN": "original_gldm_DependenceNonUniformityNormalized",
    "3GLDM_GLV": "original_gldm_GrayLevelVariance",
    "3GLDM_DV": "original_gldm_DependenceVariance",
    "3GLDM_DE": "original_gldm_DependenceEntropy",
    "3GLDM_LGLE": "original_gldm_LowGrayLevelEmphasis",
    "3GLDM_HGLE": "original_gldm_HighGrayLevelEmphasis",
    "3GLDM_SDLGLE": "original_gldm_SmallDependenceLowGrayLevelEmphasis",
    "3GLDM_SDHGLE": "original_gldm_SmallDependenceHighGrayLevelEmphasis",
    "3GLDM_LDLGLE": "original_gldm_LargeDependenceLowGrayLevelEmphasis",
    "3GLDM_LDHGLE": "original_gldm_LargeDependenceHighGrayLevelEmphasis",
    
    # ========== TEXTURE: GLRLM (Nyxus) ↔ GLRLM (PyRadiomics) ==========
    "3GLRLM_SRE": "original_glrlm_ShortRunEmphasis",
    "3GLRLM_LRE": "original_glrlm_LongRunEmphasis",
    "3GLRLM_GLN": "original_glrlm_GrayLevelNonUniformity",
    "3GLRLM_GLNN": "original_glrlm_GrayLevelNonUniformityNormalized",
    "3GLRLM_RLN": "original_glrlm_RunLengthNonUniformity",
    "3GLRLM_RLNN": "original_glrlm_RunLengthNonUniformityNormalized",
    "3GLRLM_RP": "original_glrlm_RunPercentage",
    "3GLRLM_GLV": "original_glrlm_GrayLevelVariance",
    "3GLRLM_RV": "original_glrlm_RunVariance",
    "3GLRLM_RE": "original_glrlm_RunEntropy",
    "3GLRLM_LGLRE": "original_glrlm_LowGrayLevelRunEmphasis",
    "3GLRLM_HGLRE": "original_glrlm_HighGrayLevelRunEmphasis",
    "3GLRLM_SRLGLE": "original_glrlm_ShortRunLowGrayLevelEmphasis",
    "3GLRLM_SRHGLE": "original_glrlm_ShortRunHighGrayLevelEmphasis",
    "3GLRLM_LRLGLE": "original_glrlm_LongRunLowGrayLevelEmphasis",
    "3GLRLM_LRHGLE": "original_glrlm_LongRunHighGrayLevelEmphasis",
    
    # Note: Nyxus has directional GLRLM features (0, 45, 90, 135) and averaged versions
    
    # ========== TEXTURE: GLSZM (Nyxus) ↔ GLSZM (PyRadiomics) ==========
    "3GLSZM_SAE": "original_glszm_SmallAreaEmphasis",
    "3GLSZM_LAE": "original_glszm_LargeAreaEmphasis",
    "3GLSZM_GLN": "original_glszm_GrayLevelNonUniformity",
    "3GLSZM_GLNN": "original_glszm_GrayLevelNonUniformityNormalized",
    "3GLSZM_SZN": "original_glszm_SizeZoneNonUniformity",
    "3GLSZM_SZNN": "original_glszm_SizeZoneNonUniformityNormalized",
    "3GLSZM_ZP": "original_glszm_ZonePercentage",
    "3GLSZM_GLV": "original_glszm_GrayLevelVariance",
    "3GLSZM_ZV": "original_glszm_ZoneVariance",
    "3GLSZM_ZE": "original_glszm_ZoneEntropy",
    "3GLSZM_LGLZE": "original_glszm_LowGrayLevelZoneEmphasis",
    "3GLSZM_HGLZE": "original_glszm_HighGrayLevelZoneEmphasis",
    "3GLSZM_SALGLE": "original_glszm_SmallAreaLowGrayLevelEmphasis",
    "3GLSZM_SAHGLE": "original_glszm_SmallAreaHighGrayLevelEmphasis",
    "3GLSZM_LALGLE": "original_glszm_LargeAreaLowGrayLevelEmphasis",
    "3GLSZM_LAHGLE": "original_glszm_LargeAreaHighGrayLevelEmphasis",
    
    # ========== TEXTURE: NGTDM (Nyxus) ↔ NGTDM (PyRadiomics) ==========
    "3NGTDM_COARSENESS": "original_ngtdm_Coarseness",
    "3NGTDM_CONTRAST": "original_ngtdm_Contrast",
    "3NGTDM_BUSYNESS": "original_ngtdm_Busyness",
    "3NGTDM_COMPLEXITY": "original_ngtdm_Complexity",
    "3NGTDM_STRENGTH": "original_ngtdm_Strength",
}

# Features that exist in Nyxus but not in PyRadiomics
NYXUS_ONLY_3D_FEATURES = [
    # Statistical
    "3COV", "3COVERED_IMAGE_INTENSITY_RANGE", "3EXCESS_KURTOSIS", "3HYPERFLATNESS",
    "3HYPERSKEWNESS", "3INTEGRATED_INTENSITY", "3MEDIAN_ABSOLUTE_DEVIATION", "3MODE",
    "3P01", "3P25", "3P75", "3P99", "3QCOD", "3ROBUST_MEAN", "3STANDARD_DEVIATION",
    "3STANDARD_DEVIATION_BIASED", "3STANDARD_ERROR", "3VARIANCE_BIASED", "3UNIFORMITY_PIU",
    # Shape
    "3AREA", "3COMPACTNESS1", "3COMPACTNESS2", "3SPHERICAL_DISPROPORTION", "3VOLUME_CONVEXHULL",
    # Texture - GLDZM (not in PyRadiomics)
    "3GLDZM_SDE", "3GLDZM_LDE", "3GLDZM_LGLZE", "3GLDZM_HGLZE", "3GLDZM_SDLGLE",
    "3GLDZM_SDHGLE", "3GLDZM_LDLGLE", "3GLDZM_LDHGLE", "3GLDZM_GLNU", "3GLDZM_GLNUN",
    "3GLDZM_ZDNU", "3GLDZM_ZDNUN", "3GLDZM_ZP", "3GLDZM_GLM", "3GLDZM_GLV", "3GLDZM_ZDM",
    "3GLDZM_ZDV", "3GLDZM_ZDE",
    # Texture - NGLDM (not in PyRadiomics)
    "3NGLDM_LDE", "3NGLDM_HDE", "3NGLDM_LGLCE", "3NGLDM_HGLCE", "3NGLDM_LDLGLE", "3NGLDM_LDHGLE",
    "3NGLDM_HDLGLE", "3NGLDM_HDHGLE", "3NGLDM_GLNU", "3NGLDM_GLNUN", "3NGLDM_DCNU", "3NGLDM_DCNUN",
    "3NGLDM_DCP", "3NGLDM_GLM", "3NGLDM_GLV", "3NGLDM_DCM", "3NGLDM_DCV", "3NGLDM_DCENT", "3NGLDM_DCENE",
    # Directional and averaged texture features
    "3GLCM_ASM", "3GLCM_ASM_AVE", "3GLCM_ACOR_AVE", "3GLCM_CLUPROM_AVE", "3GLCM_CLUSHADE_AVE",
    "3GLCM_CLUTEND_AVE", "3GLCM_CONTRAST_AVE", "3GLCM_CORRELATION_AVE", "3GLCM_DIFAVE_AVE",
    "3GLCM_DIFENTRO_AVE", "3GLCM_DIFVAR_AVE", "3GLCM_DIS", "3GLCM_DIS_AVE", "3GLCM_ENERGY",
    "3GLCM_ENERGY_AVE", "3GLCM_ENTROPY", "3GLCM_ENTROPY_AVE", "3GLCM_HOM1", "3GLCM_HOM1_AVE",
    "3GLCM_HOM2", "3GLCM_ID_AVE", "3GLCM_IDN", "3GLCM_IDN_AVE", "3GLCM_JVAR", "3GLCM_JVAR_AVE",
    "3GLCM_SUMVARIANCE_AVE", "3GLCM_VARIANCE_AVE",
    "3GLRLM_SRE_AVE", "3GLRLM_LRE_AVE", "3GLRLM_GLN_AVE", "3GLRLM_GLNN_AVE", "3GLRLM_RLN_AVE",
    "3GLRLM_RLNN_AVE", "3GLRLM_RP_AVE", "3GLRLM_GLV_AVE", "3GLRLM_RV_AVE", "3GLRLM_RE_AVE",
    "3GLRLM_LGLRE_AVE", "3GLRLM_HGLRE_AVE", "3GLRLM_SRLGLE_AVE", "3GLRLM_SRHGLE_AVE",
    "3GLRLM_LRLGLE_AVE", "3GLRLM_LRHGLE_AVE",
]

# Features that exist in PyRadiomics but not in Nyxus
PYRADIOMICS_ONLY_3D_FEATURES = [
    "original_shape_Maximum2DDiameterColumn",
    "original_shape_Maximum2DDiameterRow",
    "original_shape_Maximum2DDiameterSlice",
    "original_shape_Maximum3DDiameter",
]


# Nyxus to PyRadiomics 2D Feature Mapping
NYXUS_TO_PYRADIOMICS_2D_MAPPING = {
    # ========== STATISTICAL/INTENSITY (Nyxus) ↔ FIRSTORDER (PyRadiomics) ==========
    "MEAN": "original_firstorder_Mean",
    "MIN": "original_firstorder_Minimum",
    "MAX": "original_firstorder_Maximum",
    "MEDIAN": "original_firstorder_Median",
    "VARIANCE": "original_firstorder_Variance",
    "SKEWNESS": "original_firstorder_Skewness",
    "KURTOSIS": "original_firstorder_Kurtosis",
    "ENTROPY": "original_firstorder_Entropy",
    "ENERGY": "original_firstorder_Energy",
    "ROOT_MEAN_SQUARED": "original_firstorder_RootMeanSquared",
    "MEAN_ABSOLUTE_DEVIATION": "original_firstorder_MeanAbsoluteDeviation",
    "RANGE": "original_firstorder_Range",
    "INTERQUARTILE_RANGE": "original_firstorder_InterquartileRange",
    "UNIFORMITY": "original_firstorder_Uniformity",
    "P10": "original_firstorder_10Percentile",
    "P90": "original_firstorder_90Percentile",
    "ROBUST_MEAN_ABSOLUTE_DEVIATION": "original_firstorder_RobustMeanAbsoluteDeviation",
    "TOTAL_ENERGY": "original_firstorder_TotalEnergy",
    
    # Note: Nyxus has additional statistical features not in PyRadiomics:
    # COV, COVERED_IMAGE_INTENSITY_RANGE, EXCESS_KURTOSIS, HYPERFLATNESS,
    # HYPERSKEWNESS, INTEGRATED_INTENSITY, MEDIAN_ABSOLUTE_DEVIATION, MODE,
    # P01, P25, P75, P99, QCOD, ROBUST_MEAN, STANDARD_DEVIATION,
    # STANDARD_DEVIATION_BIASED, STANDARD_ERROR, VARIANCE_BIASED, UNIFORMITY_PIU
    
    # ========== SHAPE (Nyxus) ↔ SHAPE (PyRadiomics) ==========
    # Note: Nyxus 2D doesn't have direct shape features like 3D
    # PyRadiomics 2D shape features are the same as 3D
    
    # ========== TEXTURE: GLCM (Nyxus) ↔ GLCM (PyRadiomics) ==========
    # Note: Nyxus has directional GLCM features (0, 45, 90, 135) and averaged versions
    # PyRadiomics provides single aggregated values
    # Mapping averaged versions to PyRadiomics (closest match)
    "GLCM_ACOR_AVE": "original_glcm_Autocorrelation",
    "GLCM_CLUPROM_AVE": "original_glcm_ClusterProminence",
    "GLCM_CLUSHADE_AVE": "original_glcm_ClusterShade",
    "GLCM_CLUTEND_AVE": "original_glcm_ClusterTendency",
    "GLCM_CONTRAST_AVE": "original_glcm_Contrast",
    "GLCM_CORRELATION_AVE": "original_glcm_Correlation",
    "GLCM_DIFAVE_AVE": "original_glcm_DifferenceAverage",
    "GLCM_DIFENTRO_AVE": "original_glcm_DifferenceEntropy",
    "GLCM_DIFVAR_AVE": "original_glcm_DifferenceVariance",
    "GLCM_ID_AVE": "original_glcm_Id",
    "GLCM_IDM_AVE": "original_glcm_Idm",
    "GLCM_IDMN_AVE": "original_glcm_Idmn",
    "GLCM_IDN_AVE": "original_glcm_Idn",
    "GLCM_IMC1": "original_glcm_Imc1",  # Note: Nyxus uses INFOMEAS1/2, mapping to IMC1/2
    "GLCM_IMC2": "original_glcm_Imc2",
    "GLCM_INFOMEAS1_AVE": "original_glcm_Imc1",
    "GLCM_INFOMEAS2_AVE": "original_glcm_Imc2",
    "GLCM_IV_AVE": "original_glcm_InverseVariance",
    "GLCM_JAVE_AVE": "original_glcm_JointAverage",
    "GLCM_JE_AVE": "original_glcm_JointEnergy",
    "GLCM_JOINT_ENTROPY": "original_glcm_JointEntropy",  # Note: Nyxus doesn't have directional variants
    "GLCM_MCC": "original_glcm_MCC",  # Note: Nyxus doesn't have MCC, but included for completeness
    "GLCM_JMAX_AVE": "original_glcm_MaximumProbability",
    "GLCM_SUMAVERAGE_AVE": "original_glcm_SumAverage",
    "GLCM_SUMENTROPY_AVE": "original_glcm_SumEntropy",
    "GLCM_SUMVARIANCE_AVE": "original_glcm_SumSquares",
    "GLCM_VARIANCE_AVE": "original_glcm_SumSquares",
    
    # Additional GLCM features that map (using ASM for Energy)
    "GLCM_ASM_AVE": "original_glcm_JointEnergy",  # ASM (Angular Second Moment) ≈ JointEnergy
    "GLCM_ENERGY_AVE": "original_glcm_JointEnergy",
    "GLCM_ENTROPY_AVE": "original_glcm_JointEntropy",
    "GLCM_HOM1_AVE": "original_glcm_InverseVariance",  # Homogeneity1 ≈ InverseVariance
    "GLCM_DIS_AVE": "original_glcm_InverseVariance",  # Dissimilarity ≈ InverseVariance
    
    # ========== TEXTURE: GLDM (Nyxus) ↔ GLDM (PyRadiomics) ==========
    "GLDM_SDE": "original_gldm_SmallDependenceEmphasis",
    "GLDM_LDE": "original_gldm_LargeDependenceEmphasis",
    "GLDM_GLN": "original_gldm_GrayLevelNonUniformity",
    "GLDM_DN": "original_gldm_DependenceNonUniformity",
    "GLDM_DNN": "original_gldm_DependenceNonUniformityNormalized",
    "GLDM_GLV": "original_gldm_GrayLevelVariance",
    "GLDM_DV": "original_gldm_DependenceVariance",
    "GLDM_DE": "original_gldm_DependenceEntropy",
    "GLDM_LGLE": "original_gldm_LowGrayLevelEmphasis",
    "GLDM_HGLE": "original_gldm_HighGrayLevelEmphasis",
    "GLDM_SDLGLE": "original_gldm_SmallDependenceLowGrayLevelEmphasis",
    "GLDM_SDHGLE": "original_gldm_SmallDependenceHighGrayLevelEmphasis",
    "GLDM_LDLGLE": "original_gldm_LargeDependenceLowGrayLevelEmphasis",
    "GLDM_LDHGLE": "original_gldm_LargeDependenceHighGrayLevelEmphasis",
    
    # ========== TEXTURE: GLRLM (Nyxus) ↔ GLRLM (PyRadiomics) ==========
    # Note: Nyxus has directional GLRLM features (0, 45, 90, 135) and averaged versions
    # Mapping averaged versions to PyRadiomics
    "GLRLM_SRE_AVE": "original_glrlm_ShortRunEmphasis",
    "GLRLM_LRE_AVE": "original_glrlm_LongRunEmphasis",
    "GLRLM_GLN_AVE": "original_glrlm_GrayLevelNonUniformity",
    "GLRLM_GLNN_AVE": "original_glrlm_GrayLevelNonUniformityNormalized",
    "GLRLM_RLN_AVE": "original_glrlm_RunLengthNonUniformity",
    "GLRLM_RLNN_AVE": "original_glrlm_RunLengthNonUniformityNormalized",
    "GLRLM_RP_AVE": "original_glrlm_RunPercentage",
    "GLRLM_GLV_AVE": "original_glrlm_GrayLevelVariance",
    "GLRLM_RV_AVE": "original_glrlm_RunVariance",
    "GLRLM_RE_AVE": "original_glrlm_RunEntropy",
    "GLRLM_LGLRE_AVE": "original_glrlm_LowGrayLevelRunEmphasis",
    "GLRLM_HGLRE_AVE": "original_glrlm_HighGrayLevelRunEmphasis",
    "GLRLM_SRLGLE_AVE": "original_glrlm_ShortRunLowGrayLevelEmphasis",
    "GLRLM_SRHGLE_AVE": "original_glrlm_ShortRunHighGrayLevelEmphasis",
    "GLRLM_LRLGLE_AVE": "original_glrlm_LongRunLowGrayLevelEmphasis",
    "GLRLM_LRHGLE_AVE": "original_glrlm_LongRunHighGrayLevelEmphasis",
    
    # ========== TEXTURE: GLSZM (Nyxus) ↔ GLSZM (PyRadiomics) ==========
    "GLSZM_SAE": "original_glszm_SmallAreaEmphasis",
    "GLSZM_LAE": "original_glszm_LargeAreaEmphasis",
    "GLSZM_GLN": "original_glszm_GrayLevelNonUniformity",
    "GLSZM_GLNN": "original_glszm_GrayLevelNonUniformityNormalized",
    "GLSZM_SZN": "original_glszm_SizeZoneNonUniformity",
    "GLSZM_SZNN": "original_glszm_SizeZoneNonUniformityNormalized",
    "GLSZM_ZP": "original_glszm_ZonePercentage",
    "GLSZM_GLV": "original_glszm_GrayLevelVariance",
    "GLSZM_ZV": "original_glszm_ZoneVariance",
    "GLSZM_ZE": "original_glszm_ZoneEntropy",
    "GLSZM_LGLZE": "original_glszm_LowGrayLevelZoneEmphasis",
    "GLSZM_HGLZE": "original_glszm_HighGrayLevelZoneEmphasis",
    "GLSZM_SALGLE": "original_glszm_SmallAreaLowGrayLevelEmphasis",
    "GLSZM_SAHGLE": "original_glszm_SmallAreaHighGrayLevelEmphasis",
    "GLSZM_LALGLE": "original_glszm_LargeAreaLowGrayLevelEmphasis",
    "GLSZM_LAHGLE": "original_glszm_LargeAreaHighGrayLevelEmphasis",
    
    # ========== TEXTURE: NGTDM (Nyxus) ↔ NGTDM (PyRadiomics) ==========
    "NGTDM_COARSENESS": "original_ngtdm_Coarseness",
    "NGTDM_CONTRAST": "original_ngtdm_Contrast",
    "NGTDM_BUSYNESS": "original_ngtdm_Busyness",
    "NGTDM_COMPLEXITY": "original_ngtdm_Complexity",
    "NGTDM_STRENGTH": "original_ngtdm_Strength",
}

# Features that exist in Nyxus 2D but not in PyRadiomics 2D
NYXUS_ONLY_2D_FEATURES = [
    # Statistical
    "COV", "COVERED_IMAGE_INTENSITY_RANGE", "EXCESS_KURTOSIS", "HYPERFLATNESS",
    "HYPERSKEWNESS", "INTEGRATED_INTENSITY", "MEDIAN_ABSOLUTE_DEVIATION", "MODE",
    "P01", "P25", "P75", "P99", "QCOD", "ROBUST_MEAN", "STANDARD_DEVIATION",
    "STANDARD_DEVIATION_BIASED", "STANDARD_ERROR", "VARIANCE_BIASED", "UNIFORMITY_PIU",
    # Shape/Edge
    "PERIMETER", "DIAMETER_EQUAL_PERIMETER", "EDGE_MEAN_INTENSITY", "EDGE_STDDEV_INTENSITY",
    "EDGE_MAX_INTENSITY", "EDGE_MIN_INTENSITY", "EDGE_INTEGRATED_INTENSITY",
    # GLCM directional features (0, 45, 90, 135 degrees) - not averaged
    "GLCM_ASM_0", "GLCM_ASM_45", "GLCM_ASM_90", "GLCM_ASM_135",
    "GLCM_ACOR_0", "GLCM_ACOR_45", "GLCM_ACOR_90", "GLCM_ACOR_135",
    "GLCM_CLUPROM_0", "GLCM_CLUPROM_45", "GLCM_CLUPROM_90", "GLCM_CLUPROM_135",
    "GLCM_CLUSHADE_0", "GLCM_CLUSHADE_45", "GLCM_CLUSHADE_90", "GLCM_CLUSHADE_135",
    "GLCM_CLUTEND_0", "GLCM_CLUTEND_45", "GLCM_CLUTEND_90", "GLCM_CLUTEND_135",
    "GLCM_CONTRAST_0", "GLCM_CONTRAST_45", "GLCM_CONTRAST_90", "GLCM_CONTRAST_135",
    "GLCM_CORRELATION_0", "GLCM_CORRELATION_45", "GLCM_CORRELATION_90", "GLCM_CORRELATION_135",
    "GLCM_DIFAVE_0", "GLCM_DIFAVE_45", "GLCM_DIFAVE_90", "GLCM_DIFAVE_135",
    "GLCM_DIFENTRO_0", "GLCM_DIFENTRO_45", "GLCM_DIFENTRO_90", "GLCM_DIFENTRO_135",
    "GLCM_DIFVAR_0", "GLCM_DIFVAR_45", "GLCM_DIFVAR_90", "GLCM_DIFVAR_135",
    "GLCM_DIS_0", "GLCM_DIS_45", "GLCM_DIS_90", "GLCM_DIS_135",
    "GLCM_ENERGY_0", "GLCM_ENERGY_45", "GLCM_ENERGY_90", "GLCM_ENERGY_135",
    "GLCM_ENTROPY_0", "GLCM_ENTROPY_45", "GLCM_ENTROPY_90", "GLCM_ENTROPY_135",
    "GLCM_HOM1_0", "GLCM_HOM1_45", "GLCM_HOM1_90", "GLCM_HOM1_135",
    "GLCM_HOM2_0", "GLCM_HOM2_45", "GLCM_HOM2_90", "GLCM_HOM2_135",
    "GLCM_ID_0", "GLCM_ID_45", "GLCM_ID_90", "GLCM_ID_135",
    "GLCM_IDN_0", "GLCM_IDN_45", "GLCM_IDN_90", "GLCM_IDN_135",
    "GLCM_IDM_0", "GLCM_IDM_45", "GLCM_IDM_90", "GLCM_IDM_135",
    "GLCM_IDMN_0", "GLCM_IDMN_45", "GLCM_IDMN_90", "GLCM_IDMN_135",
    "GLCM_INFOMEAS1_0", "GLCM_INFOMEAS1_45", "GLCM_INFOMEAS1_90", "GLCM_INFOMEAS1_135",
    "GLCM_INFOMEAS2_0", "GLCM_INFOMEAS2_45", "GLCM_INFOMEAS2_90", "GLCM_INFOMEAS2_135",
    "GLCM_IV_0", "GLCM_IV_45", "GLCM_IV_90", "GLCM_IV_135",
    "GLCM_JAVE_0", "GLCM_JAVE_45", "GLCM_JAVE_90", "GLCM_JAVE_135",
    "GLCM_JE_0", "GLCM_JE_45", "GLCM_JE_90", "GLCM_JE_135",
    "GLCM_JMAX_0", "GLCM_JMAX_45", "GLCM_JMAX_90", "GLCM_JMAX_135",
    "GLCM_JVAR_0", "GLCM_JVAR_45", "GLCM_JVAR_90", "GLCM_JVAR_135",
    "GLCM_SUMAVERAGE_0", "GLCM_SUMAVERAGE_45", "GLCM_SUMAVERAGE_90", "GLCM_SUMAVERAGE_135",
    "GLCM_SUMENTROPY_0", "GLCM_SUMENTROPY_45", "GLCM_SUMENTROPY_90", "GLCM_SUMENTROPY_135",
    "GLCM_SUMVARIANCE_0", "GLCM_SUMVARIANCE_45", "GLCM_SUMVARIANCE_90", "GLCM_SUMVARIANCE_135",
    "GLCM_VARIANCE_0", "GLCM_VARIANCE_45", "GLCM_VARIANCE_90", "GLCM_VARIANCE_135",
    # GLRLM directional features (0, 45, 90, 135 degrees) - not averaged
    "GLRLM_SRE_0", "GLRLM_SRE_45", "GLRLM_SRE_90", "GLRLM_SRE_135",
    "GLRLM_LRE_0", "GLRLM_LRE_45", "GLRLM_LRE_90", "GLRLM_LRE_135",
    "GLRLM_GLN_0", "GLRLM_GLN_45", "GLRLM_GLN_90", "GLRLM_GLN_135",
    "GLRLM_GLNN_0", "GLRLM_GLNN_45", "GLRLM_GLNN_90", "GLRLM_GLNN_135",
    "GLRLM_RLN_0", "GLRLM_RLN_45", "GLRLM_RLN_90", "GLRLM_RLN_135",
    "GLRLM_RLNN_0", "GLRLM_RLNN_45", "GLRLM_RLNN_90", "GLRLM_RLNN_135",
    "GLRLM_RP_0", "GLRLM_RP_45", "GLRLM_RP_90", "GLRLM_RP_135",
    "GLRLM_GLV_0", "GLRLM_GLV_45", "GLRLM_GLV_90", "GLRLM_GLV_135",
    "GLRLM_RV_0", "GLRLM_RV_45", "GLRLM_RV_90", "GLRLM_RV_135",
    "GLRLM_RE_0", "GLRLM_RE_45", "GLRLM_RE_90", "GLRLM_RE_135",
    "GLRLM_LGLRE_0", "GLRLM_LGLRE_45", "GLRLM_LGLRE_90", "GLRLM_LGLRE_135",
    "GLRLM_HGLRE_0", "GLRLM_HGLRE_45", "GLRLM_HGLRE_90", "GLRLM_HGLRE_135",
    "GLRLM_SRLGLE_0", "GLRLM_SRLGLE_45", "GLRLM_SRLGLE_90", "GLRLM_SRLGLE_135",
    "GLRLM_SRHGLE_0", "GLRLM_SRHGLE_45", "GLRLM_SRHGLE_90", "GLRLM_SRHGLE_135",
    "GLRLM_LRLGLE_0", "GLRLM_LRLGLE_45", "GLRLM_LRLGLE_90", "GLRLM_LRLGLE_135",
    "GLRLM_LRHGLE_0", "GLRLM_LRHGLE_45", "GLRLM_LRHGLE_90", "GLRLM_LRHGLE_135",
    # NGLDM (not in PyRadiomics)
    "NGLDM_LDE", "NGLDM_HDE", "NGLDM_LGLCE", "NGLDM_HGLCE", "NGLDM_LDLGLE", "NGLDM_LDHGLE",
    "NGLDM_HDLGLE", "NGLDM_HDHGLE", "NGLDM_GLNU", "NGLDM_GLNUN", "NGLDM_DCNU", "NGLDM_DCNUN",
    "NGLDM_DCP", "NGLDM_GLM", "NGLDM_GLV", "NGLDM_DCM", "NGLDM_DCV", "NGLDM_DCENT", "NGLDM_DCENE",
    # Specialized 2D features (not in PyRadiomics)
    "FRAC_AT_D_0", "FRAC_AT_D_1", "FRAC_AT_D_2", "FRAC_AT_D_3", "FRAC_AT_D_4", "FRAC_AT_D_5", "FRAC_AT_D_6", "FRAC_AT_D_7",
    "GABOR_0", "GABOR_1", "GABOR_2", "GABOR_3",
    "MEAN_FRAC_0", "MEAN_FRAC_1", "MEAN_FRAC_2", "MEAN_FRAC_3", "MEAN_FRAC_4", "MEAN_FRAC_5", "MEAN_FRAC_6", "MEAN_FRAC_7",
    "RADIAL_CV_0", "RADIAL_CV_1", "RADIAL_CV_2", "RADIAL_CV_3", "RADIAL_CV_4", "RADIAL_CV_5", "RADIAL_CV_6", "RADIAL_CV_7",
    # Zernike 2D
    "ZERNIKE2D_Z0", "ZERNIKE2D_Z1", "ZERNIKE2D_Z2", "ZERNIKE2D_Z3", "ZERNIKE2D_Z4", "ZERNIKE2D_Z5", "ZERNIKE2D_Z6", "ZERNIKE2D_Z7",
    "ZERNIKE2D_Z8", "ZERNIKE2D_Z9", "ZERNIKE2D_Z10", "ZERNIKE2D_Z11", "ZERNIKE2D_Z12", "ZERNIKE2D_Z13", "ZERNIKE2D_Z14", "ZERNIKE2D_Z15",
    "ZERNIKE2D_Z16", "ZERNIKE2D_Z17", "ZERNIKE2D_Z18", "ZERNIKE2D_Z19", "ZERNIKE2D_Z20", "ZERNIKE2D_Z21", "ZERNIKE2D_Z22", "ZERNIKE2D_Z23",
    "ZERNIKE2D_Z24", "ZERNIKE2D_Z25", "ZERNIKE2D_Z26", "ZERNIKE2D_Z27", "ZERNIKE2D_Z28", "ZERNIKE2D_Z29",
    # Image Moments
    "IMOM_RM_00", "IMOM_RM_01", "IMOM_RM_02", "IMOM_RM_03", "IMOM_RM_10", "IMOM_RM_11", "IMOM_RM_12", "IMOM_RM_13",
    "IMOM_RM_20", "IMOM_RM_21", "IMOM_RM_22", "IMOM_RM_23", "IMOM_RM_30",
    "IMOM_CM_00", "IMOM_CM_01", "IMOM_CM_02", "IMOM_CM_03", "IMOM_CM_10", "IMOM_CM_11", "IMOM_CM_12", "IMOM_CM_13",
    "IMOM_CM_20", "IMOM_CM_21", "IMOM_CM_22", "IMOM_CM_23", "IMOM_CM_30", "IMOM_CM_31", "IMOM_CM_32", "IMOM_CM_33",
    "IMOM_NRM_00", "IMOM_NRM_01", "IMOM_NRM_02", "IMOM_NRM_03", "IMOM_NRM_10", "IMOM_NRM_11", "IMOM_NRM_12", "IMOM_NRM_13",
    "IMOM_NRM_20", "IMOM_NRM_21", "IMOM_NRM_22", "IMOM_NRM_23", "IMOM_NRM_30", "IMOM_NRM_31", "IMOM_NRM_32", "IMOM_NRM_33",
    "IMOM_NCM_02", "IMOM_NCM_03", "IMOM_NCM_11", "IMOM_NCM_12", "IMOM_NCM_20", "IMOM_NCM_21", "IMOM_NCM_30",
    "IMOM_HU1", "IMOM_HU2", "IMOM_HU3", "IMOM_HU4", "IMOM_HU5", "IMOM_HU6", "IMOM_HU7",
    "IMOM_WRM_00", "IMOM_WRM_01", "IMOM_WRM_02", "IMOM_WRM_03", "IMOM_WRM_10", "IMOM_WRM_11", "IMOM_WRM_12",
    "IMOM_WRM_20", "IMOM_WRM_21", "IMOM_WRM_30",
    "IMOM_WCM_02", "IMOM_WCM_03", "IMOM_WCM_11", "IMOM_WCM_12", "IMOM_WCM_20", "IMOM_WCM_21", "IMOM_WCM_30",
    "IMOM_WNCM_02", "IMOM_WNCM_03", "IMOM_WNCM_11", "IMOM_WNCM_12", "IMOM_WNCM_20", "IMOM_WNCM_21", "IMOM_WNCM_30",
    "IMOM_WHU1", "IMOM_WHU2", "IMOM_WHU3", "IMOM_WHU4", "IMOM_WHU5", "IMOM_WHU6", "IMOM_WHU7",
]

# Features that exist in PyRadiomics 2D but not in Nyxus 2D
PYRADIOMICS_ONLY_2D_FEATURES = [
    "original_shape_Elongation",
    "original_shape_Flatness",
    "original_shape_LeastAxisLength",
    "original_shape_MajorAxisLength",
    "original_shape_MinorAxisLength",
    "original_shape_Maximum2DDiameterColumn",
    "original_shape_Maximum2DDiameterRow",
    "original_shape_Maximum2DDiameterSlice",
    "original_shape_Maximum3DDiameter",
    "original_shape_MeshVolume",
    "original_shape_Sphericity",
    "original_shape_SurfaceArea",
    "original_shape_SurfaceVolumeRatio",
    "original_shape_VoxelVolume",
]