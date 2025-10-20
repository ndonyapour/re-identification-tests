import pandas as pd
import numpy as np

# 1. Read the full CSV (all rows + all columns)
df = pd.read_csv("./data/Nyxus_Texture_features.csv")

# 2. Collect feature columns (f0...f162)
feature_cols = [col for col in df.columns if col.startswith("f")]


labels = df["label"].values

features = []
for i in range(len(df)):
    features.append(df[feature_cols].values[i, :])


print(labels.shape)
print(np.array(features).shape)