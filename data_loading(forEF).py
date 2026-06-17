import numpy as np
import pandas as pd 
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples = 1000,
    n_features = 20,
    n_informative = 14, 
    n_redundant= 3, 
    n_classes=2,
    random_state=42
)

columns = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=columns)
df['target'] = y
df.to_csv('rf_dataEF.csv', index=False)

print(f"Saved {len(df)} samples with {X.shape[1]} features")
print(f"Class balance: {np.bincount(y)}")
print(df.head())