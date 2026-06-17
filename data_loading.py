import numpy as np
import pandas as pd 
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples = 500,
    n_features = 5,
    n_informative=3, 
    n_redundant=1, 
    n_classes=2,
    random_state=42
)

columns = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=columns)
df['target'] = y
df.to_csv('rf_data.csv', index=False)

print(f"Saved {len(df)} samples with {X.shape[1]} features")
print(f"Class balance: {np.bincount(y)}")
print(df.head())