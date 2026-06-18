# Extra-Trees with Stochastic Lookahead

A family of from-scratch decision tree ensemble classifiers built entirely with NumPy and pandas — no ML libraries. Includes three variants progressing from pure randomness to greedy optimization to one-step lookahead.

## Project Structure

```
├── totallyrandomforest.py          # Totally Random Forest (random splits, no Gini)
├── extratrees.py                   # Extra Trees (random splits, pick best by Gini)
├── extratreeslookahead.py          # Extra Trees + Stochastic Lookahead
├── dataloading.py                  # Generate 5-feature synthetic dataset
├── dataloading20.py                 # Generate 20-feature synthetic dataset
├── rf_data.csv                     # 500 samples, 5 features
├── rf_dataEF.csv                   # 1000 samples, 20 features
└── README.md
```

## Three Variants

### 1. Totally Random Forest

Every split is completely random — random feature, random threshold. No Gini impurity. No optimization. Pure chaos.

- **Split selection**: One random feature, one random threshold
- **Strength**: Fastest to train, demonstrates power of voting alone
- **Accuracy**: ~83% on 5-feature dataset

### 2. Extra Trees (Extremely Randomized Trees)

Tries `max_features` random features, picks random thresholds, and chooses the split with the lowest weighted Gini impurity.

- **Split selection**: `max_features` random features × random thresholds, best Gini wins
- **Strength**: Balances randomness with quality — the classic Extra Trees algorithm
- **Accuracy**: ~85% on 5-feature dataset

### 3. Extra Trees with Stochastic Lookahead

Before committing to a split, evaluates what happens ONE LEVEL DEEPER. For each candidate split, tries random splits on both children and picks the parent whose future looks most promising.

- **Split selection**: Greedy at depth 1, stochastic evaluation at depth 2
- **Strength**: Avoids splits that look good now but lead to impure grandchildren
- **Accuracy**: ~86% on 5-feature dataset

## How It Works

### Gini Impurity

Measures how "mixed" a set of labels is. 0 = perfectly pure (all same class), 1 = maximally mixed.

```
Gini = 1 - Σ(p_c)²
```

### Extra Trees Split Selection

```
1. Pick max_features random features
2. For each feature, pick a random threshold between min and max
3. Compute weighted Gini for that split
4. Pick the split with the lowest weighted Gini
5. Recurse on left and right children
```

### Stochastic Lookahead

```
1. Generate candidate splits as in Extra Trees
2. For each candidate:
   a. Generate random splits on the LEFT child, measure best Gini
   b. Generate random splits on the RIGHT child, measure best Gini
   c. Compute weighted score: (n_left × best_left + n_right × best_right) / n
3. Pick the candidate with the lowest lookahead score
4. Recurse on left and right children (standard Extra Trees from here)
```

### Forest Prediction

Each tree votes. Majority class wins.

## Usage

```bash
# Generate datasets
python dataloading.py
python dataloading20.py

# Run any variant
python totallyrandomforest.py
python extratrees.py
python extratreeslookahead.py
```

```python
from extratreeslookahead import ExtremelyRandomForestFuture
import pandas as pd
import numpy as np

df = pd.read_csv('rf_dataEF.csv')
X = df[[col for col in df.columns if col != 'target']].values
y = df['target'].values

# Train/test split
idx = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

model = ExtremelyRandomForestFuture(
    n_trees=100, max_depth=10, min_samples=3, max_features=5
)
model.fit(X_train, y_train)
print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_trees` | Number of trees in the forest | 100 |
| `max_depth` | Maximum depth of each tree | 10 |
| `min_samples` | Minimum samples to allow a split | 3 |
| `max_features` | Number of features to consider per split | 5 (or sqrt) |

## Accuracy vs Number of Trees

Both the Extra Trees and Lookahead variants include plotting code to visualize how accuracy scales with forest size. Expect diminishing returns after ~50 trees.

## Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

scikit-learn is only used for data generation (`make_classification`). The tree algorithms themselves use only NumPy and pandas.

## Implementation Notes

- Bootstrap sampling (random sampling with replacement) for each tree
- Trees are nested dictionaries with `feature`, `threshold`, `left`, `right` keys
- Leaves store `class` — the majority class prediction
- `np.bincount(y).argmax()` for majority voting
- Gini impurity for multiclass: works for any number of classes
