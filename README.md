# Random Forest Variants — From Scratch

A family of from-scratch decision tree ensemble classifiers built entirely with NumPy and pandas — no ML libraries. Includes four variants progressing from pure randomness to optimal splitting with sklearn-level features.

## Project Structure


```
├── totallyrandomforest.py              # Totally Random Forest (random splits, no Gini)
├── ExtraTrees.py                       # Extra Trees (random splits, best Gini)
├── ExtraTreesStochasticLookAhead.py    # Extra Trees + Stochastic Lookahead
├── RandomForest.py                     # Random Forest (optimal thresholds, full features)
├── data_loading.py                     # Generate 5-feature synthetic dataset
├── data_loadingEF&RF.py               # Generate 20-feature synthetic dataset
├── rf_data.csv                         # 500 samples, 5 features
├── rf_dataEF&RF.csv                     # 1000 samples, 20 features
├── randomforestbetterpng.png            #main one used         
├── random_forest_dashboard.png         # Model performance visualization
└── README.md
```

## Four Variants

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

### 4. Random Forest — The Main Event

The classic Random Forest algorithm with optimal threshold search, multiple regularization features, and sklearn-level parameterization. This is the real thing.

- **Split selection**: `max_features` random features × ALL unique thresholds × best Gini wins
- **Regularization**: min impurity decrease, max leaf nodes, max depth, min samples
- **Multiclass Gini**: Works for any number of classes
- **Strength**: Most accurate variant, fully featured, production-ready
- **Accuracy**: ~83–85% on 20-feature dataset (with honest train-test gap)

#### Random Forest Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_trees` | Number of trees in the forest | 100 |
| `max_depth` | Maximum depth of each tree | 10 |
| `min_samples` | Minimum samples to allow a split | 3 |
| `max_features` | Number of features to consider per split | 5 (or sqrt) |
| `min_impurity_decrease` | Minimum Gini improvement to split | 0.0 |
| `max_leaf_nodes` | Maximum leaf nodes per tree | ∞ |

## Random Forest vs Extra Trees

| Feature | Extra Trees | Random Forest |
|---------|-------------|---------------|
| Feature selection | K random | K random |
| Threshold selection | Random (K tries) | All unique values (optimal) |
| Training speed | Faster | Slower |
| Accuracy | ~85% | ~85% (with regularization) |
| Overfitting | More prone | Less prone (more regularization) |
| Min impurity decrease | No | Yes |
| Max leaf nodes | No | Yes |

## How It Works

### Gini Impurity (Multiclass)

Measures how "mixed" a set of labels is. 0 = perfectly pure, 1 = maximally mixed.

```
Gini = 1 - Σ(p_c)²
```

### Random Forest Split Selection

```
1. Pick max_features random features
2. For each feature, try EVERY unique value as a threshold
3. Compute weighted Gini for each split
4. Pick the split with the lowest weighted Gini
5. Check if impurity decrease > min_impurity_decrease
6. Check if leaf count < max_leaf_nodes
7. Recurse on left and right children
```

### Stochastic Lookahead (Extra Trees variant)

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

Each tree votes. Majority class wins via `np.bincount(y_pred).argmax()`.

## Usage

```bash
# Generate datasets
python dataloading.py
python dataloading20.py

# Run any variant
python totallyrandomforest.py
python extratrees.py
python extratreeslookahead.py
python randomforest.py
```

```python
from randomforest import RandomForest
import pandas as pd
import numpy as np

df = pd.read_csv('rf_dataEF&RF.csv')
X = df[[col for col in df.columns if col != 'target']].values
y = df['target'].values

idx = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

model = RandomForest(
    n_trees=100, max_depth=10, min_samples=3, max_features=5,
    min_impurity_decrease=0.0, max_leaf_nodes=float('inf')
)
model.fit(X_train, y_train)
print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
```

## Model Performance Dashboard

The Random Forest includes a three-panel dark-theme visualization:

- **Class Distribution**: Bar chart of training set labels
- **Accuracy vs Ensemble Size**: Train/test accuracy curve showing convergence and diminishing returns after ~30 trees
- **Model Summary**: Full parameter list, train/test accuracy, and gap analysis

Typical results on 20-feature data: train accuracy ~95%, test accuracy ~83%, reflecting honest overfitting from deep trees on a modest dataset.

## Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

scikit-learn is used only for data generation (`make_classification`). All tree algorithms use exclusively NumPy and pandas.

## Implementation Notes

- Bootstrap sampling (random sampling with replacement) for each tree
- Trees are nested dictionaries with `feature`, `threshold`, `left`, `right` keys
- Leaves store `class` — the majority class prediction
- `np.bincount(y).argmax()` for majority voting
- Multiclass Gini impurity: works for any number of classes
- `leaf_count` list hack to track leaf node count across recursive calls
- All variants share the same `predict_one`, `predict_forest`, and `score` methods

